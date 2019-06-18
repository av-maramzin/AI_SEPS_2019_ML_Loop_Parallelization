  for (it = 1; it <= NITER; it++) {
    //---------------------------------------------------------------------
    // The call to the conjugate gradient routine:
    //---------------------------------------------------------------------
    if (timeron) timer_start(T_conj_grad);
    conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm);
    if (timeron) timer_stop(T_conj_grad);

    //---------------------------------------------------------------------
    // zeta = shift + 1/(x.z)
    // So, first: (x.z)
    // Also, find norm of z
    // So, first: (z.z)
    //---------------------------------------------------------------------
    norm_temp1 = 0.0;
    norm_temp2 = 0.0;
    #pragma omp parallel for default(shared) private(j) \
                             reduction(+:norm_temp1,norm_temp2)
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      norm_temp1 = norm_temp1 + x[j]*z[j];
      norm_temp2 = norm_temp2 + z[j]*z[j];
    }

    norm_temp2 = 1.0 / sqrt(norm_temp2);

    zeta = SHIFT + 1.0 / norm_temp1;
    if (it == 1) 
      printf("\n   iteration           ||r||                 zeta\n");
    printf("    %5d       %20.14E%20.13f\n", it, rnorm, zeta);

    //---------------------------------------------------------------------
    // Normalize z to obtain x
    //---------------------------------------------------------------------
    #pragma omp parallel for default(shared) private(j)
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      x[j] = norm_temp2 * z[j];
    }
  } // end of main iter inv pow meth

//---------------------------------------------------------------------
// Floaging point arrays here are named as in NPB1 spec discussion of 
// CG algorithm
//---------------------------------------------------------------------
static void conj_grad(int colidx[],
                      int rowstr[],
                      double x[],
                      double z[],
                      double a[],
                      double p[],
                      double q[],
                      double r[],
                      double *rnorm)
{
  int j, k;
  int cgit, cgitmax = 25;
  double d, sum, rho, rho0, alpha, beta, suml;

  rho = 0.0;
  sum = 0.0;

  #pragma omp parallel default(shared) private(j,k,cgit,suml,alpha,beta) \
                                       shared(d,rho0,rho,sum)
  {
  //---------------------------------------------------------------------
  // Initialize the CG algorithm:
  //---------------------------------------------------------------------
  #pragma omp for
  for (j = 0; j < naa+1; j++) {
    q[j] = 0.0;
    z[j] = 0.0;
    r[j] = x[j];
    p[j] = r[j];
  }

  //---------------------------------------------------------------------
  // rho = r.r
  // Now, obtain the norm of r: First, sum squares of r elements locally...
  //---------------------------------------------------------------------
  #pragma omp for reduction(+:rho)
  for (j = 0; j < lastcol - firstcol + 1; j++) {
    rho = rho + r[j]*r[j];
  }

  //---------------------------------------------------------------------
  //---->
  // The conj grad iteration loop
  //---->
  //---------------------------------------------------------------------
  for (cgit = 1; cgit <= cgitmax; cgit++) {
    #pragma omp master
    {
      //---------------------------------------------------------------------
      // Save a temporary of rho and initialize reduction variables
      //---------------------------------------------------------------------
      rho0 = rho;
      d = 0.0;
      rho = 0.0;
    }
    #pragma omp barrier

    //---------------------------------------------------------------------
    // q = A.p
    // The partition submatrix-vector multiply: use workspace w
    //---------------------------------------------------------------------
    //
    // NOTE: this version of the multiply is actually (slightly: maybe %5) 
    //       faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
    //       below.   On the Cray t3d, the reverse is true, i.e., the 
    //       unrolled-by-two version is some 10% faster.  
    //       The unrolled-by-8 version below is significantly faster
    //       on the Cray t3d - overall speed of code is 1.5 times faster.

    #pragma omp for
    for (j = 0; j < lastrow - firstrow + 1; j++) {
      suml = 0.0;
      for (k = rowstr[j]; k < rowstr[j+1]; k++) {
        suml = suml + a[k]*p[colidx[k]];
      }
      q[j] = suml;
    }

    /*
    for (j = 0; j < lastrow - firstrow + 1; j++) {
      int i = rowstr[j];
      int iresidue = (rowstr[j+1] - i) % 2;
      double sum1 = 0.0;
      double sum2 = 0.0;
      if (iresidue == 1)
        sum1 = sum1 + a[i]*p[colidx[i]];
      for (k = i + iresidue; k <= rowstr[j+1] - 2; k += 2) {
        sum1 = sum1 + a[k]  *p[colidx[k]];
        sum2 = sum2 + a[k+1]*p[colidx[k+1]];
      }
      q[j] = sum1 + sum2;
    }
    */

    /*
    for (j = 0; j < lastrow - firstrow + 1; j++) {
      int i = rowstr[j]; 
      int iresidue = (rowstr[j+1] - i) % 8;
      suml = 0.0;
      for (k = i; k <= i + iresidue - 1; k++) {
        suml = suml + a[k]*p[colidx[k]];
      }
      for (k = i + iresidue; k <= rowstr[j+1] - 8; k += 8) {
        suml = suml + a[k  ]*p[colidx[k  ]]
                  + a[k+1]*p[colidx[k+1]]
                  + a[k+2]*p[colidx[k+2]]
                  + a[k+3]*p[colidx[k+3]]
                  + a[k+4]*p[colidx[k+4]]
                  + a[k+5]*p[colidx[k+5]]
                  + a[k+6]*p[colidx[k+6]]
                  + a[k+7]*p[colidx[k+7]];
      }
      q[j] = suml;
    }
    */

    //---------------------------------------------------------------------
    // Obtain p.q
    //---------------------------------------------------------------------
    #pragma omp for reduction(+:d)
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      d = d + p[j]*q[j];
    }

    //---------------------------------------------------------------------
    // Obtain alpha = rho / (p.q)
    //---------------------------------------------------------------------
    alpha = rho0 / d;

    //---------------------------------------------------------------------
    // Obtain z = z + alpha*p
    // and    r = r - alpha*q
    //---------------------------------------------------------------------
    #pragma omp for reduction(+:rho)
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      z[j] = z[j] + alpha*p[j];
      r[j] = r[j] - alpha*q[j];
            
      //---------------------------------------------------------------------
      // rho = r.r
      // Now, obtain the norm of r: First, sum squares of r elements locally..
      //---------------------------------------------------------------------
      rho = rho + r[j]*r[j];
    }

    //---------------------------------------------------------------------
    // Obtain beta:
    //---------------------------------------------------------------------
    beta = rho / rho0;

    //---------------------------------------------------------------------
    // p = r + beta*p
    //---------------------------------------------------------------------
    #pragma omp for
    for (j = 0; j < lastcol - firstcol + 1; j++) {
      p[j] = r[j] + beta*p[j];
    }
  } // end of do cgit=1,cgitmax

  //---------------------------------------------------------------------
  // Compute residual norm explicitly:  ||r|| = ||x - A.z||
  // First, form A.z
  // The partition submatrix-vector multiply
  //---------------------------------------------------------------------
  #pragma omp for
  for (j = 0; j < lastrow - firstrow + 1; j++) {
    suml = 0.0;
    for (k = rowstr[j]; k < rowstr[j+1]; k++) {
      suml = suml + a[k]*z[colidx[k]];
    }
    r[j] = suml;
  }

  //---------------------------------------------------------------------
  // At this point, r contains A.z
  //---------------------------------------------------------------------
  #pragma omp for reduction(+:sum) nowait
  for (j = 0; j < lastcol-firstcol+1; j++) {
    suml = x[j] - r[j];
    sum  = sum + suml*suml;
  }
  }

  *rnorm = sqrt(sum);
}
