#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

categories = [
    "missed opportunity", "false positive", "discovery", "icc shielding",
    "icc & predictor agreement"
]

data = [3, 8, 10, 2, 77]

#cmap = plt.cm.Blues
#colors = cmap(np.linspace(0.2, 0.9, len(data)))
colors = ['skyblue','red','green','lightgreen','lightblue']


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.0f}%\n".format(pct, absolute)


wedges, texts, autotexts = ax.pie(
    data,
    pctdistance=1.1,
    autopct=lambda pct: func(pct, data),
    textprops=dict(color="black"),
    colors=colors,
    radius=1.8,
    startangle=180)

ax.legend(
    wedges,
    categories,
    title="",
    loc="lower center",
    #bbox_to_anchor=(1, 1, 0, 1),
    bbox_to_anchor=(0.5,-0.5),
    ncol=3,
    prop={'size': 8})

# ax.set_title("A pie")

plt.setp(autotexts, size=10, weight="bold")

# output
plt.savefig("pie.pdf", bbox_inches='tight')
# plt.show()
