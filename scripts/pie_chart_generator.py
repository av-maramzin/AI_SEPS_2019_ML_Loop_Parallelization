#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

categories = [
    "missed opportunity", "false positive", "discovery", "icc shielding",
    "correct 0", "correct 1"
]

data = [2, 3, 8, 10, 22, 55]

cmap = plt.cm.Blues
colors = cmap(np.linspace(0.2, 0.9, len(data)))


def func(pct, allvals):
    absolute = int(pct / 100. * np.sum(allvals))
    return "{:.0f}%\n".format(pct, absolute)


wedges, texts, autotexts = ax.pie(
    data,
    pctdistance=1.3,
    autopct=lambda pct: func(pct, data),
    textprops=dict(color="black"),
    colors=colors,
    radius=1.8,
    startangle=180)

ax.legend(
    wedges,
    categories,
    title="",
    loc="center left",
    bbox_to_anchor=(1, 1, 0, 1),
    prop={'size': 8})

# ax.set_title("A pie")

plt.setp(autotexts, size=10, weight="bold")

# output
plt.savefig("pie.pdf", bbox_inches='tight')
# plt.show()
