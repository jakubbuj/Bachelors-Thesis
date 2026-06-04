# rq1_analysis.py
# RQ1: how do path properties depend on the resistance distribution?
#
# produces 3 figures:
#   rq1_scatter_overview.png      - tortuosity and resistance vs mean and variance
#   rq1_phase_transition.png      - bimodal sweep showing resistance collapse and tortuosity peak
#   rq1_fixed_mean_variance.png   - fixed mean family showing independent effect of variance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("/Users/jakubb/Desktop/Thesis/FINAL/data100.csv") # change the path to yours

def get_family(label):
    if label.startswith("bimodal"):   return "bimodal"
    if label.startswith("uniform"):   return "uniform"
    if label.startswith("obstacles"): return "obstacles"
    if label.startswith("low"):       return "low_resistance"
    if label.startswith("exp"):       return "exponential"
    if label.startswith("fixed"):     return "fixed_mean"
    return "random"

df["family"] = df["dist_label"].apply(get_family)

COLORS = {
    "bimodal":        "#e41a1c",
    "uniform":        "#377eb8",
    "obstacles":      "#ff7f00",
    "low_resistance": "#4daf4a",
    "exponential":    "#984ea3",
    "fixed_mean":     "#a65628",
    "random":         "#999999",
}

legend_handles = [mpatches.Patch(color=c, label=f) for f, c in COLORS.items()]
plt.rcParams.update({"font.size": 12, "figure.dpi": 130})

# scatter overview
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RQ1 — path properties vs distribution parameters\n"
             "(each point = one distribution, colour = family)", fontsize=14)

plots = [
    (axes[0, 0], "dist_mean", "mean_tortuosity", "mean μ",      "tortuosity τ"),
    (axes[0, 1], "dist_var",  "mean_tortuosity", "variance σ²", "tortuosity τ"),
    (axes[1, 0], "dist_mean", "mean_resistance", "mean μ",      "total resistance R"),
    (axes[1, 1], "dist_var",  "mean_resistance", "variance σ²", "total resistance R"),
]

for ax, xcol, ycol, xlabel, ylabel in plots:
    for fam, color in COLORS.items():
        sub = df[df["family"] == fam]
        ax.scatter(sub[xcol], sub[ycol], c=color, alpha=0.7, s=40)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

axes[0, 0].legend(handles=legend_handles, fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig("rq1_scatter_overview.png", bbox_inches="tight")
plt.show()
print("saved: rq1_scatter_overview.png")


# bimodal phase transition
bim = df[df["family"] == "bimodal"].copy()
bim["p0_val"] = bim["p0"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RQ1 — phase transition in the bimodal distribution\n"
             "p₀ = probability of free edge (R=0)", fontsize=13)

axes[0].plot(bim["p0_val"], bim["mean_resistance"], "o-", color="#e41a1c",
             linewidth=2, markersize=5)
axes[0].fill_between(bim["p0_val"],
                     bim["mean_resistance"] - bim["std_resistance"],
                     bim["mean_resistance"] + bim["std_resistance"],
                     alpha=0.2, color="#e41a1c")
axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="p₀ = 0.5")
axes[0].set_xlabel("p₀")
axes[0].set_ylabel("mean total resistance")
axes[0].set_title("resistance vs p₀")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(bim["p0_val"], bim["mean_tortuosity"], "o-", color="#377eb8",
             linewidth=2, markersize=5)
axes[1].fill_between(bim["p0_val"],
                     bim["mean_tortuosity"] - bim["std_tortuosity"],
                     bim["mean_tortuosity"] + bim["std_tortuosity"],
                     alpha=0.2, color="#377eb8")
axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="p₀ = 0.5")
axes[1].set_xlabel("p₀")
axes[1].set_ylabel("mean tortuosity τ")
axes[1].set_title("tortuosity vs p₀")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rq1_phase_transition.png", bbox_inches="tight")
plt.show()
print("saved: rq1_phase_transition.png")


# fixed mean family
fix = df[df["family"] == "fixed_mean"].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RQ1 — fixed mean (μ=5), varying variance", fontsize=13)

axes[0].plot(fix["dist_var"], fix["mean_tortuosity"], "o-", color="#a65628",
             linewidth=2, markersize=6)
axes[0].set_xlabel("variance σ²")
axes[0].set_ylabel("mean tortuosity τ")
axes[0].set_title("tortuosity vs σ²")
axes[0].grid(True, alpha=0.3)

axes[1].plot(fix["dist_var"], fix["mean_resistance"], "o-", color="#984ea3",
             linewidth=2, markersize=6)
axes[1].set_xlabel("variance σ²")
axes[1].set_ylabel("mean total resistance")
axes[1].set_title("resistance vs σ²")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rq1_fixed_mean_variance.png", bbox_inches="tight")
plt.show()
print("saved: rq1_fixed_mean_variance.png")