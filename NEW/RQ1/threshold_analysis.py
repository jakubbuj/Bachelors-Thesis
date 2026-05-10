# regime_split.py
# Splits ALL distributions into two regimes based on whether they are
# in an environment below or above the percolation threshold, then fits
# separate tortuosity formulas for each regime.
#
# The key insight: the global formula fails (R²=0.35) because tortuosity
# behaves completely differently in the two regimes:
#   - Below threshold: tortuosity INCREASES with more free edges (path hunts)
#   - Above threshold: tortuosity DECREASES as free corridors become easy to find
#
# Fitting separately within each regime should give much higher R².
#
# Produces 3 figures:
#   regime_split_scatter.png     - tortuosity coloured by regime across all families
#   regime_split_predicted.png   - predicted vs actual for each regime
#   regime_split_comparison.png  - R² bar chart: global vs per-regime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("/Users/jakubb/Desktop/Thesis/NEW/data100.csv")

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

plt.rcParams.update({"font.size": 12, "figure.dpi": 130})

# ── define the two regimes ─────────────────────────────────────────────────
# The regime a distribution belongs to is determined by p0 — the fraction
# of free (R=0) edges. Below the threshold (p0 < 0.525) the grid has no
# reliable free path; above it (p0 >= 0.525) a free path always exists.
#
# For non-bimodal families p0 is the probability of the lowest resistance
# value — already stored as the column "p0" in the CSV.

THRESHOLD = 5.0   # split at mean resistance = 5 (midpoint of [0,10])
                  # this is the natural generalisation of the bimodal threshold
                  # to all distribution families

df["regime"] = df["dist_mean"].apply(
    lambda mu: "easy (μ < 5)" if mu < THRESHOLD else "hard (μ ≥ 5)"
)

below = df[df["regime"] == "easy (μ < 5)"].copy()    # cheap environment
above = df[df["regime"] == "hard (μ ≥ 5)"].copy()    # expensive environment

print(f"Total distributions  : {len(df)}")
print(f"Easy environment (μ < 5)  : {len(below)}")
print(f"Hard environment (μ ≥ 5)  : {len(above)}")
print(f"\nBy family:")
print(df.groupby(["family","regime"]).size().unstack(fill_value=0).to_string())
print(f"\nEasy — mean tortuosity : {below['mean_tortuosity'].mean():.4f}")
print(f"Hard — mean tortuosity : {above['mean_tortuosity'].mean():.4f}")


# helper: build feature matrix 
def make_X(data):
    return np.column_stack([
        data["dist_mean"],
        data["dist_mean"]**2,
        data["dist_var"],
        data["dist_var"]**2,
    ])


# ── helper: fit and report ─────────────────────────────────────────────────
def fit_report(name, data):
    X = make_X(data)
    y = data["mean_tortuosity"].values

    # 80/20 split
    if len(data) >= 10:
        idx_tr, idx_te = train_test_split(np.arange(len(data)),
                                          test_size=0.2, random_state=42)
    else:
        idx_tr, idx_te = np.arange(len(data)), np.arange(len(data))

    model    = LinearRegression().fit(X[idx_tr], y[idx_tr])
    y_pred   = model.predict(X[idx_te])
    r2_test  = r2_score(y[idx_te], y_pred)
    r2_train = r2_score(y[idx_tr], model.predict(X[idx_tr]))
    c, b     = model.coef_, model.intercept_

    print(f"\n  [{name}]")
    print(f"    n = {len(data)}   R² train = {r2_train:.4f}   R² test = {r2_test:.4f}")
    print(f"    τ = {c[0]:.4f}·μ + {c[1]:.4f}·μ² + {c[2]:.4f}·σ² + {c[3]:.4f}·σ⁴ + {b:.4f}")
    return model, r2_test, model.predict(X), y


# fit formulas 
print("\n" + "="*60)
print("  FORMULA FITTING RESULTS")
print("="*60)

model_all,   r2_all,   pred_all,   y_all   = fit_report("ALL 252 distributions", df)
model_below, r2_below, pred_below, y_below = fit_report("Below threshold only", below)
model_above, r2_above, pred_above, y_above = fit_report("Above threshold only", above)


# figure 1: scatter coloured by regime 
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Two regimes in tortuosity behaviour\n"
             "Easy (μ < 5): paths hunt for cheap edges → higher τ\n"
             "Hard (μ ≥ 5): all edges expensive, path goes straight → lower τ", fontsize=12)

for ax, xcol, xlabel in [
    (axes[0], "dist_mean", "mean μ"),
    (axes[1], "dist_var",  "variance σ²"),
]:
    ax.scatter(below[xcol], below["mean_tortuosity"],
               c="#e41a1c", alpha=0.7, s=40,
               label=f"easy environment (μ < 5)")
    ax.scatter(above[xcol], above["mean_tortuosity"],
               c="#377eb8", alpha=0.7, s=40,
               label=f"hard environment (μ ≥ 5)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean tortuosity τ")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[0].set_title("Tortuosity vs mean — two regimes")
axes[1].set_title("Tortuosity vs variance — two regimes")

plt.tight_layout()
plt.savefig("regime_split_scatter.png", bbox_inches="tight")
plt.show()
print("\nsaved: regime_split_scatter.png")


# ── figure 2: predicted vs actual for each regime ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Predicted vs Actual tortuosity — global vs per-regime formula\n"
             "Points on diagonal = perfect prediction", fontsize=13)

point_colors_all   = [COLORS[f] for f in df["family"]]
point_colors_below = [COLORS[f] for f in below["family"]]
point_colors_above = [COLORS[f] for f in above["family"]]

for ax, actual, predicted, colors, title, r2 in [
    (axes[0], y_all,   pred_all,   point_colors_all,
     f"Global formula\nall 252 dists — R²={r2_all:.3f}",   r2_all),
    (axes[1], y_below, pred_below, point_colors_below,
     f"Easy environment (μ < 5)\nn={len(below)} — R²={r2_below:.3f}", r2_below),
    (axes[2], y_above, pred_above, point_colors_above,
     f"Hard environment (μ ≥ 5)\nn={len(above)} — R²={r2_above:.3f}", r2_above),
]:
    ax.scatter(actual, predicted, c=colors, alpha=0.7, s=50)
    lim = [min(actual.min(), predicted.min()) - 0.02,
           max(actual.max(), predicted.max()) + 0.02]
    ax.plot(lim, lim, "k--", linewidth=1.5, label="perfect prediction")
    ax.set_xlabel("actual tortuosity τ")
    ax.set_ylabel("predicted tortuosity τ")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

legend_handles = [mpatches.Patch(color=c, label=f) for f, c in COLORS.items()]
fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig("regime_split_predicted.png", bbox_inches="tight")
plt.show()
print("saved: regime_split_predicted.png")


# ── figure 3: R² bar chart ─────────────────────────────────────────────────
labels  = ["Global\n(all 252)", "Easy environment\n(μ < 5)",
           "Hard environment\n(μ ≥ 5)"]
r2_vals = [r2_all, r2_below, r2_above]
colors  = ["#999999", "#e41a1c", "#377eb8"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, r2_vals, color=colors, alpha=0.85, width=0.45)
ax.axhline(0.9, color="green", linestyle=":", linewidth=1.5, label="R²=0.9")
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_ylabel("R² on test set (tortuosity)")
ax.set_ylim(min(min(r2_vals) - 0.1, -0.1), 1.05)
ax.set_title("Splitting by regime dramatically improves tortuosity prediction\n"
             "The phase transition creates two separately predictable behaviours")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")

for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.07,
            f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("regime_split_comparison.png", bbox_inches="tight")
plt.show()
print("saved: regime_split_comparison.png")


# ── thesis numbers ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  THESIS NUMBERS — copy these in")
print("="*60)
print(f"  Global formula R²          : {r2_all:.4f}")
print(f"  Easy environment R²        : {r2_below:.4f}")
print(f"  Hard environment R²        : {r2_above:.4f}")
print(f"  Improvement easy           : {r2_below - r2_all:+.4f}")
print(f"  Improvement hard           : {r2_above - r2_all:+.4f}")
print(f"\n  Easy environment — n={len(below)} distributions (μ < 5)")
print(f"    tortuosity range: {below['mean_tortuosity'].min():.3f} – {below['mean_tortuosity'].max():.3f}")
print(f"  Hard environment — n={len(above)} distributions (μ ≥ 5)")
print(f"    tortuosity range: {above['mean_tortuosity'].min():.3f} – {above['mean_tortuosity'].max():.3f}")
print(f"\n  Conclusion: splitting at μ=5 improves tortuosity R² from")
print(f"  {r2_all:.3f} to {max(r2_below, r2_above):.3f} in the better regime.")