# rq2_analysis.py
# RQ2: can a formula predict path properties from distribution parameters alone?
#
# reads data100.csv produced by big_simulation.py
# fits 4 formulas of increasing complexity for tortuosity, resistance and lateral deviation
# evaluates each on a held-out test set (20% of data, never seen during fitting)
#
# produces 3 figures:
#   rq2_formula_comparison.png  — R² bar chart comparing all 4 formulas
#   rq2_predicted_vs_actual.png — predicted vs actual scatter for best formula
#   rq2_residuals.png           — where the formula fails and why

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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

legend_handles = [mpatches.Patch(color=c, label=f) for f, c in COLORS.items()]
plt.rcParams.update({"font.size": 12, "figure.dpi": 130})
print(f"loaded {len(df)} distributions")

# formula inputs
X_mean = df["dist_mean"].values
X_var  = df["dist_var"].values
X_skew = df["dist_skew"].values

# formula targets — three properties to predict
y_tau  = df["mean_tortuosity"].values
y_res  = df["mean_resistance"].values
y_lat  = df["mean_lateral_dev"].values

# 80/20 train/test split
idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
print(f"training on {len(idx_train)}, testing on {len(idx_test)}")


def fit_and_report(name, X_tr, X_te, y_tr, y_te, target):
    model     = LinearRegression().fit(X_tr, y_tr)
    pred_test = model.predict(X_te)
    r2_tr     = r2_score(y_tr, model.predict(X_tr))
    r2_te     = r2_score(y_te, pred_test)
    c         = model.coef_
    b         = model.intercept_
    print(f"  [{target}] {name}")
    print(f"    R² train={r2_tr:.4f}   R² test={r2_te:.4f}")
    if len(c) == 1:
        print(f"    {target} = {c[0]:.4f}*μ + {b:.4f}")
    elif len(c) == 2:
        print(f"    {target} = {c[0]:.4f}*μ + {c[1]:.4f}*σ² + {b:.4f}")
    elif len(c) == 3:
        print(f"    {target} = {c[0]:.4f}*μ + {c[1]:.4f}*σ² + {c[2]:.4f}*skew + {b:.4f}")
    elif len(c) == 4:
        print(f"    {target} = {c[0]:.4f}*μ + {c[1]:.4f}*μ² + {c[2]:.4f}*σ² + {c[3]:.4f}*σ⁴ + {b:.4f}")
    return r2_te, pred_test


# build the 4 feature sets
X1_tr = X_mean[idx_train].reshape(-1, 1)
X1_te = X_mean[idx_test].reshape(-1, 1)

X2_tr = np.column_stack([X_mean[idx_train], X_var[idx_train]])
X2_te = np.column_stack([X_mean[idx_test],  X_var[idx_test]])

X3_tr = np.column_stack([X_mean[idx_train], X_var[idx_train], X_skew[idx_train]])
X3_te = np.column_stack([X_mean[idx_test],  X_var[idx_test],  X_skew[idx_test]])

X4_tr = np.column_stack([X_mean[idx_train], X_mean[idx_train]**2,
                          X_var[idx_train],  X_var[idx_train]**2])
X4_te = np.column_stack([X_mean[idx_test],  X_mean[idx_test]**2,
                          X_var[idx_test],   X_var[idx_test]**2])

feature_sets = [
    ("Linear(μ)",          X1_tr, X1_te),
    ("Linear(μ,σ²)",       X2_tr, X2_te),
    ("Linear(μ,σ²,skew)",  X3_tr, X3_te),
    ("Polynomial",          X4_tr, X4_te),
]

print("\n--- tortuosity ---")
results_tau = {}
for name, Xtr, Xte in feature_sets:
    r2, pred = fit_and_report(name, Xtr, Xte, y_tau[idx_train], y_tau[idx_test], "τ")
    results_tau[name] = (r2, pred)

best_tau = max(results_tau, key=lambda k: results_tau[k][0])
print(f"\n  best: {best_tau}  R²={results_tau[best_tau][0]:.4f}")

print("\n--- resistance ---")
results_res = {}
for name, Xtr, Xte in feature_sets:
    r2, pred = fit_and_report(name, Xtr, Xte, y_res[idx_train], y_res[idx_test], "R")
    results_res[name] = (r2, pred)

best_res = max(results_res, key=lambda k: results_res[k][0])
print(f"\n  best: {best_res}  R²={results_res[best_res][0]:.4f}")

print("\n--- max lateral deviation ---")
results_lat = {}
for name, Xtr, Xte in feature_sets:
    r2, pred = fit_and_report(name, Xtr, Xte, y_lat[idx_train], y_lat[idx_test], "Δx")
    results_lat[name] = (r2, pred)

best_lat = max(results_lat, key=lambda k: results_lat[k][0])
print(f"\n  best: {best_lat}  R²={results_lat[best_lat][0]:.4f}")


# ── figure 1: R² bar chart — all three targets ─────────────────────────────
formula_names = [n for n, _, _ in feature_sets]
r2_tau_vals   = [results_tau[n][0] for n in formula_names]
r2_res_vals   = [results_res[n][0] for n in formula_names]
r2_lat_vals   = [results_lat[n][0] for n in formula_names]

x     = np.arange(len(formula_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 5))
bars1 = ax.bar(x - width, r2_tau_vals, width, label="tortuosity τ",
               color="#377eb8", alpha=0.8)
bars2 = ax.bar(x,          r2_res_vals, width, label="resistance R",
               color="#e41a1c", alpha=0.8)
bars3 = ax.bar(x + width,  r2_lat_vals, width, label="lateral deviation Δx",
               color="#4daf4a", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(formula_names, fontsize=10)
ax.set_ylabel("R² on test set")
ax.set_ylim(min(min(r2_tau_vals), min(r2_lat_vals), 0) - 0.05, 1.05)
ax.set_title("RQ2 — formula comparison: R² on unseen test data\nhigher = better predictive power")
ax.legend()
ax.axhline(0.9, color="green", linestyle=":", linewidth=1.5)
ax.grid(True, alpha=0.3, axis="y")

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + 0.01 if h >= 0 else h - 0.06,
                f"{h:.3f}", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("rq2_formula_comparison.png", bbox_inches="tight")
plt.show()
print("saved: rq2_formula_comparison.png")


# ── figure 2: predicted vs actual — three targets ─────────────────────────
test_colors   = [COLORS[f] for f in df["family"].values[idx_test]]
best_tau_pred = results_tau[best_tau][1]
best_res_pred = results_res[best_res][1]
best_lat_pred = results_lat[best_lat][1]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("RQ2 — predicted vs actual (test set, 20% of data)\n"
             "points on the diagonal = perfect prediction", fontsize=13)

for ax, actual, predicted, xlabel, ylabel, title in [
    (axes[0], y_tau[idx_test], best_tau_pred,
     "actual tortuosity τ", "predicted tortuosity τ",
     f"tortuosity — {best_tau}\nR² = {results_tau[best_tau][0]:.3f}"),
    (axes[1], y_res[idx_test], best_res_pred,
     "actual resistance R", "predicted resistance R",
     f"resistance — {best_res}\nR² = {results_res[best_res][0]:.3f}"),
    (axes[2], y_lat[idx_test], best_lat_pred,
     "actual lateral deviation Δx", "predicted lateral deviation Δx",
     f"lateral deviation — {best_lat}\nR² = {results_lat[best_lat][0]:.3f}"),
]:
    ax.scatter(actual, predicted, c=test_colors, alpha=0.7, s=50)
    lim = [min(actual.min(), predicted.min()) - 0.5,
           max(actual.max(), predicted.max()) + 0.5]
    ax.plot(lim, lim, "k--", linewidth=1.5, label="perfect prediction")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("rq2_predicted_vs_actual.png", bbox_inches="tight")
plt.show()
print("saved: rq2_predicted_vs_actual.png")


# ── figure 3: residuals for tortuosity ────────────────────────────────────
residuals = y_tau[idx_test] - best_tau_pred

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_tau[idx_test], residuals, c=test_colors, alpha=0.7, s=50)
ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("actual tortuosity τ")
ax.set_ylabel("residual (actual − predicted)")
ax.set_title("RQ2 — residuals of tortuosity formula\n"
             "red points (bimodal) show where the formula struggles most")
ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rq2_residuals.png", bbox_inches="tight")
plt.show()
print("saved: rq2_residuals.png")


# ── thesis numbers ─────────────────────────────────────────────────────────
print("\n--- thesis numbers ---")
print(f"  tortuosity range       : {y_tau.min():.3f} – {y_tau.max():.3f}")
print(f"  resistance range       : {y_res.min():.1f} – {y_res.max():.1f}")
print(f"  lateral deviation range: {y_lat.min():.3f} – {y_lat.max():.3f}")
print()
print(f"  {'formula':<25}  {'τ R²':>8}  {'R R²':>8}  {'Δx R²':>8}")
for name in formula_names:
    print(f"  {name:<25}  "
          f"{results_tau[name][0]:>8.4f}  "
          f"{results_res[name][0]:>8.4f}  "
          f"{results_lat[name][0]:>8.4f}")