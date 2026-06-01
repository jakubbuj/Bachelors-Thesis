# rq2_analysis.py
# RQ2: can a formula predict path properties from distribution parameters alone?
#
# produces 6 figures:
#   rq2_formula_comparison.png   — R² bar chart for all 4 formulas
#   rq2_predicted_vs_actual.png  — predicted vs actual for best formula
#   rq2_residuals.png            — tortuosity residuals by family
#   regime_split_comparison.png  — R² bar chart hard vs easy regimes
#   regime_split_predicted.png   — predicted vs actual per regime
#   regime_split_scatter.png     — all distributions coloured hard/easy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

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

X_mean = df["dist_mean"].values
X_var  = df["dist_var"].values
X_skew = df["dist_skew"].values
y_tau  = df["mean_tortuosity"].values
y_res  = df["mean_resistance"].values

idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
print(f"training on {len(idx_train)}, testing on {len(idx_test)}")


def fit_and_report(name, X_tr, X_te, y_tr, y_te, target):
    model     = LinearRegression().fit(X_tr, y_tr)
    pred_test = model.predict(X_te)
    r2_tr     = r2_score(y_tr, model.predict(X_tr))
    r2_te     = r2_score(y_te, pred_test)
    c         = model.coef_
    b         = model.intercept_
    print(f"  [{target}] {name}  R² train={r2_tr:.4f}  R² test={r2_te:.4f}")
    return r2_te, pred_test, model


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
    ("Linear(μ)",         X1_tr, X1_te),
    ("Linear(μ,σ²)",      X2_tr, X2_te),
    ("Linear(μ,σ²,skew)", X3_tr, X3_te),
    ("Polynomial",         X4_tr, X4_te),
]

print("\n--- tortuosity ---")
results_tau = {}
for name, Xtr, Xte in feature_sets:
    r2, pred, model = fit_and_report(name, Xtr, Xte, y_tau[idx_train], y_tau[idx_test], "τ")
    results_tau[name] = (r2, pred)

best_tau = max(results_tau, key=lambda k: results_tau[k][0])

print("\n--- resistance ---")
results_res = {}
for name, Xtr, Xte in feature_sets:
    r2, pred, model = fit_and_report(name, Xtr, Xte, y_res[idx_train], y_res[idx_test], "R")
    results_res[name] = (r2, pred)

best_res = max(results_res, key=lambda k: results_res[k][0])


# figure 1: R² bar chart
formula_names = [n for n, _, _ in feature_sets]
r2_tau_vals   = [results_tau[n][0] for n in formula_names]
r2_res_vals   = [results_res[n][0] for n in formula_names]

x     = np.arange(len(formula_names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, r2_tau_vals, width, label="tortuosity τ",
               color="#377eb8", alpha=0.8)
bars2 = ax.bar(x + width/2, r2_res_vals, width, label="resistance R",
               color="#e41a1c", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(formula_names, fontsize=10)
ax.set_ylabel("R² on test set")
ax.set_ylim(min(min(r2_tau_vals), 0) - 0.05, 1.05)
ax.set_title("RQ2 — formula comparison: R² on unseen test data\nhigher = better predictive power")
ax.legend()
ax.axhline(0.9, color="green", linestyle=":", linewidth=1.5)
ax.grid(True, alpha=0.3, axis="y")
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + 0.01 if h >= 0 else h - 0.05,
            f"{h:.3f}", ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("rq2_formula_comparison.png", bbox_inches="tight")
plt.show()
print("saved: rq2_formula_comparison.png")


# figure 2: predicted vs actual
test_colors   = [COLORS[f] for f in df["family"].values[idx_test]]
best_tau_pred = results_tau[best_tau][1]
best_res_pred = results_res[best_res][1]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RQ2 — predicted vs actual (test set)\n"
             "points on the diagonal = perfect prediction", fontsize=13)
for ax, actual, predicted, xlabel, ylabel, title in [
    (axes[0], y_tau[idx_test], best_tau_pred,
     "actual tortuosity τ", "predicted tortuosity τ",
     f"tortuosity — {best_tau}  R² = {results_tau[best_tau][0]:.3f}"),
    (axes[1], y_res[idx_test], best_res_pred,
     "actual resistance R", "predicted resistance R",
     f"resistance — {best_res}  R² = {results_res[best_res][0]:.3f}"),
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


# figure 3: residuals
residuals = y_tau[idx_test] - best_tau_pred
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_tau[idx_test], residuals, c=test_colors, alpha=0.7, s=50)
ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("actual tortuosity τ")
ax.set_ylabel("residual (actual − predicted)")
ax.set_title("RQ2 — tortuosity residuals\nred = bimodal, shows where the formula struggles most")
ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rq2_residuals.png", bbox_inches="tight")
plt.show()
print("saved: rq2_residuals.png")


# figure 4: regime split — R² comparison bar chart
SPLIT = 5.0
hard_idx = np.where(df["dist_mean"].values >= SPLIT)[0]
easy_idx = np.where(df["dist_mean"].values <  SPLIT)[0]

def regime_r2(idx_subset, X_full, y, degree=2):
    tr = np.intersect1d(idx_train, idx_subset)
    te = np.intersect1d(idx_test,  idx_subset)
    if len(tr) < 5 or len(te) < 3:
        return None, None
    poly  = PolynomialFeatures(degree=degree, include_bias=True)
    X_tr  = poly.fit_transform(X_full[tr])
    X_te  = poly.transform(X_full[te])
    model = LinearRegression().fit(X_tr, y[tr])
    pred  = model.predict(X_te)
    return r2_score(y[te], pred), pred

X_mu_var = np.column_stack([X_mean, X_var])

r2_hard_tau, _ = regime_r2(hard_idx, X_mu_var, y_tau)
r2_easy_tau, _ = regime_r2(easy_idx, X_mu_var, y_tau)
r2_hard_res, _ = regime_r2(hard_idx, X_mu_var, y_res)
r2_easy_res, _ = regime_r2(easy_idx, X_mu_var, y_res)

print(f"\nRegime split at μ = {SPLIT}:")
print(f"  hard (μ≥{SPLIT}): τ R²={r2_hard_tau:.3f}  R R²={r2_hard_res:.3f}  n={len(hard_idx)}")
print(f"  easy (μ<{SPLIT}):  τ R²={r2_easy_tau:.3f}  R R²={r2_easy_res:.3f}  n={len(easy_idx)}")

fig, ax = plt.subplots(figsize=(9, 5))
regimes   = ["hard (μ≥5)\nτ", "easy (μ<5)\nτ", "hard (μ≥5)\nR", "easy (μ<5)\nR"]
r2_vals   = [r2_hard_tau, r2_easy_tau, r2_hard_res, r2_easy_res]
bar_colors = ["#377eb8", "#aec7e8", "#e41a1c", "#f7a8a8"]
bars = ax.bar(regimes, r2_vals, color=bar_colors, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("R² on test set")
ax.set_title(f"RQ2 — regime split at μ = {SPLIT}\npolynomial formula fitted separately per regime")
ax.grid(True, alpha=0.3, axis="y")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + 0.01 if h >= 0 else h - 0.06,
            f"{h:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("regime_split_comparison.png", bbox_inches="tight")
plt.show()
print("saved: regime_split_comparison.png")


# figure 5: regime split predicted vs actual
hard_te = np.intersect1d(idx_test, hard_idx)
easy_te = np.intersect1d(idx_test, easy_idx)

def get_preds(idx_subset, X_full, y, degree=2):
    tr    = np.intersect1d(idx_train, idx_subset)
    te    = np.intersect1d(idx_test,  idx_subset)
    poly  = PolynomialFeatures(degree=degree, include_bias=True)
    model = LinearRegression().fit(poly.fit_transform(X_full[tr]), y[tr])
    return te, model.predict(poly.transform(X_full[te]))

hard_te_idx, hard_tau_pred = get_preds(hard_idx, X_mu_var, y_tau)
easy_te_idx, easy_tau_pred = get_preds(easy_idx, X_mu_var, y_tau)

hard_colors = [COLORS[df["family"].values[i]] for i in hard_te_idx]
easy_colors = [COLORS[df["family"].values[i]] for i in easy_te_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RQ2 — regime split: predicted vs actual tortuosity\n"
             "polynomial formula fitted separately for each regime", fontsize=13)

for ax, actual_idx, pred, colors, title in [
    (axes[0], hard_te_idx, hard_tau_pred, hard_colors,
     f"hard environments (μ≥5)  R²={r2_hard_tau:.3f}"),
    (axes[1], easy_te_idx, easy_tau_pred, easy_colors,
     f"easy environments (μ<5)  R²={r2_easy_tau:.3f}"),
]:
    actual = y_tau[actual_idx]
    ax.scatter(actual, pred, c=colors, alpha=0.75, s=55)
    lim = [min(actual.min(), pred.min()) - 0.02,
           max(actual.max(), pred.max()) + 0.02]
    ax.plot(lim, lim, "k--", linewidth=1.5, label="perfect prediction")
    ax.set_xlabel("actual tortuosity τ")
    ax.set_ylabel("predicted tortuosity τ")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("regime_split_predicted.png", bbox_inches="tight")
plt.show()
print("saved: regime_split_predicted.png")


# figure 6: regime split scatter — all distributions coloured hard/easy
all_colors = ["#e41a1c" if df["dist_mean"].values[i] >= SPLIT else "#377eb8"
              for i in range(len(df))]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RQ2 — hard vs easy environments\n"
             "red = hard (μ≥5), blue = easy (μ<5)", fontsize=13)

axes[0].scatter(df["dist_mean"].values, y_tau, c=all_colors, alpha=0.6, s=40)
axes[0].axvline(SPLIT, color="black", linestyle="--", linewidth=1.5,
                label=f"split at μ = {SPLIT}")
axes[0].set_xlabel("mean μ")
axes[0].set_ylabel("mean tortuosity τ")
axes[0].set_title("tortuosity across all distributions")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df["dist_mean"].values, y_res, c=all_colors, alpha=0.6, s=40)
axes[1].axvline(SPLIT, color="black", linestyle="--", linewidth=1.5,
                label=f"split at μ = {SPLIT}")
axes[1].set_xlabel("mean μ")
axes[1].set_ylabel("mean resistance R")
axes[1].set_title("resistance across all distributions")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

hard_patch = mpatches.Patch(color="#e41a1c", label=f"hard (μ≥{SPLIT})")
easy_patch = mpatches.Patch(color="#377eb8", label=f"easy (μ<{SPLIT})")
fig.legend(handles=[hard_patch, easy_patch], loc="lower center", ncol=2,
           fontsize=10, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("regime_split_scatter.png", bbox_inches="tight")
plt.show()
print("saved: regime_split_scatter.png")


# thesis numbers
print("\n--- thesis numbers ---")
print(f"  tortuosity range : {y_tau.min():.3f} – {y_tau.max():.3f}")
print(f"  resistance range : {y_res.min():.1f} – {y_res.max():.1f}")
print(f"\n  {'formula':<25}  {'τ R²':>8}  {'R R²':>8}")
for name in formula_names:
    print(f"  {name:<25}  {results_tau[name][0]:>8.4f}  {results_res[name][0]:>8.4f}")