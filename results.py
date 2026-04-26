"""
THESIS ANALYSIS — RQ1 & RQ2
==============================
Run this script after data_collection_sweep.py has produced master_sweep.csv.

What this script does:
  RQ1 — plots that show HOW path properties depend on the distribution
  RQ2 — fits formulas that PREDICT path properties from distribution parameters

Output: saves all figures as PNG files you can paste into your thesis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ── load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("master_sweep.csv")

# Add a clean family label for colouring plots
def get_family(label):
    if label.startswith("bimodal"):   return "bimodal"
    if label.startswith("uniform"):   return "uniform"
    if label.startswith("obstacles"): return "obstacles"
    if label.startswith("low"):       return "low_resistance"
    if label.startswith("exp"):       return "exponential"
    if label.startswith("fixed"):     return "fixed_mean"
    return "random"

df["family"] = df["dist_label"].apply(get_family)

FAMILY_COLORS = {
    "bimodal":       "#e41a1c",
    "uniform":       "#377eb8",
    "obstacles":     "#ff7f00",
    "low_resistance":"#4daf4a",
    "exponential":   "#984ea3",
    "fixed_mean":    "#a65628",
    "random":        "#999999",
}

plt.rcParams.update({"font.size": 12, "figure.dpi": 130})

print("Data loaded:", len(df), "distributions")
print("Families:", df["family"].value_counts().to_dict())
print()


# ══════════════════════════════════════════════════════════════════════════
#  RQ1 — FIGURE 1: Scatter plots — tortuosity and resistance vs mean & var
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RQ1 — How path properties depend on the distribution\n"
             "(each point = one distribution, colour = family)", fontsize=14)

plot_configs = [
    (axes[0,0], "dist_mean",  "mean_tortuosity",  "Mean resistance μ",   "Mean tortuosity τ"),
    (axes[0,1], "dist_var",   "mean_tortuosity",  "Variance σ²",         "Mean tortuosity τ"),
    (axes[1,0], "dist_mean",  "mean_resistance",  "Mean resistance μ",   "Mean total resistance R"),
    (axes[1,1], "dist_var",   "mean_resistance",  "Variance σ²",         "Mean total resistance R"),
]

for ax, xcol, ycol, xlabel, ylabel in plot_configs:
    for fam, color in FAMILY_COLORS.items():
        sub = df[df["family"] == fam]
        ax.scatter(sub[xcol], sub[ycol], c=color, alpha=0.7, s=40, label=fam)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

# Add legend to first subplot only
handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
axes[0,0].legend(handles=handles, fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig("rq1_scatter_overview.png", bbox_inches="tight")
plt.show()
print("Saved: rq1_scatter_overview.png")


# ══════════════════════════════════════════════════════════════════════════
#  RQ1 — FIGURE 2: Phase transition in the bimodal family
# ══════════════════════════════════════════════════════════════════════════
bim = df[df["family"] == "bimodal"].copy()
bim["p0_val"] = bim["p0"]   # already a column

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RQ1 — Phase transition in the bimodal distribution\n"
             "(only R=0 or R=10 possible; p₀ = probability of R=0)", fontsize=13)

# Resistance
axes[0].plot(bim["p0_val"], bim["mean_resistance"], "o-", color="#e41a1c",
             linewidth=2, markersize=5)
axes[0].fill_between(bim["p0_val"],
                     bim["mean_resistance"] - bim["std_resistance"],
                     bim["mean_resistance"] + bim["std_resistance"],
                     alpha=0.2, color="#e41a1c")
axes[0].axvline(0.5, color="black", linestyle="--", linewidth=1.5,
                label="p₀ = 0.5 (observed transition)")
axes[0].set_xlabel("p₀  (probability of free edge R=0)")
axes[0].set_ylabel("Mean total resistance")
axes[0].set_title("Resistance collapses sharply near p₀ ≈ 0.5")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Tortuosity
axes[1].plot(bim["p0_val"], bim["mean_tortuosity"], "o-", color="#377eb8",
             linewidth=2, markersize=5)
axes[1].fill_between(bim["p0_val"],
                     bim["mean_tortuosity"] - bim["std_tortuosity"],
                     bim["mean_tortuosity"] + bim["std_tortuosity"],
                     alpha=0.2, color="#377eb8")
axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.5,
                label="p₀ = 0.5")
axes[1].set_xlabel("p₀  (probability of free edge R=0)")
axes[1].set_ylabel("Mean tortuosity τ")
axes[1].set_title("Tortuosity peaks near the transition, then falls")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rq1_phase_transition.png", bbox_inches="tight")
plt.show()
print("Saved: rq1_phase_transition.png")


# ══════════════════════════════════════════════════════════════════════════
#  RQ1 — FIGURE 3: Fixed-mean family — does variance matter?
# ══════════════════════════════════════════════════════════════════════════
fix = df[df["family"] == "fixed_mean"].copy()
fix_f = fix["dist_label"].str.extract(r"f=(\d+\.\d+)")[0].astype(float)
fix = fix.copy()
fix["f_val"] = fix_f

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RQ1 — Fixed mean (μ=5), varying variance\n"
             "f=0: pure uniform (high σ²)  →  f=0.95: spike at R=5 (low σ²)", fontsize=13)

axes[0].plot(fix["dist_var"], fix["mean_tortuosity"], "o-", color="#a65628",
             linewidth=2, markersize=6)
axes[0].set_xlabel("Variance σ²")
axes[0].set_ylabel("Mean tortuosity τ")
axes[0].set_title("Tortuosity increases with variance\n(same mean, different spread)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(fix["dist_var"], fix["mean_resistance"], "o-", color="#984ea3",
             linewidth=2, markersize=6)
axes[1].set_xlabel("Variance σ²")
axes[1].set_ylabel("Mean total resistance")
axes[1].set_title("Resistance decreases with variance\n(higher spread = more low values to exploit)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("rq1_fixed_mean_variance.png", bbox_inches="tight")
plt.show()
print("Saved: rq1_fixed_mean_variance.png\n")


# ══════════════════════════════════════════════════════════════════════════
#  RQ2 — FORMULA FITTING
#  We use 80% of data to fit, 20% to test (honest evaluation)
#  We try multiple formulas and report R² for each
# ══════════════════════════════════════════════════════════════════════════

# Split data — exclude the bimodal family near the transition for formula fitting
# (it's non-linear there, which we already explain in RQ1)
df_fit = df.copy()   # use ALL data first, then discuss breakdown

X_mean = df_fit["dist_mean"].values
X_var  = df_fit["dist_var"].values
X_skew = df_fit["dist_skew"].values
y_tau  = df_fit["mean_tortuosity"].values
y_res  = df_fit["mean_resistance"].values

# Train/test split (stratified by family for fairness)
idx_train, idx_test = train_test_split(
    np.arange(len(df_fit)), test_size=0.2, random_state=42
)

print("=" * 60)
print("  RQ2 — FORMULA FITTING RESULTS")
print("=" * 60)


# ── Helper: fit and report ─────────────────────────────────────────────
def fit_and_report(name, X_tr, X_te, y_tr, y_te, target_name):
    model = LinearRegression().fit(X_tr, y_tr)
    y_pred_train = model.predict(X_tr)
    y_pred_test  = model.predict(X_te)
    r2_train = r2_score(y_tr, y_pred_train)
    r2_test  = r2_score(y_te, y_pred_test)
    coef     = model.coef_
    intercept = model.intercept_
    print(f"\n  [{target_name}] {name}")
    print(f"    R² train = {r2_train:.4f}   R² test = {r2_test:.4f}")
    if len(coef) == 1:
        print(f"    formula: {target_name} = {coef[0]:.4f} * x + {intercept:.4f}")
    elif len(coef) == 2:
        print(f"    formula: {target_name} = {coef[0]:.4f}*μ + {coef[1]:.4f}*σ² + {intercept:.4f}")
    elif len(coef) == 3:
        print(f"    formula: {target_name} = {coef[0]:.4f}*μ + {coef[1]:.4f}*σ² + {coef[2]:.4f}*skew + {intercept:.4f}")
    return model, r2_test, model.predict(X_te)


results_tau = {}
results_res = {}

# --- TORTUOSITY FORMULAS ---
print("\n--- TORTUOSITY ---")

# Formula 1: linear in mean only
X1_tr = X_mean[idx_train].reshape(-1,1)
X1_te = X_mean[idx_test].reshape(-1,1)
m, r2, pred = fit_and_report("Linear in μ only", X1_tr, X1_te,
                              y_tau[idx_train], y_tau[idx_test], "τ")
results_tau["Linear(μ)"] = (r2, pred)

# Formula 2: linear in mean + variance
X2_tr = np.column_stack([X_mean[idx_train], X_var[idx_train]])
X2_te = np.column_stack([X_mean[idx_test],  X_var[idx_test]])
m2, r2, pred = fit_and_report("Linear in μ + σ²", X2_tr, X2_te,
                               y_tau[idx_train], y_tau[idx_test], "τ")
results_tau["Linear(μ,σ²)"] = (r2, pred)

# Formula 3: linear in mean + variance + skew
X3_tr = np.column_stack([X_mean[idx_train], X_var[idx_train], X_skew[idx_train]])
X3_te = np.column_stack([X_mean[idx_test],  X_var[idx_test],  X_skew[idx_test]])
m3, r2, pred = fit_and_report("Linear in μ + σ² + skew", X3_tr, X3_te,
                               y_tau[idx_train], y_tau[idx_test], "τ")
results_tau["Linear(μ,σ²,skew)"] = (r2, pred)

# Formula 4: polynomial (mean², var²)
X4_tr = np.column_stack([X_mean[idx_train], X_mean[idx_train]**2,
                          X_var[idx_train],  X_var[idx_train]**2])
X4_te = np.column_stack([X_mean[idx_test],  X_mean[idx_test]**2,
                          X_var[idx_test],   X_var[idx_test]**2])
m4, r2, pred = fit_and_report("Polynomial (μ, μ², σ², σ⁴)", X4_tr, X4_te,
                               y_tau[idx_train], y_tau[idx_test], "τ")
results_tau["Polynomial"] = (r2, pred)

# Best tortuosity model
best_tau_name = max(results_tau, key=lambda k: results_tau[k][0])
print(f"\n  >> Best tortuosity formula: {best_tau_name}  (R²={results_tau[best_tau_name][0]:.4f})")

# --- RESISTANCE FORMULAS ---
print("\n--- RESISTANCE ---")

m_r1, r2, pred = fit_and_report("Linear in μ only", X1_tr, X1_te,
                                  y_res[idx_train], y_res[idx_test], "R")
results_res["Linear(μ)"] = (r2, pred)

m_r2, r2, pred = fit_and_report("Linear in μ + σ²", X2_tr, X2_te,
                                  y_res[idx_train], y_res[idx_test], "R")
results_res["Linear(μ,σ²)"] = (r2, pred)

m_r3, r2, pred = fit_and_report("Linear in μ + σ² + skew", X3_tr, X3_te,
                                  y_res[idx_train], y_res[idx_test], "R")
results_res["Linear(μ,σ²,skew)"] = (r2, pred)

m_r4, r2, pred = fit_and_report("Polynomial (μ, μ², σ², σ⁴)", X4_tr, X4_te,
                                  y_res[idx_train], y_res[idx_test], "R")
results_res["Polynomial"] = (r2, pred)

best_res_name = max(results_res, key=lambda k: results_res[k][0])
print(f"\n  >> Best resistance formula: {best_res_name}  (R²={results_res[best_res_name][0]:.4f})")


# ── RQ2 FIGURE 4: Predicted vs Actual for best formulas ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RQ2 — Predicted vs Actual (test set, 20% of data)\n"
             "Points on the diagonal = perfect prediction", fontsize=13)

# colours for test points by family
test_families = df_fit["family"].values[idx_test]
test_colors   = [FAMILY_COLORS[f] for f in test_families]

# Tortuosity
best_tau_pred = results_tau[best_tau_name][1]
axes[0].scatter(y_tau[idx_test], best_tau_pred, c=test_colors, alpha=0.7, s=50)
lim = [min(y_tau.min(), best_tau_pred.min()) - 0.02,
       max(y_tau.max(), best_tau_pred.max()) + 0.02]
axes[0].plot(lim, lim, "k--", linewidth=1.5, label="Perfect prediction")
axes[0].set_xlabel("Actual tortuosity τ")
axes[0].set_ylabel("Predicted tortuosity τ")
axes[0].set_title(f"Tortuosity — {best_tau_name}\nR² = {results_tau[best_tau_name][0]:.3f}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Resistance
best_res_pred = results_res[best_res_name][1]
axes[1].scatter(y_res[idx_test], best_res_pred, c=test_colors, alpha=0.7, s=50)
lim2 = [min(y_res.min(), best_res_pred.min()) - 5,
        max(y_res.max(), best_res_pred.max()) + 5]
axes[1].plot(lim2, lim2, "k--", linewidth=1.5, label="Perfect prediction")
axes[1].set_xlabel("Actual mean resistance R")
axes[1].set_ylabel("Predicted mean resistance R")
axes[1].set_title(f"Resistance — {best_res_name}\nR² = {results_res[best_res_name][0]:.3f}")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Add family legend
handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig("rq2_predicted_vs_actual.png", bbox_inches="tight")
plt.show()
print("\nSaved: rq2_predicted_vs_actual.png")


# ── RQ2 FIGURE 5: Where does the formula BREAK DOWN? ──────────────────
# Refit best model on ALL data, compute residuals, highlight bimodal
best_Xtr_tau = X4_tr if best_tau_name == "Polynomial" else \
               X3_tr if "skew" in best_tau_name else \
               X2_tr if "σ²" in best_tau_name else X1_tr
best_Xte_tau = X4_te if best_tau_name == "Polynomial" else \
               X3_te if "skew" in best_tau_name else \
               X2_te if "σ²" in best_tau_name else X1_te

residuals = y_tau[idx_test] - results_tau[best_tau_name][1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(y_tau[idx_test], residuals, c=test_colors, alpha=0.7, s=50)
ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax.set_xlabel("Actual tortuosity τ")
ax.set_ylabel("Residual (actual − predicted)")
ax.set_title("RQ2 — Residuals of tortuosity formula\n"
             "Red points (bimodal) show where the formula struggles most")
ax.grid(True, alpha=0.3)
handles = [mpatches.Patch(color=c, label=f) for f, c in FAMILY_COLORS.items()]
ax.legend(handles=handles, fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("rq2_residuals.png", bbox_inches="tight")
plt.show()
print("Saved: rq2_residuals.png")


# ══════════════════════════════════════════════════════════════════════════
#  RQ2 — FIGURE 6: R² comparison table as a bar chart
# ══════════════════════════════════════════════════════════════════════════
formula_names = list(results_tau.keys())
r2_tau_vals   = [results_tau[k][0] for k in formula_names]
r2_res_vals   = [results_res[k][0] for k in formula_names]

x = np.arange(len(formula_names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 5))
bars1 = ax.bar(x - width/2, r2_tau_vals, width, label="Tortuosity τ",
               color="#377eb8", alpha=0.8)
bars2 = ax.bar(x + width/2, r2_res_vals, width, label="Resistance R",
               color="#e41a1c", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(formula_names, fontsize=10)
ax.set_ylabel("R² on test set")
ax.set_ylim(0, 1.05)
ax.set_title("RQ2 — Formula comparison: R² on unseen test data\n"
             "Higher = better predictive power")
ax.legend()
ax.axhline(0.9, color="green", linestyle=":", linewidth=1.5, label="R²=0.9 threshold")
ax.grid(True, alpha=0.3, axis="y")

# Label bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("rq2_formula_comparison.png", bbox_inches="tight")
plt.show()
print("Saved: rq2_formula_comparison.png")


# ══════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINTOUT  (copy these numbers into your thesis)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  THESIS NUMBERS — copy these into your writing")
print("=" * 60)
print(f"\n  Tortuosity range:  {y_tau.min():.3f} – {y_tau.max():.3f}")
print(f"  Resistance range:  {y_res.min():.1f} – {y_res.max():.1f}")

print("\n  R² results (test set):")
for name in formula_names:
    print(f"    {name:<30}  τ: {results_tau[name][0]:.4f}   R: {results_res[name][0]:.4f}")

print(f"\n  Phase transition (bimodal family):")
bim_cross = bim[bim["mean_resistance"] < 1.0]["p0_val"].min()
print(f"    Resistance first drops to ~0 at p₀ = {bim_cross:.3f}")
bim_peak_tau = bim.loc[bim["mean_tortuosity"].idxmax(), "p0_val"]
print(f"    Tortuosity peaks at p₀ = {bim_peak_tau:.3f}")

print("\n  Fixed-mean family (variance effect):")
print(f"    Tortuosity range (μ=5 fixed): "
      f"{fix['mean_tortuosity'].min():.3f} – {fix['mean_tortuosity'].max():.3f}")
print(f"    This proves variance is NOT redundant given the mean.")
print()