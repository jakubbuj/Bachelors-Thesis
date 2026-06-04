# rq2_analysis.py
# RQ2: can a formula predict path properties from distribution parameters alone?
#
# produces 12 figures:
#   rq2_formula_comparison.png          - R² bar chart for all 4 formulas (test set)
#   rq2_predicted_vs_actual.png         - predicted vs actual for best formula (test set)
#   rq2_residuals.png                   - tortuosity residuals by family
#   rq2_regime_split_comparison.png     - R² bar chart hard vs easy regimes (test set)
#   rq2_regime_split_predicted.png      - predicted vs actual per regime (test set)
#   rq2_regime_split_scatter.png        - all distributions coloured hard vs easy
#   rq2_formulas_resistance.png         - all 4 formulas for resistance
#   rq2_formulas_tortuosity.png         - all 4 formulas for tortuosity
#   rq2_surfaces_resistance.png         - fitted surfaces F2 F3 F4 for resistance
#   rq2_surfaces_tortuosity.png         - fitted surfaces F2 F3 F4 for tortuosity
#   rq2_regime_surface_resistance.png   - regime split plane for resistance
#   rq2_regime_surface_tortuosity.png   - regime split plane for tortuosity

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("/Users/jakubb/Desktop/Thesis/FINAL/data100.csv")

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
ax.set_title("RQ2 — formula comparison: R² on test set")
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
fig.suptitle("RQ2 — predicted vs actual (test set)", fontsize=13)
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
ax.set_title("RQ2 — tortuosity residuals by family")
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

# global tortuosity R² on test set (from Polynomial formula)
r2_global_tau = results_tau["Polynomial"][0]

# hard and easy R² — separate model per regime (same as regime_r2 above)
# r2_hard_tau and r2_easy_tau are already computed by regime_r2

fig, ax = plt.subplots(figsize=(8, 5))
labels     = ["global τ", f"hard (μ≥{SPLIT}) τ", f"easy (μ<{SPLIT}) τ"]
r2_vals    = [r2_global_tau, r2_hard_tau, r2_easy_tau]
bar_colors = ["#999999", "#377eb8", "#aec7e8"]
bars = ax.bar(labels, r2_vals, color=bar_colors, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("R² on test set")
ax.set_title(f"RQ2 — tortuosity prediction: global vs regime split at μ = {SPLIT}")
ax.set_ylim(-0.1, 1.05)
ax.grid(True, alpha=0.3, axis="y")
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + 0.01 if h >= 0 else h - 0.06,
            f"{h:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("rq2_regime_split_comparison.png", bbox_inches="tight")
plt.show()
print("saved: rq2_regime_split_comparison.png")


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
fig.suptitle("RQ2 — regime split: predicted vs actual tortuosity", fontsize=13)

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
plt.savefig("rq2_regime_split_predicted.png", bbox_inches="tight")
plt.show()
print("saved: regime_split_predicted.png")


# figure 6: regime split scatter — all distributions coloured hard/easy
all_colors = ["#e41a1c" if df["dist_mean"].values[i] >= SPLIT else "#377eb8"
              for i in range(len(df))]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("RQ2 — hard vs easy environments (μ split at 5)", fontsize=13)

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
plt.savefig("rq2_regime_split_scatter.png", bbox_inches="tight")
plt.show()
print("saved: regime_split_scatter.png")


# thesis numbers
print("\n--- thesis numbers ---")
print(f"  tortuosity range : {y_tau.min():.3f} – {y_tau.max():.3f}")
print(f"  resistance range : {y_res.min():.1f} – {y_res.max():.1f}")
print(f"\n  {'formula':<25}  {'τ R²':>8}  {'R R²':>8}")
for name in formula_names:
    print(f"  {name:<25}  {results_tau[name][0]:>8.4f}  {results_res[name][0]:>8.4f}")


# figure 7: separate plot for each formula — predicted vs mu and predicted vs actual
from mpl_toolkits.mplot3d import Axes3D

X1_all = X_mean.reshape(-1, 1)
X2_all = np.column_stack([X_mean, X_var])
X3_all = np.column_stack([X_mean, X_var, X_skew])
X4_all = np.column_stack([X_mean, X_mean**2, X_var, X_var**2])

all_sets = [
    ("F1: Linear(μ)",         X1_all, "#e41a1c"),
    ("F2: Linear(μ,σ²)",      X2_all, "#377eb8"),
    ("F3: Linear(μ,σ²,skew)", X3_all, "#ff7f00"),
    ("F4: Polynomial",         X4_all, "#4daf4a"),
]

mu_sort  = np.argsort(X_mean)
mu_grid  = np.linspace(X_mean.min(), X_mean.max(), 30)
var_grid = np.linspace(X_var.min(),  X_var.max(),  30)
MU, VAR  = np.meshgrid(mu_grid, var_grid)

for target, y, ylabel in [
    ("resistance", y_res, "resistance R"),
    ("tortuosity", y_tau, "tortuosity τ"),
]:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"All four formulas — {target}", fontsize=13)

    for col, (name, X, color) in enumerate(all_sets):
        model = LinearRegression().fit(X[idx_train], y[idx_train])
        pred  = model.predict(X[idx_test])
        r2    = r2_score(y[idx_test], pred)

        ax_top = axes[0, col]
        ax_top.scatter(X_mean[idx_test], y[idx_test], color="#cccccc", s=15, alpha=0.5, label="actual")
        ax_top.scatter(X_mean[idx_test], pred, color=color, s=12, alpha=0.7, label="predicted")
        ax_top.set_xlabel("mean μ")
        ax_top.set_ylabel(ylabel)
        ax_top.set_title(f"{name}\nR²={r2:.3f}")
        ax_top.legend(fontsize=8)
        ax_top.grid(True, alpha=0.3)

        ax_bot = axes[1, col]
        lim = [min(y[idx_test].min(), pred.min()) - 0.02 * abs(y.max()),
               max(y[idx_test].max(), pred.max()) + 0.02 * abs(y.max())]
        ax_bot.plot(lim, lim, "k--", linewidth=1.5, label="perfect")
        ax_bot.scatter(y[idx_test], pred, color=color, s=18, alpha=0.6)
        ax_bot.set_xlabel(f"actual {ylabel}")
        ax_bot.set_ylabel(f"predicted {ylabel}")
        ax_bot.set_title(f"predicted vs actual  R²={r2:.3f}")
        ax_bot.legend(fontsize=8)
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"rq2_formulas_{target}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=130)
    plt.show()
    print(f"saved: {fname}")


# figure 8: 3D surfaces — F2 plane, F3 plane, F4 curved surface (test set)
skew_mean = X_skew[idx_test].mean()

for target, y_data, ylabel, surf_color in [
    ("resistance", y_res, "resistance R", "Reds"),
    ("tortuosity", y_tau, "tortuosity τ", "Blues"),
]:
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"Fitted surfaces — {target}", fontsize=12)

    for col, (name, X_all) in enumerate([
        ("F2: Linear(μ,σ²)",      X2_all),
        ("F3: Linear(μ,σ²,skew)", X3_all),
        ("F4: Polynomial",         X4_all),
    ]):
        model = LinearRegression().fit(X_all[idx_train], y_data[idx_train])
        pred  = model.predict(X_all[idx_test])
        r2    = r2_score(y_data[idx_test], pred)

        if name == "F2: Linear(μ,σ²)":
            Z = model.coef_[0] * MU + model.coef_[1] * VAR + model.intercept_
        elif name == "F3: Linear(μ,σ²,skew)":
            Z = (model.coef_[0] * MU + model.coef_[1] * VAR
                 + model.coef_[2] * skew_mean + model.intercept_)
        else:
            Z = (model.coef_[0] * MU + model.coef_[1] * MU**2
                 + model.coef_[2] * VAR + model.coef_[3] * VAR**2
                 + model.intercept_)

        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.scatter(X_mean[idx_test], X_var[idx_test], y_data[idx_test],
                   color="#aaaaaa", s=12, alpha=0.35)
        ax.plot_surface(MU, VAR, Z, alpha=0.45, cmap=surf_color)
        ax.set_xlabel("mean μ")
        ax.set_ylabel("variance σ²")
        ax.set_zlabel(ylabel)
        ax.set_title(f"{name}\nR²={r2:.3f}")

    plt.tight_layout()
    fname = f"rq2_surfaces_{target}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=130)
    plt.show()
    print(f"saved: {fname}")


# figure 9: hard vs easy — degree-2 polynomial fitted on train, evaluated on test
SPLIT = 5.0
hard_mask = X_mean >= SPLIT
easy_mask = X_mean <  SPLIT

for target, y_data, ylabel, h_color, e_color in [
    ("resistance", y_res, "resistance R", "Reds",  "Oranges"),
    ("tortuosity", y_tau, "tortuosity τ", "Blues", "Greens"),
]:
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(f"Regime split — {target}", fontsize=11)

    for col, (mask, label, cmap) in enumerate([
        (hard_mask, f"hard (μ≥{SPLIT})", h_color),
        (easy_mask, f"easy (μ<{SPLIT})",  e_color),
    ]):
        tr_idx = np.intersect1d(idx_train, np.where(mask)[0])
        te_idx = np.intersect1d(idx_test,  np.where(mask)[0])

        poly  = PolynomialFeatures(degree=2, include_bias=True)
        X_tr  = poly.fit_transform(X_mu_var[tr_idx])
        X_te  = poly.transform(X_mu_var[te_idx])
        model = LinearRegression().fit(X_tr, y_data[tr_idx])
        r2    = r2_score(y_data[te_idx], model.predict(X_te))

        mu_g   = np.linspace(X_mean[mask].min(), X_mean[mask].max(), 25)
        var_g  = np.linspace(X_var[mask].min(),  X_var[mask].max(),  25)
        Mg, Vg = np.meshgrid(mu_g, var_g)
        Z = model.predict(
            poly.transform(np.column_stack([Mg.ravel(), Vg.ravel()]))
        ).reshape(Mg.shape)

        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.scatter(X_mean[te_idx], X_var[te_idx], y_data[te_idx],
                   color="#aaaaaa", s=8, alpha=0.4)
        ax.plot_surface(Mg, Vg, Z, alpha=0.45, cmap=cmap)
        ax.set_xlabel("μ", labelpad=4, fontsize=9)
        ax.set_ylabel("σ²", labelpad=4, fontsize=9)
        ax.set_zlabel(ylabel, labelpad=4, fontsize=9)
        ax.set_title(f"{label}  R²={r2:.3f}", pad=8, fontsize=10)
        ax.view_init(elev=22, azim=45)
        ax.tick_params(axis='both', labelsize=7)

    plt.subplots_adjust(left=0.08, right=0.92, wspace=0.05)
    fname = f"rq2_regime_surface_{target}.png"
    plt.savefig(fname, dpi=130, bbox_inches=None, pad_inches=0.3)
    plt.show()
    print(f"saved: {fname}")