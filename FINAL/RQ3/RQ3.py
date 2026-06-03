# rq3_analysis.py
# RQ3: greedy algorithm vs dijkstra comparison
#
# produces 4 figures:
#   rq3_comparison.png       - greedy vs dijkstra tortuosity and resistance
#   rq3_gap_by_family.png    - how much worse greedy is per family
#   rq3_formula_transfer.png - does the formula predict both algorithms?
#   rq3_efficiency.png       - efficiency ratio: how close is greedy to optimal?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import networkx as nx
import time

H    = 400     
W    = 800
RUNS = 15

plt.rcParams.update({"font.size": 12, "figure.dpi": 130})

COLORS = {
    "bimodal":        "#e41a1c",
    "uniform":        "#377eb8",
    "obstacles":      "#ff7f00",
    "low_resistance": "#4daf4a",
    "exponential":    "#984ea3",
    "fixed_mean":     "#a65628",
    "random":         "#999999",
}


def get_family(label):
    if label.startswith("bimodal"):   return "bimodal"
    if label.startswith("uniform"):   return "uniform"
    if label.startswith("obstacles"): return "obstacles"
    if label.startswith("low"):       return "low_resistance"
    if label.startswith("exp"):       return "exponential"
    if label.startswith("fixed"):     return "fixed_mean"
    return "random"


# dijkstra
def dijkstra_single_run(height, width, probs):
    G = nx.grid_2d_graph(height, width)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.choice(range(11), p=probs)
    G.add_node("source")
    G.add_node("sink")
    for j in range(width):
        G.add_edge("source", (0, j), weight=0)
        G.add_edge((height - 1, j), "sink", weight=0)
    path   = nx.shortest_path(G, "source", "sink", weight='weight')
    actual = [n for n in path if isinstance(n, tuple)]
    x_coords = [n[1] for n in actual]
    weights  = [G.edges[actual[k], actual[k+1]]['weight']
                for k in range(len(actual)-1)]
    return {
        "total_resistance": sum(weights),
        "tortuosity":       (len(actual) - 1) / height,
        "max_lateral_dev":  max(x_coords) - min(x_coords),
    }


# greedy — gravity penalty: down=0, sideways=+2, up=+8
def greedy_single_run(height, width, probs):
    G = nx.grid_2d_graph(height, width)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.choice(range(11), p=probs)

    curr_r  = 0
    curr_c  = np.random.randint(0, width)
    path    = [(curr_r, curr_c)]
    visited = {(curr_r, curr_c)}
    edge_weights = []

    while curr_r < height - 1:
        options = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                edge_cost = G.edges[(curr_r, curr_c), (nr, nc)]['weight']
                gravity   = 0 if nr > curr_r else 2 if nr == curr_r else 8
                options.append((edge_cost + gravity, edge_cost, nr, nc))
        if not options:
            break
        options.sort()
        _, actual_cost, next_r, next_c = options[0]
        edge_weights.append(actual_cost)
        curr_r, curr_c = next_r, next_c
        path.append((curr_r, curr_c))
        visited.add((curr_r, curr_c))
        if len(path) > height * width:
            break

    x_coords = [n[1] for n in path]
    return {
        "total_resistance": sum(edge_weights),
        "tortuosity":       (len(path) - 1) / height,
        "max_lateral_dev":  max(x_coords) - min(x_coords),
    }


def run_distribution(run_fn, probs, n_runs=RUNS):
    probs   = np.array(probs, dtype=float)
    probs  /= probs.sum()
    t0      = time.perf_counter()
    records = [run_fn(H, W, probs) for _ in range(n_runs)]
    elapsed = time.perf_counter() - t0
    df      = pd.DataFrame(records)
    return {
        "mean_resistance": df["total_resistance"].mean(),
        "std_resistance":  df["total_resistance"].std(),
        "mean_tortuosity": df["tortuosity"].mean(),
        "std_tortuosity":  df["tortuosity"].std(),
        "mean_lateral":    df["max_lateral_dev"].mean(),
        "mean_time_ms":    (elapsed / n_runs) * 1000,
    }


df_all = pd.read_csv("/Users/jakubb/Desktop/Thesis/FINAL/data100.csv")
df_all["family"] = df_all["dist_label"].apply(get_family)

subset_rows = []
for fam in df_all["family"].unique():
    fam_df = df_all[df_all["family"] == fam]
    subset_rows.append(fam_df.sample(min(8, len(fam_df)), random_state=42))
subset = pd.concat(subset_rows).reset_index(drop=True)

print(f"comparing greedy vs dijkstra on {len(subset)} distributions")
print(f"grid: {H}x{W}   runs: {RUNS}")

dijk_results   = []
greedy_results = []
t0 = time.time()

for i, row in subset.iterrows():
    probs = np.array([row[f"p{k}"] for k in range(11)], dtype=float)
    probs /= probs.sum()

    d_stats = run_distribution(dijkstra_single_run, probs)
    g_stats = run_distribution(greedy_single_run,   probs)

    dijk_results.append(d_stats)
    greedy_results.append(g_stats)

    done = subset.index.get_loc(i) + 1
    eta  = ((time.time() - t0) / done) * (len(subset) - done) / 60
    print(f"  [{done:>2}/{len(subset)}]  {row['dist_label']:<38}"
          f"  dijk τ={d_stats['mean_tortuosity']:.3f}"
          f"  greedy τ={g_stats['mean_tortuosity']:.3f}"
          f"  ETA {eta:.1f} min")

dijk_df   = pd.DataFrame(dijk_results).add_prefix("dijk_")
greedy_df = pd.DataFrame(greedy_results).add_prefix("greedy_")
comp      = pd.concat([subset.reset_index(drop=True), dijk_df, greedy_df], axis=1)

comp["tau_gap"] = comp["greedy_mean_tortuosity"] - comp["dijk_mean_tortuosity"]
comp["res_gap"] = comp["greedy_mean_resistance"] - comp["dijk_mean_resistance"]

comp["efficiency"] = np.where(
    comp["greedy_mean_resistance"] > 1,
    comp["dijk_mean_resistance"] / comp["greedy_mean_resistance"],
    1.0
)

print(f"\ndone in {(time.time()-t0)/60:.1f} minutes")
legend_handles = [mpatches.Patch(color=c, label=f) for f, c in COLORS.items()]


# figure 1: comparison scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("RQ3 — greedy vs dijkstra\n"
             "points above the diagonal = greedy is worse", fontsize=13)

for ax, xcol, ycol, xlabel, ylabel, title in [
    (axes[0],
     "dijk_mean_tortuosity", "greedy_mean_tortuosity",
     "dijkstra tortuosity τ", "greedy tortuosity τ", "tortuosity"),
    (axes[1],
     "dijk_mean_resistance", "greedy_mean_resistance",
     "dijkstra resistance R", "greedy resistance R", "resistance"),
]:
    for fam, color in COLORS.items():
        sub = comp[comp["family"] == fam]
        ax.scatter(sub[xcol], sub[ycol], c=color, alpha=0.75, s=60)
    lim = [min(comp[xcol].min(), comp[ycol].min()) - 2,
           max(comp[xcol].max(), comp[ycol].max()) + 2]
    ax.plot(lim, lim, "k--", linewidth=1.5, label="equal performance")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("rq3_comparison.png", bbox_inches="tight")
plt.show()
print("saved: rq3_comparison.png")


# figure 2: gap by family
families_present = list(comp["family"].unique())
gap_colors       = [COLORS[f] for f in families_present]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("RQ3 — optimality gap (greedy minus dijkstra)\n"
             "positive = greedy performs worse", fontsize=13)

for ax, col, ylabel, title in [
    (axes[0], "tau_gap", "greedy τ − dijkstra τ", "tortuosity gap by family"),
    (axes[1], "res_gap", "greedy R − dijkstra R", "resistance gap by family"),
]:
    data = [comp[comp["family"] == f][col].values for f in families_present]
    bp   = ax.boxplot(data, patch_artist=True, tick_labels=families_present)
    for patch, color in zip(bp["boxes"], gap_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("rq3_gap_by_family.png", bbox_inches="tight")
plt.show()
print("saved: rq3_gap_by_family.png")


# figure 3: formula transfer
X = np.column_stack([
    comp["dist_mean"],
    comp["dist_mean"]**2,
    comp["dist_var"],
    comp["dist_var"]**2,
])

model_d = LinearRegression().fit(X, comp["dijk_mean_tortuosity"])
model_g = LinearRegression().fit(X, comp["greedy_mean_tortuosity"])
pred_d  = model_d.predict(X)
pred_g  = model_g.predict(X)
r2_d    = r2_score(comp["dijk_mean_tortuosity"],   pred_d)
r2_g    = r2_score(comp["greedy_mean_tortuosity"], pred_g)

point_colors = [COLORS[f] for f in comp["family"]]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("RQ3 — does the formula predict both algorithms?\n"
             "polynomial: τ = a·μ + b·μ² + c·σ² + d·σ⁴ + e", fontsize=13)

for ax, actual, predicted, title in [
    (axes[0], comp["dijk_mean_tortuosity"], pred_d,
     f"dijkstra — R² = {r2_d:.3f}"),
    (axes[1], comp["greedy_mean_tortuosity"], pred_g,
     f"greedy — R² = {r2_g:.3f}"),
]:
    ax.scatter(actual, predicted, c=point_colors, alpha=0.75, s=55)
    lim = [min(actual.min(), predicted.min()) - 0.05,
           max(actual.max(), predicted.max()) + 0.05]
    ax.plot(lim, lim, "k--", linewidth=1.5, label="perfect prediction")
    ax.set_xlabel("actual tortuosity τ")
    ax.set_ylabel("predicted tortuosity τ")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("rq3_formula_transfer.png", bbox_inches="tight")
plt.show()
print("saved: rq3_formula_transfer.png")


# figure 4: efficiency ratio
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("RQ3 — greedy efficiency ratio\n"
             "efficiency = dijkstra R / greedy R  "
             "(1.0 = matches optimal, lower = worse)", fontsize=13)

for fam, color in COLORS.items():
    sub = comp[comp["family"] == fam]
    axes[0].scatter(sub["dist_mean"], sub["efficiency"], c=color, alpha=0.75, s=60)
    axes[1].scatter(sub["dist_var"],  sub["efficiency"], c=color, alpha=0.75, s=60)

for ax, xlabel, title in [
    (axes[0], "mean resistance μ",
     "efficiency vs mean resistance\nhigher μ = greedy closer to optimal"),
    (axes[1], "variance σ²",
     "efficiency vs variance\nhigher σ² = greedy further from optimal"),
]:
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2,
               label="greedy = optimal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("efficiency (dijkstra R / greedy R)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("rq3_efficiency.png", bbox_inches="tight")
plt.show()
print("saved: rq3_efficiency.png")


# figure 5: runtime comparison
comp["dijk_time_ms"]   = [r["mean_time_ms"] for r in dijk_results]
comp["greedy_time_ms"] = [r["mean_time_ms"] for r in greedy_results]
comp["speedup"]        = comp["dijk_time_ms"] / comp["greedy_time_ms"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("RQ3 — runtime comparison\n"
             "mean time per single simulation run (ms)", fontsize=13)

for fam, color in COLORS.items():
    sub = comp[comp["family"] == fam]
    axes[0].scatter(sub["dist_mean"], sub["dijk_time_ms"],
                    c=color, alpha=0.75, s=60, marker="o")
    axes[0].scatter(sub["dist_mean"], sub["greedy_time_ms"],
                    c=color, alpha=0.75, s=60, marker="^")

axes[0].set_xlabel("mean μ")
axes[0].set_ylabel("time per run (ms)")
axes[0].set_title("runtime vs mean μ\ncircle = Dijkstra, triangle = greedy")
axes[0].grid(True, alpha=0.3)

# speedup ratio per distribution
for fam, color in COLORS.items():
    sub = comp[comp["family"] == fam]
    axes[1].scatter(sub["dist_mean"], sub["speedup"], c=color, alpha=0.75, s=60)

axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2,
                label="equal runtime")
axes[1].set_xlabel("mean μ")
axes[1].set_ylabel("speedup (Dijkstra time / greedy time)")
axes[1].set_title("speedup ratio vs mean μ\n>1 means greedy is faster")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

# bar chart: mean runtime per algorithm per family
families_present = list(comp["family"].unique())
x = np.arange(len(families_present))
w = 0.35
mean_dijk   = [comp[comp["family"]==f]["dijk_time_ms"].mean()   for f in families_present]
mean_greedy = [comp[comp["family"]==f]["greedy_time_ms"].mean() for f in families_present]

axes[2].bar(x - w/2, mean_dijk,   w, label="Dijkstra", color="#e41a1c", alpha=0.8)
axes[2].bar(x + w/2, mean_greedy, w, label="greedy",   color="#377eb8", alpha=0.8)
axes[2].set_xticks(x)
axes[2].set_xticklabels(families_present, rotation=30, fontsize=9)
axes[2].set_ylabel("mean time per run (ms)")
axes[2].set_title("mean runtime by family")
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3, axis="y")

fig.legend(handles=legend_handles, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.savefig("rq3_runtime.png", bbox_inches="tight")
plt.show()
print("saved: rq3_runtime.png")


# save results
comp.to_csv("rq3_results.csv", index=False)
print("\nsaved: rq3_results.csv")

# thesis numbers
print("\n" + "="*55)
print("  RESULTS ")
print("="*55)
print(f"  distributions compared   : {len(comp)}")
print(f"  mean tortuosity gap      : {comp['tau_gap'].mean():+.4f}")
print(f"  mean resistance gap      : {comp['res_gap'].mean():+.2f}")
print(f"  mean efficiency ratio    : {comp['efficiency'].mean():.4f}")
print(f"  min efficiency ratio     : {comp['efficiency'].min():.4f}  "
      f"({comp.loc[comp['efficiency'].idxmin(), 'dist_label']})")
print(f"  max efficiency ratio     : {comp['efficiency'].max():.4f}  "
      f"({comp.loc[comp['efficiency'].idxmax(), 'dist_label']})")
print(f"  formula R² dijkstra      : {r2_d:.4f}")
print(f"  formula R² greedy        : {r2_g:.4f}")
print(f"\n  mean Dijkstra time/run   : {comp['dijk_time_ms'].mean():.2f} ms")
print(f"  mean greedy time/run     : {comp['greedy_time_ms'].mean():.2f} ms")
print(f"  mean speedup ratio       : {comp['speedup'].mean():.2f}x  (Dijkstra / greedy)")
print("\n  efficiency by family:")
print(comp.groupby("family")["efficiency"].agg(["mean","min","max"]).round(3).to_string())
print("\n  speedup by family:")
print(comp.groupby("family")["speedup"].agg(["mean","min","max"]).round(2).to_string())
print("\n  gap by family:")
print(comp.groupby("family")[["tau_gap","res_gap"]].mean().round(3).to_string())