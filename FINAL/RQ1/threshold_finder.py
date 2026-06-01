# threshold_fine_sweep.py
# Runs a fine-grained bimodal sweep around the transition region
# to find the exact empirical percolation threshold.
#
# Instead of the coarse step of 0.025 used in the main data collection,
# this uses steps of 0.005 between p0=0.45 and p0=0.56 with 500 runs
# per point - giving a much more precise estimate of where resistance
# first drops to zero.

import networkx as nx
import numpy as np
import pandas as pd

H    = 50
W    = 100
RUNS = 500

p0_sweep = np.arange(0.45, 0.561, 0.005)
results  = []

for p0 in p0_sweep:
    p0 = round(float(p0), 4)

    # bimodal: only R=0 and R=10
    p      = [0.0] * 11
    p[0]   = p0
    p[10]  = round(1.0 - p0, 4)
    probs  = np.array(p)

    R_vals = []
    for _ in range(RUNS):
        G = nx.grid_2d_graph(H, W)
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.choice(range(11), p=probs)
        G.add_node("source")
        G.add_node("sink")
        for j in range(W):
            G.add_edge("source", (0, j), weight=0)
            G.add_edge((H - 1, j), "sink", weight=0)
        path   = nx.shortest_path(G, "source", "sink", weight='weight')
        actual = [n for n in path if isinstance(n, tuple)]
        R_vals.append(nx.path_weight(G, actual, weight='weight'))

    mean_R = np.mean(R_vals)
    results.append({"p0": p0, "mean_R": mean_R, "std_R": np.std(R_vals)})
    print(f"p0={p0:.4f}  mean_R={mean_R:.4f}")

df = pd.DataFrame(results)

# threshold = first p0 where mean resistance is exactly 0
strictly_zero = df[df["mean_R"] == 0.0]
threshold = strictly_zero["p0"].min() if len(strictly_zero) > 0 else None

print(f"\nEmpirical threshold: pc = {threshold}")
print(f"Directed theory:     pc = 0.5000")