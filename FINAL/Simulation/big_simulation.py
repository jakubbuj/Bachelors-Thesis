"""
Data collection for lightning path thesis
Jakub Bujak

Runs the Dijkstra simulation over many resistance distributions and
records average path statistics for each one. The idea is to build
a dataset where each row is one distribution type + what paths through
it look like on average. This is then used to fit predictive formulas.

Output: data.csv

I tested 7 types of distributions (234 total):
  - bimodal: only R=0 and R=10, varying how many free edges there are
  - uniform range: uniform over some window [a,b]
  - obstacles: increasing fraction of R=10 edges
  - low resistance: increasing fraction of R=0 edges
  - exponential: probability falls off exponentially with R
  - fixed mean: mean stays at 5, variance changes
  - random: randomly sampled distributions for robustness testing
"""

import networkx as nx
import numpy as np
import pandas as pd
import time

# grid dimensions and number of runs per distribution
# set H=50 for the final version, 30 is faster for testing
H      = 50
W      = H * 2
RUNS   = 200
OUTPUT = "data100.csv"


def calculate_fractal_dimension(path_coords, grid_size):
    # box-counting estimate of fractal dimension
    # checks how many boxes of decreasing size are needed to cover the path
    binary_grid = np.zeros(grid_size)
    for r, c in path_coords:
        binary_grid[r, c] = 1
    sizes = [1, 2, 4, 8]
    counts = []
    for s in sizes:
        count = sum(
            1
            for r in range(0, grid_size[0], s)
            for c in range(0, grid_size[1], s)
            if np.any(binary_grid[r:r+s, c:c+s])
        )
        counts.append(count)
    # slope of log(N) vs log(1/eps) gives the dimension
    coeffs = np.polyfit(np.log(1.0 / np.array(sizes)), np.log(counts), 1)
    return coeffs[0]


def single_run(height, width, probs):
    # build grid, assign random resistances, find cheapest path top to bottom
    G = nx.grid_2d_graph(height, width)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.choice(range(11), p=probs)

    # virtual source connects to entire top row, sink connects from entire bottom row
    # this lets the path start and end anywhere horizontally
    G.add_node("source")
    G.add_node("sink")
    for j in range(width):
        G.add_edge("source", (0, j), weight=0)
        G.add_edge((height - 1, j), "sink", weight=0)

    path = nx.shortest_path(G, source="source", target="sink", weight='weight')
    actual_path = [node for node in path if isinstance(node, tuple)]

    x_coords     = [n[1] for n in actual_path]
    total_res    = nx.path_weight(G, actual_path, weight='weight')
    path_len     = len(actual_path)
    max_lat      = max(x_coords) - min(x_coords)
    f_dim        = calculate_fractal_dimension(actual_path, (height, width))
    edge_weights = [
        G.edges[actual_path[k], actual_path[k+1]]['weight']
        for k in range(len(actual_path) - 1)
    ]
    z_ratio = edge_weights.count(0) / len(edge_weights) if edge_weights else 0

    return {
        "total_resistance": total_res,
        "tortuosity":       path_len / height,
        "max_lateral_dev":  max_lat,
        "fractal_dim":      f_dim,
        "zero_cost_ratio":  z_ratio,
    }


def run_distribution(probs, n_runs=RUNS):
    # run the simulation n_runs times for one distribution and return averages
    probs = np.array(probs, dtype=float)
    probs /= probs.sum()

    records = [single_run(H, W, probs) for _ in range(n_runs)]
    df      = pd.DataFrame(records)

    return {
        "mean_tortuosity":      df["tortuosity"].mean(),
        "std_tortuosity":       df["tortuosity"].std(),
        "mean_resistance":      df["total_resistance"].mean(),
        "std_resistance":       df["total_resistance"].std(),
        "mean_fractal_dim":     df["fractal_dim"].mean(),
        "std_fractal_dim":      df["fractal_dim"].std(),
        "mean_lateral_dev":     df["max_lateral_dev"].mean(),
        "std_lateral_dev":      df["max_lateral_dev"].std(),
        "mean_zero_cost_ratio": df["zero_cost_ratio"].mean(),
    }


# --- distribution families ---
# each function returns a list of (label, probability_vector) pairs


def family_bimodal():
    # only R=0 (free) and R=10 (blocked) are possible
    # sweeping p0 from 0 to 1 lets us observe the percolation phase transition
    # where resistance suddenly collapses as free edges form a connected path
    dists = []
    for p0 in np.arange(0.0, 1.01, 0.025):
        p      = [0.0] * 11
        p[0]   = round(float(p0), 6)
        p[10]  = round(float(1.0 - p0), 6)
        dists.append((f"bimodal_p0={p0:.3f}", p))
    return dists


def family_uniform_range():
    # uniform probability over the window [a, b], zero everywhere else
    # varying a controls the mean (position on the scale)
    # varying b-a controls the variance (width of the window)
    # this lets us study mean and variance effects somewhat independently
    dists = []
    for a in range(0, 9):
        for b in range(a + 1, 11):
            p = [0.0] * 11
            n = b - a + 1
            for i in range(a, b + 1):
                p[i] = 1.0 / n
            dists.append((f"uniform_{a}to{b}", p))
    return dists


def family_obstacles():
    # a fraction p10 of edges are fully blocked (R=10)
    # the rest are uniformly distributed over R=0..9
    # models a grid with physical barriers, increasing obstruction
    dists = []
    for p10 in np.arange(0.0, 0.96, 0.05):
        p10   = round(float(p10), 6)
        p     = [round((1.0 - p10) / 10, 8)] * 11
        p[10] = p10
        dists.append((f"obstacles_p10={p10:.3f}", p))
    return dists


def family_low_resistance():
    # mirror of the obstacles family
    # a fraction p0 of edges are completely free (R=0)
    # the rest are uniform over R=1..10
    # tests how the path exploits an increasing supply of free edges
    dists = []
    for p0 in np.arange(0.0, 0.96, 0.05):
        p0   = round(float(p0), 6)
        p    = [round((1.0 - p0) / 10, 8)] * 11
        p[0] = p0
        dists.append((f"low_res_p0={p0:.3f}", p))
    return dists


def family_exponential():
    # p_k proportional to exp(lambda * k) for k = 0..10
    # lambda < 0: cheap edges dominate (easy environment)
    # lambda = 0: reduces to uniform
    # lambda > 0: expensive edges dominate (hard environment)
    # motivated by exponential distributions in physical random media
    dists  = []
    values = np.arange(11)
    for lam in np.arange(-0.8, 0.81, 0.1):
        lam = round(float(lam), 2)
        raw = np.exp(lam * values)
        p   = (raw / raw.sum()).tolist()
        dists.append((f"exponential_lam={lam:+.2f}", p))
    return dists


def family_fixed_mean():
    # all distributions in this family have mean = 5
    # constructed as a mixture: f * spike(R=5) + (1-f) * uniform
    # f=0 gives pure uniform (high variance), f=0.95 gives near-spike (low variance)
    # used to test whether variance affects path properties independently of the mean
    # if tortuosity changes across this family, variance must go into the formula
    dists     = []
    uniform_p = np.full(11, 1 / 11)
    spike_p   = np.zeros(11)
    spike_p[5] = 1.0

    for f in np.arange(0.0, 0.96, 0.05):
        f = round(float(f), 4)
        p = ((1 - f) * uniform_p + f * spike_p).tolist()
        dists.append((f"fixed_mean_f={f:.2f}", p))
    return dists


def family_random_dirichlet(n=80, seed=42):
    # randomly sampled distributions from the full simplex over {0,...,10}
    # alpha=0.5 gives sparse peaked distributions (first half)
    # alpha=1.5 gives smoother spread-out ones (second half)
    # used to check that the formula generalises beyond the structured families
    rng   = np.random.default_rng(seed)
    dists = []
    for i in range(n):
        alpha = 0.5 if i < n // 2 else 1.5
        p     = rng.dirichlet([alpha] * 11).tolist()
        dists.append((f"random_{i:03d}_a={alpha}", p))
    return dists


def main():
    all_dists  = []
    all_dists += family_bimodal()
    all_dists += family_uniform_range()
    all_dists += family_obstacles()
    all_dists += family_low_resistance()
    all_dists += family_exponential()
    all_dists += family_fixed_mean()
    all_dists += family_random_dirichlet()

    print("=" * 62)
    print("   LIGHTNING PATH THESIS — DATA COLLECTION")
    print("=" * 62)
    print(f"   distributions : {len(all_dists)}")
    print(f"   runs each     : {RUNS}")
    print(f"   grid size     : {H} x {W}")
    print(f"   total runs    : {len(all_dists) * RUNS:,}")
    print(f"   output        : {OUTPUT}")
    print("=" * 62)

    rows    = []
    t_start = time.time()
    values  = np.arange(11)

    for idx, (label, probs) in enumerate(all_dists):
        p = np.array(probs, dtype=float)
        p /= p.sum()

        # compute distribution summary stats used as formula inputs later
        mu   = float(np.dot(values, p))
        var  = float(np.dot((values - mu) ** 2, p))
        skew = float(np.dot((values - mu) ** 3, p) / (var ** 1.5 + 1e-9))

        path_stats = run_distribution(p, n_runs=RUNS)

        row = {
            "dist_id":    idx,
            "dist_label": label,
            **{f"p{i}": round(float(p[i]), 6) for i in range(11)},
            "dist_mean": round(mu,   4),
            "dist_var":  round(var,  4),
            "dist_skew": round(skew, 4),
            **{k: round(v, 4) for k, v in path_stats.items()},
            "n_runs": RUNS,
            "grid_H": H,
            "grid_W": W,
        }
        rows.append(row)

        elapsed  = time.time() - t_start
        eta_min  = (elapsed / (idx + 1)) * (len(all_dists) - idx - 1) / 60
        print(f"[{idx+1:>3}/{len(all_dists)}]  {label:<44}"
              f"  tau={path_stats['mean_tortuosity']:.3f}"
              f"  R={path_stats['mean_resistance']:7.2f}"
              f"  ETA {eta_min:.1f} min")

        # save progress every 20 so we don't lose everything if it crashes
        if (idx + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT, index=False)
            print(f"           saved checkpoint -> {OUTPUT}\n")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT, index=False)
    total_min = (time.time() - t_start) / 60

    print("\n" + "=" * 62)
    print(f"   done. {len(df)} rows saved to '{OUTPUT}'")
    print(f"   total time: {total_min:.1f} minutes")
    print("=" * 62)

    print("\n--- averages per family ---")
    df["family"] = df["dist_label"].str.split("_").str[0]
    cols = ["dist_mean", "dist_var", "mean_tortuosity",
            "mean_resistance", "mean_fractal_dim"]
    print(df.groupby("family")[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()