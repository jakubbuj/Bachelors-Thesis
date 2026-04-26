import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_fractal_dimension(path_coords, grid_size):
    """
    Measures 'roughness' or fractal-like features of the shape[cite: 86].
    Estimates D by seeing how many boxes of size epsilon are needed to cover the path.
    """
    binary_grid = np.zeros(grid_size)
    for r, c in path_coords:
        binary_grid[r, c] = 1
    
    # We test box sizes (epsilon) as powers of 2 for stability
    sizes = [1, 2, 4, 8] 
    counts = []
    for s in sizes:
        count = 0
        for r in range(0, grid_size[0], s):
            for c in range(0, grid_size[1], s):
                if np.any(binary_grid[r:r+s, c:c+s]):
                    count += 1
        counts.append(count)
    
    # Regression slope of log(N) vs log(1/eps) gives Fractal Dimension [cite: 93]
    coeffs = np.polyfit(np.log(1.0/np.array(sizes)), np.log(counts), 1)
    return coeffs[0]

def run_simulation(height, width, probs_input, num_runs, filename="lightning_stats.csv"):
    """
    Runs systematic experiments varying the resistance distribution[cite: 91].
    """
    # Check for uniform distribution request
    if isinstance(probs_input, str) and probs_input.lower() == 'u':
        probs = [1/11] * 11
        print("Using Uniform Distribution (p0...p10 = 1/11)")
    else:
        probs = probs_input
        # Ensure it sums to 1.0 [cite: 80]
        if not np.isclose(sum(probs), 1.0):
            probs = np.array(probs) / np.sum(probs)

    results = []
    print(f"Starting {num_runs} simulations...")

    for i in range(num_runs):
        # 1. Build Grid & Assign Resistances R in {0...10} [cite: 80]
        G = nx.grid_2d_graph(height, width)
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.choice(range(11), p=probs)

        # 2. Add Virtual Nodes (Source at top, Sink at bottom boundary) [cite: 80]
        G.add_node("source")
        G.add_node("sink")
        for j in range(width):
            G.add_edge("source", (0, j), weight=0)
            G.add_edge((height-1, j), "sink", weight=0)

        # 3. Compute Shortest Path (Minimizes sum of edge resistances) [cite: 81]
        path = nx.shortest_path(G, source="source", target="sink", weight='weight')
        actual_path = [node for node in path if isinstance(node, tuple)]
        
        # --- STATISTICS CALCULATION ---
        x_coords = [n[1] for n in actual_path]
        start_x = x_coords[0]
        end_x = x_coords[-1]
        
        # a) Total Resistance [cite: 86]
        total_res = nx.path_weight(G, actual_path, weight='weight')
        
        # b) Path Length (Geometric steps) [cite: 86]
        path_len = len(actual_path)
        
        # c) Max Lateral Deviation [cite: 86]
        max_lat_dev = max(x_coords) - min(x_coords)
        
        # d) Net Drift (End vs Start)
        net_drift = end_x - start_x
        
        # e) Avg Deviation from Start column
        avg_dev_start = np.mean([abs(x - start_x) for x in x_coords])
        
        # f) Fractal Dimension (Roughness) [cite: 86]
        f_dim = calculate_fractal_dimension(actual_path, (height, width))
        
        # g) Zero-Cost Ratio (Path efficiency)
        weights = [G.edges[actual_path[k], actual_path[k+1]]['weight'] for k in range(len(actual_path)-1)]
        z_ratio = weights.count(0) / len(weights) if len(weights) > 0 else 0

        results.append({
            "run": i + 1,
            "total_resistance": total_res,
            "path_length": path_len,
            "tortuosity": path_len / height, # Scaling of path length with height [cite: 93]
            "max_lateral_deviation": max_lat_dev,
            "net_drift": net_drift,
            "avg_dev_from_start": round(avg_dev_start, 3),
            "fractal_dimension": round(f_dim, 4),
            "zero_cost_ratio": round(z_ratio, 4)
        })

    # 5. Save Results to CSV [cite: 90]
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    return df

# --- Example Usage ---
H, W = 50, 100

# Run with Uniform distribution
# run_comprehensive_simulation(H, W, 'u', num_runs=1000, filename="1000uniform_results.csv")

# Run with Specific distribution
# specific_p = u
# specific_p = [0.006, 0.006, 0.006, 0.006, 0.006, 0.005, 0.005, 0.005, 0.005, 0.15, 0.80] #obstacles
specific_p = [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5] #bimodal
run_simulation(H, W, specific_p, num_runs=1000, filename="bimodal.csv")