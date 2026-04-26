import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def get_user_input():
    print("--- 8-Connected Dijkstra Simulation ---")
    h = int(input("Enter grid height (e.g., 30): "))
    w = h * 2 
    
    print("\nEnter 11 probabilities for R=0 to R=10 (must sum to 1.0).")
    print("Example: 0.45 0 0 0 0 0 0 0 0 0 0.55")
    p_input = input("Enter probabilities (or 'u' for uniform): ")
    
    if p_input.lower() == 'u':
        probs = [1/11] * 11
    else:
        probs = [float(x) for x in p_input.split()]
        if not np.isclose(sum(probs), 1.0):
            probs = np.array(probs) / np.sum(probs)
            
    return h, w, probs

def run_simulation():
    # Removed num_paths input
    height, width, p = get_user_input()
    
    # Setup plotting
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Generate a master weight matrix for the environment
    weight_matrix = np.random.choice(range(11), size=(height, width), p=p)

    # 1. Create Graph for 8-connectivity
    G = nx.Graph()
    for r in range(height):
        for c in range(width):
            # Standard neighbors (Right and Down)
            if c + 1 < width:
                G.add_edge((r, c), (r, c+1), weight=weight_matrix[r, c+1])
            if r + 1 < height:
                G.add_edge((r, c), (r+1, c), weight=weight_matrix[r+1, c])
            
            # DIAGONAL neighbors (8-connectivity)
            if r + 1 < height and c + 1 < width:
                G.add_edge((r, c), (r+1, c+1), weight=weight_matrix[r+1, c+1])
            if r + 1 < height and c - 1 >= 0:
                G.add_edge((r, c), (r+1, c-1), weight=weight_matrix[r+1, c-1])

    # 2. Virtual Nodes (Super Source/Sink)
    G.add_node("source")
    G.add_node("sink")
    for j in range(width):
        G.add_edge("source", (0, j), weight=weight_matrix[0, j])
        G.add_edge((height-1, j), "sink", weight=0)

    # 3. Solve Dijkstra
    path = nx.shortest_path(G, source="source", target="sink", weight='weight')
    actual_path = [node for node in path if isinstance(node, tuple)]
    
    # --- Statistics Calculation ---
    total_res = nx.path_weight(G, actual_path, weight='weight')
    path_len = len(actual_path) - 1
    tortuosity = path_len / height
    
    # Extract weight values along the path
    path_weights = [weight_matrix[node[0], node[1]] for node in actual_path]
    avg_res = np.mean(path_weights)
    med_res = np.median(path_weights)

    # 4. Plotting
    path_x = [n[1] for n in actual_path]
    path_y = [n[0] for n in actual_path]
    
    # Background Heatmap
    im = ax.imshow(weight_matrix, cmap='magma', alpha=0.4, origin='upper')
    
    # The Path (Double layered for "glow" effect)
    ax.plot(path_x, path_y, color='white', linewidth=3, alpha=0.5)
    ax.plot(path_x, path_y, color='gold', linewidth=1.5, alpha=1.0, label="Dijkstra Path")
    
    # 5. On-Graph Statistics Box
    stats_text = (
        f"--- Path Statistics ---\n"
        f"Total Resistance: {total_res}\n"
        f"Path Length:      {path_len} steps\n"
        f"Tortuosity:       {tortuosity:.3f}\n"
        f"Avg Res/Step:     {avg_res:.2f}\n"
        f"Median Res/Step:  {med_res:.1f}"
    )
    
    props = dict(boxstyle='round', facecolor='black', alpha=0.7)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='white', bbox=props, family='monospace')

    # Formatting
    ax.set_aspect('equal', adjustable='box')
    plt.title(f"8-Connected Dijkstra Path Analysis\nGrid: {height}x{width}", fontsize=14)
    plt.xlabel("Grid Width")
    plt.ylabel("Grid Height")
    plt.colorbar(im, label="Local Resistance (R)", fraction=0.046, pad=0.04)
    plt.grid(False) 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()