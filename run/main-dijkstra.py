import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def get_user_input():
    print("--- Lightning Path Simulation Parameters ---")
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
    height, width, p = get_user_input()
    G_base = nx.grid_2d_graph(height, width)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#2b2b2b')

    weight_matrix = np.zeros((height, width))
    G = G_base.copy()
    
    # Assign weights
    for (u, v) in G.edges():
        res = np.random.choice(range(11), p=p)
        G.edges[u, v]['weight'] = res
        weight_matrix[u[0], u[1]] = res

    # Virtual Nodes
    G.add_node("source")
    G.add_node("sink")
    for j in range(width):
        G.add_edge("source", (0, j), weight=0)
        G.add_edge((height-1, j), "sink", weight=0)

    # Solve Dijkstra
    path = nx.shortest_path(G, source="source", target="sink", weight='weight')
    actual_path = [node for node in path if isinstance(node, tuple)]
    
    # --- Statistics Calculation ---
    total_res = nx.path_weight(G, actual_path, weight='weight')
    path_len = len(actual_path) - 1
    tortuosity = path_len / height
    
    path_res_values = []
    for i in range(len(actual_path)-1):
        path_res_values.append(G[actual_path[i]][actual_path[i+1]]['weight'])
    
    avg_path_res = np.mean(path_res_values) if path_res_values else 0
    med_path_res = np.median(path_res_values) if path_res_values else 0

    # --- Plotting ---
    path_x = [n[1] for n in actual_path]
    path_y = [n[0] for n in actual_path]

    # Stormy sky colormap
    stormy_colors = ['#282828', '#323232', '#3c3c3c', '#464646', '#505050', '#5a5a5a']
    stormy_cmap = LinearSegmentedColormap.from_list('stormy', stormy_colors)

    im = ax.imshow(weight_matrix, cmap=stormy_cmap, alpha=0.75, origin='upper')

    # Lightning glow layers (outermost → innermost)
    ax.plot(path_x, path_y, color='#ffffff', linewidth=7,   alpha=0.04)
    ax.plot(path_x, path_y, color='#ffe066', linewidth=5,   alpha=0.12)
    ax.plot(path_x, path_y, color='#ffd700', linewidth=3,   alpha=0.35)
    ax.plot(path_x, path_y, color='#ffec00', linewidth=1.8, alpha=0.85)
    ax.plot(path_x, path_y, color='#ffffff', linewidth=0.6, alpha=0.9)

    # Start / end markers
    ax.plot(path_x[0],  path_y[0],  'o', color='#ffffff', markersize=6, zorder=5)
    ax.plot(path_x[-1], path_y[-1], 'v', color='#ffd700', markersize=8, zorder=5)

    # Statistics box
    stats_text = (
        f"--- Path Statistics ---\n"
        f"Total Resistance: {total_res}\n"
        f"Path Length: {path_len} units\n"
        f"Tortuosity: {tortuosity:.3f}\n"
        f"Avg Res/Step: {avg_path_res:.2f}\n"
        f"Median Res/Step: {med_path_res:.1f}"
    )
    props = dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#ffd700',
                 linewidth=1.2, alpha=0.85)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#ffd700', bbox=props, family='monospace')

    # Axis / title styling
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')

    plt.title(f"Lightning Path Analysis ({height}x{width})", fontsize=14,
              pad=20, color='#cccccc')
    plt.xlabel("Grid Width",  color='#aaaaaa')
    plt.ylabel("Grid Height", color='#aaaaaa')

    cbar = plt.colorbar(im, label="Local Resistance (R)", fraction=0.046, pad=0.04)
    cbar.set_label("Local Resistance (R)", color='#aaaaaa')
    cbar.ax.yaxis.set_tick_params(color='#aaaaaa')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaaaaa')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()