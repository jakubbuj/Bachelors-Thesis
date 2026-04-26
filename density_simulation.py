import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def run_aligned_simulation(height, width, probs_input, num_runs=50):
    # --- ADDED: Support for 'u' as uniform ---
    if isinstance(probs_input, str) and probs_input.lower() == 'u':
        probs = [1/11] * 11
        print("Using Uniform Distribution (p0...p10 = 1/11)")
    else:
        probs = probs_input
        # Normalization check to ensure probabilities sum to 1.0 
        if not np.isclose(sum(probs), 1.0):
            probs = np.array(probs) / np.sum(probs)

    density_matrix = np.zeros((height, width))
    all_paths = []
    
    # 1. Generate Data
    G_base = nx.grid_2d_graph(height, width) # Formalizing lattice as weighted graph [cite: 89]
    sample_weights = np.zeros((height, width))

    for i in range(num_runs):
        G = G_base.copy()
        for (u, v) in G.edges():
            # Edge resistance R in {0...10} drawn from distribution 
            res = np.random.choice(range(11), p=probs)
            G.edges[u, v]['weight'] = res
            if i == 0: sample_weights[u[0], u[1]] = res

        # Virtual nodes to allow any top node to be the start [cite: 80, 81]
        G.add_node("source")
        G.add_node("sink")
        for j in range(width):
            G.add_edge("source", (0, j), weight=0)
            G.add_edge((height-1, j), "sink", weight=0)

        # Compute minimum cost path [cite: 89]
        path_nodes = nx.shortest_path(G, source="source", target="sink", weight='weight')
        actual_path = [n for n in path_nodes if isinstance(n, tuple)]
        all_paths.append(actual_path)
        
        # Record resulting path statistics for the density matrix [cite: 90]
        for (r, c) in actual_path: 
            density_matrix[r, c] += 1

    # 2. Plotting 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # --- TOP PLOT: Density Heatmap ---
    im1 = ax1.imshow(density_matrix, cmap='inferno', origin='upper', aspect='equal')
    ax1.set_title(f"Global Path Density ({num_runs} Runs)")
    
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax1, label='Visit Count')

    # --- BOTTOM PLOT: Individual Realizations ---
    ax2.imshow(sample_weights, cmap='binary', alpha=0.15, origin='upper', aspect='equal')
    
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cax2.axis('off') 

    colors = cm.get_cmap('tab10', 10) 
    for i, path in enumerate(all_paths):
        px = [n[1] for n in path]
        py = [n[0] for n in path]
        if i < 10:
            ax2.plot(px, py, color=colors(i), alpha=0.9, linewidth=1.5)
        else:
            ax2.plot(px, py, color='gray', alpha=0.03, linewidth=1)

    ax2.set_title("Stochastic Realizations (10 Multi-Color Paths)")
    ax2.set_xlabel("Grid Width")
    ax2.set_ylabel("Grid Height")
    
    plt.tight_layout()
    plt.show()

# Example: Run with uniform distribution
run_aligned_simulation(50, 100, 'u', num_runs=100)