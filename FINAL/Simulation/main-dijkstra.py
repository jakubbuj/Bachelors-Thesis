import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.colors as mcolors

def get_user_input():
    print("--- Lightning Path Simulation Parameters ---")
    h = int(input("Enter grid height (e.g., 50): "))
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

    G = nx.grid_2d_graph(height, width)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.choice(range(11), p=p)

    G.add_node("source")
    G.add_node("sink")
    for j in range(width):
        G.add_edge("source", (0, j), weight=0)
        G.add_edge((height - 1, j), "sink", weight=0)

    path = nx.shortest_path(G, source="source", target="sink", weight='weight')
    actual_path = [node for node in path if isinstance(node, tuple)]

    total_res  = nx.path_weight(G, actual_path, weight='weight')
    path_len   = len(actual_path) - 1
    tortuosity = path_len / height

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    def pos(node):
        return node[1], height - 1 - node[0]

    segments  = []
    edge_vals = []

    for u, v, data in G.edges(data=True):
        if not isinstance(u, tuple) or not isinstance(v, tuple):
            continue
        x0, y0 = pos(u)
        x1, y1 = pos(v)
        segments.append([(x0, y0), (x1, y1)])
        edge_vals.append(data['weight'])

    edge_arr = np.array(edge_vals)

    # map R=0 to 0.75 (light grey) and R=10 to 0.08 (near black)
    # so even R=0 edges are visible against the white background
    grey_vals = 0.75 - (edge_arr / 10.0) * 0.67
    colors_bg = plt.cm.Greys(grey_vals)

    lc = mc.LineCollection(segments, colors=colors_bg,
                           linewidths=0.7, zorder=1)
    ax.add_collection(lc)

    # manual colorbar
    sm = plt.cm.ScalarMappable(cmap='Greys',
                           norm=mcolors.Normalize(vmin=0, vmax=10))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Edge resistance R", fontsize=10)

    # ── optimal path: solid red line ────────────────────────────────────
    path_segments = []
    for i in range(len(actual_path) - 1):
        x0, y0 = pos(actual_path[i])
        x1, y1 = pos(actual_path[i + 1])
        path_segments.append([(x0, y0), (x1, y1)])

    lc_path = mc.LineCollection(path_segments, colors='red',
                                 linewidths=2.0, zorder=2)
    ax.add_collection(lc_path)

    # entry and exit markers
    sx, sy = pos(actual_path[0])
    ex, ey = pos(actual_path[-1])
    ax.plot(sx, sy, 'o', color='red', markersize=6, zorder=3)
    ax.plot(ex, ey, 's', color='red', markersize=6, zorder=3)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(
        f"Optimal path on a {height}x{width} grid  |  "
        f"R = {total_res}   tau = {tortuosity:.3f}",
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig("lightning_path_thesis.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.show()
    print("saved: lightning_path_thesis.png")


if __name__ == "__main__":
    run_simulation()