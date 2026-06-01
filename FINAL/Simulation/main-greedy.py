import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.colors as mcolors

def get_user_input():
    print("--- Greedy Lightning Simulation ---")
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

    # greedy — gravity penalty: down=0, sideways=+2, up=+8
    curr_r = 0
    curr_c = np.random.randint(0, width)
    path   = [(curr_r, curr_c)]
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

    total_res  = sum(edge_weights)
    path_len   = len(path) - 1
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

    edge_arr  = np.array(edge_vals)
    grey_vals = 0.75 - (edge_arr / 10.0) * 0.67
    colors_bg = plt.cm.Greys(grey_vals)

    lc = mc.LineCollection(segments, colors=colors_bg, linewidths=0.7, zorder=1)
    ax.add_collection(lc)

    sm = plt.cm.ScalarMappable(cmap='Greys_r',
                               norm=mcolors.Normalize(vmin=0, vmax=10))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Edge resistance R", fontsize=10)

    path_segments = []
    for i in range(len(path) - 1):
        x0, y0 = pos(path[i])
        x1, y1 = pos(path[i + 1])
        path_segments.append([(x0, y0), (x1, y1)])

    lc_path = mc.LineCollection(path_segments, colors='red',
                                linewidths=2.0, zorder=2)
    ax.add_collection(lc_path)

    sx, sy = pos(path[0])
    ex, ey = pos(path[-1])
    ax.plot(sx, sy, 'o', color='red', markersize=6, zorder=3)
    ax.plot(ex, ey, 's', color='red', markersize=6, zorder=3)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(
        f"Greedy path on a {height}x{width} grid  |  "
        f"R = {total_res}   tau = {tortuosity:.3f}",
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig("lightning_path_greedy.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.show()
    print("saved: lightning_path_greedy.png")


if __name__ == "__main__":
    run_simulation()