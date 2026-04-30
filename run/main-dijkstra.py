import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.colors as mcolors

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

    # build graph and assign resistance to edges 
    G = nx.grid_2d_graph(height, width)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = np.random.choice(range(11), p=p)

    # virtual source and sink
    G.add_node("source")
    G.add_node("sink")
    for j in range(width):
        G.add_edge("source", (0, j), weight=0)
        G.add_edge((height - 1, j), "sink", weight=0)

    # find shortest path
    path = nx.shortest_path(G, source="source", target="sink", weight='weight')
    actual_path = [node for node in path if isinstance(node, tuple)]

    # statistics
    total_res  = nx.path_weight(G, actual_path, weight='weight')
    path_len   = len(actual_path) - 1
    tortuosity = path_len / height

    path_res_values = [
        G[actual_path[i]][actual_path[i + 1]]['weight']
        for i in range(len(actual_path) - 1)
    ]
    avg_path_res = np.mean(path_res_values) if path_res_values else 0
    med_path_res = np.median(path_res_values) if path_res_values else 0

    # ── plotting ────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a1a')

    # node position: (row, col) → x=col, y=row (y flipped so row 0 is top)
    def pos(node):
        return node[1], height - 1 - node[0]

    # collect line segments and their resistance values
    segments  = []
    edge_vals = []

    for u, v, data in G.edges(data=True):
        if not isinstance(u, tuple) or not isinstance(v, tuple):
            continue   # skip source/sink connections
        x0, y0 = pos(u)
        x1, y1 = pos(v)
        segments.append([(x0, y0), (x1, y1)])
        edge_vals.append(data['weight'])

    # use a grey colormap: low resistance = light, high = dark
    cmap_bg = plt.cm.Greys_r
    norm_bg = mcolors.Normalize(vmin=0, vmax=10)

    lc_bg = mc.LineCollection(segments, cmap=cmap_bg, norm=norm_bg,
                               linewidths=1.2, alpha=0.55, zorder=1)
    lc_bg.set_array(np.array(edge_vals))
    ax.add_collection(lc_bg)

    cbar = plt.colorbar(lc_bg, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Edge Resistance (R)", color='#aaaaaa')
    cbar.ax.yaxis.set_tick_params(color='#aaaaaa')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaaaaa')

    # ── draw the PATH edges as lightning glow ───────────────────────────
    # build list of (x,y) coordinates for consecutive path edges
    path_segments = []
    for i in range(len(actual_path) - 1):
        x0, y0 = pos(actual_path[i])
        x1, y1 = pos(actual_path[i + 1])
        path_segments.append([(x0, y0), (x1, y1)])

    # glow layers — outermost to innermost
    glow_layers = [
        ('#ffffff', 8,   0.04),
        ('#ffe066', 5,   0.12),
        ('#ffd700', 3,   0.35),
        ('#ffec00', 1.8, 0.85),
        ('#ffffff', 0.6, 0.90),
    ]

    for color, lw, alpha in glow_layers:
        lc = mc.LineCollection(path_segments, colors=color,
                               linewidths=lw, alpha=alpha, zorder=3)
        ax.add_collection(lc)

    # start and end markers
    sx, sy = pos(actual_path[0])
    ex, ey = pos(actual_path[-1])
    ax.plot(sx, sy, 'o', color='#ffffff', markersize=7, zorder=5)
    ax.plot(ex, ey, 'v', color='#ffd700', markersize=9, zorder=5)

    # ── axis limits and styling ─────────────────────────────────────────
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')

    # ── statistics box ──────────────────────────────────────────────────
    stats_text = (
        f"--- Path Statistics ---\n"
        f"Total Resistance : {total_res}\n"
        f"Path Length      : {path_len} edges\n"
        f"Tortuosity       : {tortuosity:.3f}\n"
        f"Avg Res/Edge     : {avg_path_res:.2f}\n"
        f"Median Res/Edge  : {med_path_res:.1f}"
    )
    props = dict(boxstyle='round', facecolor='#1a1a1a',
                 edgecolor='#ffd700', linewidth=1.2, alpha=0.85)
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#ffd700', bbox=props,
            family='monospace')

    plt.title(f"Lightning Path ({height}x{width} grid)",
              fontsize=13, pad=14, color='#cccccc')
    plt.xlabel("Grid Width",  color='#aaaaaa')
    plt.ylabel("Grid Height", color='#aaaaaa')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_simulation()