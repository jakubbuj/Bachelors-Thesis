import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def get_user_input():
    print("--- 8-Neighbor Greedy Lightning Simulation ---")
    h = int(input("Enter grid height (e.g., 50): "))
    w = h * 2 
    
    print("\nEnter 11 probabilities for R=0 to R=10 (must sum to 1.0).")
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
    
    # --- Stormy sky figure background ---
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')   # very dark blue-grey (outer border)
    ax.set_facecolor('#2b2b2b')           # dark stormy grey canvas
    
    # Generate background weight matrix
    bg_weights = np.random.choice(range(11), size=(height, width), p=p)

    # Simulation logic for a single path
    curr_r = 0
    curr_c = np.random.randint(0, width)
    
    path = [(curr_r, curr_c)]
    visited = set([(curr_r, curr_c)])
    total_resistance = 0
    
    while curr_r < height - 1:
        options = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                
                nr, nc = curr_r + dr, curr_c + dc
                
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited:
                    base_cost = bg_weights[nr, nc]
                    
                    if nr > curr_r:
                        gravity_penalty = 0
                    elif nr == curr_r:
                        gravity_penalty = 2
                    else:
                        gravity_penalty = 8
                        
                    options.append((base_cost + gravity_penalty, nr, nc, base_cost))
        
        if not options:
            break
        
        options.sort()
        best_total_cost, next_r, next_c, raw_res = options[0]
        
        total_resistance += raw_res
        curr_r, curr_c = next_r, next_c
        path.append((curr_r, curr_c))
        visited.add((curr_r, curr_c))
        
        if len(path) > (height * width): break

    # Metrics
    path_len = len(path) - 1
    tortuosity = path_len / height
    avg_res = total_resistance / path_len if path_len > 0 else 0

    # Plotting
    px = [n[1] for n in path]
    py = [n[0] for n in path]

    # --- Stormy sky colormap: very narrow grey range so squares blend together ---
    stormy_colors = ['#282828', '#323232', '#3c3c3c', '#464646', '#505050', '#5a5a5a']
    stormy_cmap = LinearSegmentedColormap.from_list('stormy', stormy_colors)

    # Background heatmap with stormy palette
    im = ax.imshow(bg_weights, cmap=stormy_cmap, alpha=0.75, origin='upper')

    # --- Lightning glow layers (outermost → innermost) ---
    ax.plot(px, py, color='#ffffff', linewidth=7,  alpha=0.04)   # faint white aura
    ax.plot(px, py, color='#ffe066', linewidth=5,  alpha=0.12)   # soft yellow halo
    ax.plot(px, py, color='#ffd700', linewidth=3,  alpha=0.35)   # golden glow
    ax.plot(px, py, color='#ffec00', linewidth=1.8, alpha=0.85)  # bright yellow core
    ax.plot(px, py, color='#ffffff', linewidth=0.6, alpha=0.9)   # white-hot centre

    # Mark start and end
    ax.plot(px[0],  py[0],  'o', color='#ffffff', markersize=6, zorder=5)
    ax.plot(px[-1], py[-1], 'v', color='#ffd700', markersize=8, zorder=5)

    # Statistics box — styled for the dark theme
    stats_text = (
        f"--- Greedy Path Stats ---\n"
        f"Total Resistance: {total_resistance}\n"
        f"Path Length:      {path_len} steps\n"
        f"Tortuosity:       {tortuosity:.3f}\n"
        f"Avg Res/Step:     {avg_res:.2f}\n"
        f"Start Column:     {path[0][1]}"
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

    plt.title(f"8-Neighbor Greedy Lightning Simulation\n{height}×{width} Grid",
              fontsize=14, color='#cccccc')
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