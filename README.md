# Lightning Path Simulation — Code Overview

Bachelor thesis: *Lightning Path Simulation on Random Resistive Grids*
Author: Jakub Bujak - Maastricht University, DACS

---

## Requirements

```
python 3.9+
networkx
numpy
pandas
matplotlib
scikit-learn
scipy
```

Install with:
```
pip install networkx numpy pandas matplotlib scikit-learn scipy
```

---

## Getting Started

All analysis scripts read from `data100.csv`. Before running any script, update the data path at the top of each file to match your local setup:

```python
df = pd.read_csv("/Users/jakubb/Desktop/Thesis/FINAL/data100.csv")
```

Replace the path with wherever `data100.csv` is located on your machine. The visualisation scripts (`main-dijkstra.py`, `main-greedy.py`) do not read from CSV — they take interactive input at runtime.

---

## Folder Structure

```
THESIS/
└── FINAL/
    ├── RQ1/
    │   ├── RQ1.py
    │   ├── threshold_finder.py
    │   ├── rq1_scatter_overview.png
    │   ├── rq1_phase_transition.png
    │   ├── rq1_fixed_mean_variance.png
    │   └── threshold_sweep_output.png
    ├── RQ2/
    │   ├── RQ2.py
    │   ├── rq2_formula_comparison.png
    │   ├── rq2_predicted_vs_actual.png
    │   ├── rq2_residuals.png
    │   ├── rq2_regime_split_comparison.png
    │   ├── rq2_regime_split_predicted.png
    │   ├── rq2_regime_split_scatter.png
    │   ├── rq2_formulas_resistance.png
    │   ├── rq2_formulas_tortuosity.png
    │   ├── rq2_surfaces_resistance.png
    │   ├── rq2_surfaces_tortuosity.png
    │   ├── rq2_regime_surface_resistance.png
    │   └── rq2_regime_surface_tortuosity.png
    ├── RQ3/
    │   ├── RQ3.py
    │   ├── rq3_comparison.png
    │   ├── rq3_efficiency.png
    │   ├── rq3_formula_transfer.png
    │   ├── rq3_gap_by_family.png
    │   └── rq3_results.csv
    ├── Simulation/
    │   ├── big_simulation.py
    │   ├── main-dijkstra.py
    │   ├── main-greedy.py
    │   └── lightning_path_thesis.png
    └── data100.csv
```

---

## Scripts

### `Simulation/big_simulation.py`
Main data collection. Generates all 252 distributions across 7 families,
runs Dijkstra on 200 random grids per distribution, and records all path
metrics. Saves a checkpoint every 20 distributions in case of crash.

Running this from scratch takes approximately 3–4 hours. If `data100.csv`
is already available, skip this step.

**Output:** `data100.csv`

---

### `Simulation/main-dijkstra.py`
Interactive visualisation for Dijkstra. Enter a grid height and 11
probability values (or `u` for uniform) when prompted. The script builds
a random grid, finds the optimal path, and plots it with edges shaded by
resistance.

**Input (interactive):** grid height, probability vector
**Output:** `lightning_path_thesis.png`

---

### `Simulation/main-greedy.py`
Interactive visualisation for the greedy algorithm. Same input format as
`main-dijkstra.py`. Uses the same white-background edge style.

**Input (interactive):** grid height, probability vector
**Output:** displayed plot

---

### `RQ1/RQ1.py`
Research question 1. Reads `data100.csv` and produces three figures
examining how tortuosity and resistance depend on µ and σ².

**Output:**
- `rq1_scatter_overview.png` — tortuosity and resistance vs µ and σ²
- `rq1_phase_transition.png` — bimodal phase transition
- `rq1_fixed_mean_variance.png` — fixed mean family showing variance effect

---

### `RQ1/threshold_finder.py`
Fine-grained bimodal sweep to locate the empirical percolation threshold.
Varies p₀ from 0.45 to 0.56 in steps of 0.005 with 500 runs per point.

**Output:** printed terminal output — save as `threshold_sweep_output.png`

---

### `RQ2/RQ2.py`
Research question 2. Fits four formulas for tortuosity and resistance on
an 80/20 train/test split and evaluates regime-split predictability at µ = 5.
All R² values shown in figures and printed output are test set R².

**Output (12 figures):**
- `rq2_formula_comparison.png` — R² bar chart for all 4 formulas
- `rq2_predicted_vs_actual.png` — predicted vs actual for best formula
- `rq2_residuals.png` — tortuosity residuals by family
- `rq2_regime_split_comparison.png` — global vs hard vs easy tortuosity R²
- `rq2_regime_split_predicted.png` — predicted vs actual per regime
- `rq2_regime_split_scatter.png` — all distributions coloured hard vs easy
- `rq2_formulas_resistance.png` — all 4 formulas for resistance
- `rq2_formulas_tortuosity.png` — all 4 formulas for tortuosity
- `rq2_surfaces_resistance.png` — fitted surfaces F2 F3 F4 for resistance
- `rq2_surfaces_tortuosity.png` — fitted surfaces F2 F3 F4 for tortuosity
- `rq2_regime_surface_resistance.png` — polynomial surface per regime, resistance
- `rq2_regime_surface_tortuosity.png` — polynomial surface per regime, tortuosity

---

### `RQ3/rq3_analysis.py`
Research question 3. Selects 56 distributions (8 per family), runs both
algorithms on identical grids, and records solution quality and runtime.
Change `H`, `W`, and `RUNS` at the top of the file to run on different
grid sizes.

```python
H    = 50   # grid height
W    = 100  # grid width (typically H * 2)
RUNS = 200  # runs per distribution
```

**Output (5 figures + CSV):**
- `rq3_comparison.png` — greedy vs Dijkstra scatter
- `rq3_gap_by_family.png` — optimality gap per family
- `rq3_formula_transfer.png` — polynomial formula on both algorithms
- `rq3_efficiency.png` — efficiency ratio vs µ and σ²
- `rq3_runtime.png` — runtime and speedup by family
- `rq3_results.csv` — full per-distribution results

---

## Acknowledgements

Parts of this codebase were developed with assistance from Claude
(Anthropic), which was used as a coding and debugging aid throughout
the project. All simulation logic, research design, and analytical
decisions were made by the author.