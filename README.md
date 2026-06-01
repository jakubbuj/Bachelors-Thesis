# Lightning Path Simulation — Code Overview

Bachelor thesis: *Lightning Path Simulation on Random Resistive Grids*
Author: Jakub Bujak — Maastricht University, DACS

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

## Folder Structure

```
THESIS/
└── FINAL/
    ├── RQ1/
    │   ├── RQ1.py
    │   ├── threshold_finder.py
    │   ├── rq1_scatter_overview.png
    │   ├── rq1_phase_transition.png
    │   └── threshold_sweep_output.png
    ├── RQ2/
    │   ├── RQ2.py
    │   ├── rq2_formula_comparison.png
    │   ├── rq2_predicted_vs_actual.png
    │   ├── rq2_residuals.png
    │   ├── regime_split_comparison.png
    │   ├── regime_split_predicted.png
    │   └── regime_split_scatter.png
    ├── RQ3/
    │   ├── rq3_analysis.py
    │   ├── RQ3.py
    │   ├── rq3_comparison.png
    │   ├── rq3_efficiency.png
    │   ├── rq3_formula_transfer.png
    │   └── rq3_gap_by_family.png
    ├── Simulation/
    │   ├── big_simulation.py
    │   ├── main-dijkstra.py
    │   ├── main-greedy.py
    │   └── lightning_path_thesis.png
    └── data100.csv
```

---

## Data

`data100.csv` sits at the root of FINAL/ and is read by all analysis scripts.
Each row is one distribution with path statistics averaged over 200 runs.

The path at the top of each script points to this file:
```python
df = pd.read_csv("/Users/jakubb/Desktop/Thesis/FINAL/data100.csv")
```
Update this path if you move the folder.

---

## Scripts

### `Simulation/big_simulation.py`
Main data collection. Generates all 252 distributions across 7 families,
runs Dijkstra on 200 random grids per distribution, and records all path
metrics. Saves a checkpoint every 20 distributions in case of crash.

**Output:** `data100.csv`

---

### `Simulation/main-dijkstra.py`
Interactive visualisation for Dijkstra. Takes user input for grid size and
distribution probabilities, runs Dijkstra once, and plots the optimal path
with edges shaded by resistance (light = cheap, dark = expensive).

**Input:** grid height, 11 probability values (or `u` for uniform)
**Output:** `lightning_path_thesis.png`

---

### `Simulation/main-greedy.py`
Interactive visualisation for the greedy algorithm. Same input as
`main-dijkstra.py` but uses the gravity-penalised greedy strategy.
Dark background with lightning-style rendering.

**Input:** grid height, 11 probability values (or `u` for uniform)
**Output:** displayed plot

---

### `RQ1/RQ1.py`
Research question 1 — how do path properties depend on the distribution?

**Output:**
- `rq1_scatter_overview.png` — tortuosity and resistance vs mean and variance
- `rq1_phase_transition.png` — bimodal phase transition
- `rq1_fixed_mean_variance.png` — fixed mean family showing variance effect

---

### `RQ1/threshold_finder.py`
Fine-grained bimodal experiment to locate the empirical percolation threshold.
Varies p₀ from 0.45 to 0.56 in steps of 0.005 with 500 runs per point.
Threshold defined as the first p₀ where mean resistance reaches exactly zero.

**Output:** printed terminal output — take a screenshot and save as `threshold_sweep_output.png`

---

### `RQ2/RQ2.py`
Research question 2 — can formulas predict path properties from distributional
parameters? Fits four formulas for tortuosity and resistance on an 80/20
train/test split, and runs the regime split experiment at µ = 5.

**Output:**
- `rq2_formula_comparison.png` — R² bar chart for all 4 formulas
- `rq2_predicted_vs_actual.png` — predicted vs actual for best formula
- `rq2_residuals.png` — tortuosity residuals coloured by family
- `regime_split_comparison.png` — R² bar chart hard vs easy regimes
- `regime_split_predicted.png` — predicted vs actual per regime
- `regime_split_scatter.png` — all distributions coloured hard vs easy

---

### `RQ3/rq3_analysis.py`
Research question 3 — greedy vs Dijkstra. Selects 56 distributions (8 per
family), runs both algorithms on identical grids for 200 runs each, and
computes efficiency ratio, tortuosity gap, and resistance gap.

**Output:**
- `rq3_comparison.png` — greedy vs Dijkstra scatter
- `rq3_gap_by_family.png` — optimality gap per family (boxplot)
- `rq3_formula_transfer.png` — does the formula predict both algorithms?
- `rq3_efficiency.png` — efficiency ratio vs µ and σ²
- `rq3_results.csv` — full per-distribution results

---

## Figures needed for LaTeX compilation

Place all of the following in the same folder as `thesis_draft.tex`:

```
lightning_path_thesis.png        (Simulation/)
rq1_scatter_overview.png         (RQ1/)
rq1_phase_transition.png         (RQ1/)
rq1_fixed_mean_variance.png      (RQ1/)
threshold_sweep_output.png       (RQ1/)
rq2_formula_comparison.png       (RQ2/)
rq2_predicted_vs_actual.png      (RQ2/)
regime_split_predicted.png       (RQ2/)
regime_split_comparison.png      (RQ2/)
rq3_comparison.png               (RQ3/)
rq3_efficiency.png               (RQ3/)
rq3_gap_by_family.png            (RQ3/)
```