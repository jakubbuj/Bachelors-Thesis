import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def comprehensive_thesis_analysis(filename, dist_name="Uniform Distribution"):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return

    # Prepare data (remove 'run' column for math)
    analysis_df = df.drop(columns=['run'])

    # --- PART 1: DEEP STATISTICAL TABLE ---
    print("="*80)
    print(f"   COMPREHENSIVE STATISTICAL PROFILE: {dist_name.upper()}")
    print("="*80)
    
    # Calculating detailed statistics 
    summary = pd.DataFrame({
        'Mean': analysis_df.mean(),
        'Median': analysis_df.median(),
        'Std Dev': analysis_df.std(),
        'Min': analysis_df.min(),
        'Max': analysis_df.max(),
        'Skewness': analysis_df.skew()
    })
    
    print(summary.round(4))

    # --- PART 2: THE "AHA!" INSIGHTS ---
    print("\n" + "="*80)
    print("      RESEARCH INTERPRETATIONS (OBJ i & iii)      ")
    print("="*80)
    
    # Efficiency vs Baseline [cite: 84, 93]
    baseline_chance = 1/11
    eff_gain = (analysis_df['zero_cost_ratio'].mean() / baseline_chance)
    print(f"* Selection Pressure : {eff_gain:.2f}x (Path is {eff_gain:.2f}x more likely to use R=0 than random choice)")
    
    # Geometric Stretch 
    extra_dist = (analysis_df['tortuosity'].mean() - 1) * 100
    print(f"* Tortuosity Impact  : Path is {extra_dist:.1f}% longer than a straight vertical line.")
    
    # Roughness Stability [cite: 95]
    print(f"* Roughness (D) Consistency: Std Dev of {analysis_df['fractal_dimension'].std():.4f} suggests D is a stable predictor.")

    # --- PART 3: VISUALIZATION DASHBOARD ---
    plt.figure(figsize=(20, 12))
    plt.suptitle(f"Lightning Path Analysis Dashboard: {dist_name}", fontsize=18, fontweight='bold')

    # 1. Distribution & Skewness of Fractal Dimension
    plt.subplot(2, 3, 1)
    sns.histplot(df['fractal_dimension'], kde=True, color='teal')
    plt.title("Fractal Dimension (Roughness) [cite: 86, 93]")
    
    # 2. Tortuosity vs Resistance (The Optimization Trade-off)
    plt.subplot(2, 3, 2)
    sns.scatterplot(data=df, x='tortuosity', y='total_resistance', hue='fractal_dimension', palette='viridis')
    plt.title("Tortuosity vs. Resistance [cite: 81, 82]")

    # 3. Correlation Matrix (For formula derivation) [cite: 85, 94]
    plt.subplot(2, 3, 3)
    sns.heatmap(analysis_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")

    # 4. Boxplot for Lateral Deviation (Visualizing Spread/Skew)
    plt.subplot(2, 3, 4)
    sns.boxplot(x=df['max_lateral_deviation'], color='lightblue')
    plt.title("Max Lateral Deviation Spread ")

    # 5. Resistance vs. Zero-Cost Efficiency
    plt.subplot(2, 3, 5)
    sns.regplot(data=df, x='zero_cost_ratio', y='total_resistance', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title("Impact of Zero-Cost Edges on Total Resistance")

    # 6. Jointplot: Path Length vs. Fractal Dimension
    plt.subplot(2, 3, 6)
    sns.kdeplot(data=df, x='path_length', y='fractal_dimension', fill=True, cmap='rocket')
    plt.title("Density: Path Length vs. Roughness")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Run the analysis on your 1000 uniform runs
comprehensive_thesis_analysis("1000uniform_results.csv", dist_name="Obstacles Simulation Baseline")