"""
Step 6: Create Visualizations

PURPOSE:
--------
This step generates all publication-ready visualizations for your analysis.
These figures help understand and communicate your results.

WHAT THIS STEP DOES:
-------------------
1. Loads all data and results (behavioral, neural, model predictions)
2. Creates 6 publication-ready plots:
   - Model comparison (predicted vs observed)
   - Uncertainty effects (uncertainty-control relationship)
   - Block effects (behavioral metrics across blocks)
   - Model fit metrics (comparison bar charts)
   - Neural correlates (brain-behavior relationships)
   - Individual differences (uncertainty tolerance effects)
3. Saves all figures as PNG files

WHY THIS STEP MATTERS:
--------------------
- Visualizations make results interpretable
- Publication-ready figures for your thesis/paper
- Helps communicate findings to others
- Reveals patterns that numbers alone might miss

KEY PLOTS:
---------
1. Model Comparison: Shows which model fits better (scatter plots)
2. Uncertainty Effects: Shows relationship between uncertainty and control
3. Block Effects: Shows how behavior changes across experimental blocks
4. Model Fit Metrics: Bar charts comparing R², RMSE, correlation
5. Neural Correlates: Shows brain-behavior relationships
6. Individual Differences: Shows how uncertainty tolerance affects control

OUTPUT FILES:
------------
- results/figures/01_model_comparison.png
- results/figures/02_uncertainty_effects.png
- results/figures/03_block_effects.png
- results/figures/04_model_fit_metrics.png
- results/figures/05_neural_correlates.png
- results/figures/06_individual_differences.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from utils.visualization import EVCVisualizer


def main():
    """
    Main function to create all visualizations.
    
    WORKFLOW:
    ---------
    1. Load all data and results
    2. Initialize visualizer
    3. Create each plot sequentially
    4. Save all figures
    """
    print("=" * 70)
    print("STEP 6: CREATE VISUALIZATIONS")
    print("=" * 70)
    
    # LOAD ALL DATA AND RESULTS
    # Need data from Step 1, predictions from Steps 3 & 4, comparison from Step 5
    print("\nLoading data...")
    try:
        behavioral_data = pd.read_csv('data/behavioral_data.csv')      # From Step 1
        neural_data = pd.read_csv('data/neural_data.csv')              # From Step 1
        trad_predictions = pd.read_csv('results/traditional_evc_predictions.csv')  # From Step 3
        bayes_predictions = pd.read_csv('results/bayesian_evc_predictions.csv')    # From Step 4
        comparison = pd.read_csv('results/model_comparison.csv')        # From Step 5
        print("✓ All data loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please run previous steps first.")
        return
    
    # INITIALIZE VISUALIZER
    # EVCVisualizer provides plotting functions for all visualizations
    visualizer = EVCVisualizer()
    
    # CREATE FIGURES DIRECTORY
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    # PLOT 1: MODEL COMPARISON
    # Shows predicted vs observed control for both models
    # Closer to diagonal line = better predictions
    print("\n1. Creating model comparison plot...")
    
    # Extract observed and predicted control signals
    observed = trad_predictions['control_signal'].values           # Actual control (ground truth)
    traditional_pred = trad_predictions['traditional_pred'].values # Traditional EVC predictions
    bayesian_pred = bayes_predictions['bayesian_pred'].values      # Bayesian EVC predictions
    
    visualizer.plot_model_comparison(
        observed,              # x-axis: observed control
        traditional_pred,      # y-axis: Traditional predictions
        bayesian_pred,         # y-axis: Bayesian predictions
        save_path='results/figures/01_model_comparison.png'
    )
    plt.close('all')  # Close figures to free memory
    print("   ✓ Saved: results/figures/01_model_comparison.png")
    
    # PLOT 2: UNCERTAINTY EFFECTS
    # Shows relationship between uncertainty and control allocation
    # Positive relationship → higher uncertainty → more control (if uncertainty intolerant)
    print("2. Creating uncertainty effects plot...")
    visualizer.plot_uncertainty_effects(
        behavioral_data,
        uncertainty_col='total_uncertainty',  # x-axis: uncertainty level
        control_col='control_signal',         # y-axis: control allocation
        save_path='results/figures/02_uncertainty_effects.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/02_uncertainty_effects.png")
    
    # PLOT 3: BLOCK EFFECTS
    # Shows how behavioral metrics change across experimental blocks
    # Blocks have different uncertainty levels → should see differences
    print("3. Creating block effects plot...")
    visualizer.plot_block_effects(
        behavioral_data,
        block_col='block',                    # x-axis: experimental block
        metrics=['accuracy', 'reaction_time', 'control_signal'],  # y-axes: behavioral metrics
        save_path='results/figures/03_block_effects.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/03_block_effects.png")
    
    # PLOT 4: MODEL FIT METRICS
    # Bar charts comparing R², RMSE, correlation for both models
    # Higher bars = better (except RMSE where lower is better)
    print("4. Creating model fit metrics plot...")
    
    # Load results and convert to dictionary format for plotting
    trad_results = pd.read_csv('results/traditional_evc_results.csv')
    bayes_results = pd.read_csv('results/bayesian_evc_results.csv')
    
    # Convert to dictionaries: {'r2': value, 'rmse': value, 'correlation': value}
    trad_test = trad_results.set_index('metric')['test'].to_dict()
    bayes_test = bayes_results.set_index('metric')['test'].to_dict()
    
    metrics_dict = {
        'Traditional EVC': trad_test,   # Traditional model metrics
        'Bayesian EVC': bayes_test       # Bayesian model metrics
    }
    
    visualizer.plot_model_fit_metrics(
        metrics_dict,  # Dictionary with metrics for each model
        save_path='results/figures/04_model_fit_metrics.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/04_model_fit_metrics.png")
    
    # PLOT 5: NEURAL CORRELATES
    # Shows relationships between behavioral measures and neural activity
    # Validates that neural data correlates with behavior as expected
    print("5. Creating neural correlates plot...")
    visualizer.plot_neural_correlates(
        behavioral_data,  # Behavioral measures
        neural_data,      # Neural activity (DLPFC, ACC, striatum)
        save_path='results/figures/05_neural_correlates.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/05_neural_correlates.png")
    
    # PLOT 6: INDIVIDUAL DIFFERENCES
    # Shows how uncertainty tolerance affects control allocation
    # Individual differences: some people more/less tolerant of uncertainty
    print("6. Creating individual differences plot...")
    visualizer.plot_individual_differences(
        behavioral_data,
        subject_col='subject_id',                    # Column identifying subjects
        uncertainty_tolerance_col='uncertainty_tolerance',  # Individual tolerance level
        control_col='control_signal',                # Control allocation
        save_path='results/figures/06_individual_differences.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/06_individual_differences.png")
    
    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nAll figures saved to: results/figures/")
    print("\nGenerated plots:")
    print("  1. 01_model_comparison.png - Predicted vs observed control")
    print("     Shows which model fits better (points closer to diagonal)")
    print("  2. 02_uncertainty_effects.png - Uncertainty-control relationship")
    print("     Shows if higher uncertainty → more control allocation")
    print("  3. 03_block_effects.png - Behavioral metrics by block")
    print("     Shows how behavior changes across experimental blocks")
    print("  4. 04_model_fit_metrics.png - Model comparison bar charts")
    print("     Shows R², RMSE, correlation for both models (side-by-side)")
    print("  5. 05_neural_correlates.png - Neural-behavioral correlations")
    print("     Shows relationships between brain activity and behavior")
    print("  6. 06_individual_differences.png - Uncertainty tolerance effects")
    print("     Shows how individual differences affect control allocation")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nAll results are in the 'results/' directory.")
    print("Check 'results/figures/' for visualizations.")
    print("\nSummary:")
    print("  - Model comparison: results/model_comparison.csv")
    print("  - Model predictions: results/*_predictions.csv")
    print("  - Visualizations: results/figures/*.png")


if __name__ == '__main__':
    main()
