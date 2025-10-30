"""
Step 6: Create Visualizations

Generates all plots for the analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utils.visualization import EVCVisualizer


def main():
    print("=" * 70)
    print("STEP 6: CREATE VISUALIZATIONS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    try:
        behavioral_data = pd.read_csv('data/behavioral_data.csv')
        neural_data = pd.read_csv('data/neural_data.csv')
        trad_predictions = pd.read_csv('results/traditional_evc_predictions.csv')
        bayes_predictions = pd.read_csv('results/bayesian_evc_predictions.csv')
        comparison = pd.read_csv('results/model_comparison.csv')
        print("✓ All data loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please run previous steps first.")
        return
    
    # Initialize visualizer
    visualizer = EVCVisualizer()
    
    # Create figures directory
    import os
    os.makedirs('results/figures', exist_ok=True)
    
    # 1. Model comparison
    print("\n1. Creating model comparison plot...")
    observed = trad_predictions['control_signal'].values
    traditional_pred = trad_predictions['traditional_pred'].values
    bayesian_pred = bayes_predictions['bayesian_pred'].values
    
    visualizer.plot_model_comparison(
        observed,
        traditional_pred,
        bayesian_pred,
        save_path='results/figures/01_model_comparison.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/01_model_comparison.png")
    
    # 2. Uncertainty effects
    print("2. Creating uncertainty effects plot...")
    visualizer.plot_uncertainty_effects(
        behavioral_data,
        uncertainty_col='total_uncertainty',
        control_col='control_signal',
        save_path='results/figures/02_uncertainty_effects.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/02_uncertainty_effects.png")
    
    # 3. Block effects
    print("3. Creating block effects plot...")
    visualizer.plot_block_effects(
        behavioral_data,
        block_col='block',
        metrics=['accuracy', 'reaction_time', 'control_signal'],
        save_path='results/figures/03_block_effects.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/03_block_effects.png")
    
    # 4. Model fit metrics
    print("4. Creating model fit metrics plot...")
    
    # Load results for metrics
    trad_results = pd.read_csv('results/traditional_evc_results.csv')
    bayes_results = pd.read_csv('results/bayesian_evc_results.csv')
    
    trad_test = trad_results.set_index('metric')['test'].to_dict()
    bayes_test = bayes_results.set_index('metric')['test'].to_dict()
    
    metrics_dict = {
        'Traditional EVC': trad_test,
        'Bayesian EVC': bayes_test
    }
    
    visualizer.plot_model_fit_metrics(
        metrics_dict,
        save_path='results/figures/04_model_fit_metrics.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/04_model_fit_metrics.png")
    
    # 5. Neural correlates
    print("5. Creating neural correlates plot...")
    visualizer.plot_neural_correlates(
        behavioral_data,
        neural_data,
        save_path='results/figures/05_neural_correlates.png'
    )
    plt.close('all')
    print("   ✓ Saved: results/figures/05_neural_correlates.png")
    
    # 6. Individual differences
    print("6. Creating individual differences plot...")
    visualizer.plot_individual_differences(
        behavioral_data,
        subject_col='subject_id',
        uncertainty_tolerance_col='uncertainty_tolerance',
        control_col='control_signal',
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
    print("  2. 02_uncertainty_effects.png - Uncertainty-control relationship")
    print("  3. 03_block_effects.png - Behavioral metrics by block")
    print("  4. 04_model_fit_metrics.png - Model comparison bar charts")
    print("  5. 05_neural_correlates.png - Neural-behavioral correlations")
    print("  6. 06_individual_differences.png - Uncertainty tolerance effects")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nAll results are in the 'results/' directory.")
    print("Check 'results/figures/' for visualizations.")


if __name__ == '__main__':
    main()

