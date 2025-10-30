"""
Step 5: Compare Models

Compares Traditional EVC vs Bayesian EVC performance.
"""

import pandas as pd
import numpy as np


def main():
    print("=" * 70)
    print("STEP 5: COMPARE MODELS")
    print("=" * 70)
    
    # Load results
    print("\nLoading model results...")
    try:
        trad_results = pd.read_csv('results/traditional_evc_results.csv')
        bayes_results = pd.read_csv('results/bayesian_evc_results.csv')
        print("✓ Loaded both model results")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please run steps 3 and 4 first.")
        return
    
    # Extract test metrics
    trad_test = trad_results.set_index('metric')['test']
    bayes_test = bayes_results.set_index('metric')['test']
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON - TEST SET PERFORMANCE")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Traditional':<15} {'Bayesian':<15} {'Δ':<15} {'Better':<10}")
    print("-" * 75)
    
    metrics = ['r2', 'rmse', 'correlation']
    improvements = {}
    
    for metric in metrics:
        trad_val = trad_test[metric]
        bayes_val = bayes_test[metric]
        
        # For RMSE, lower is better
        if metric == 'rmse':
            delta = trad_val - bayes_val  # Positive delta means Bayesian is better
            better = "Bayesian ✓" if delta > 0 else "Traditional"
        else:
            delta = bayes_val - trad_val
            better = "Bayesian ✓" if delta > 0 else "Traditional"
        
        improvements[metric] = delta
        
        print(f"{metric.upper():<20} {trad_val:<15.4f} {bayes_val:<15.4f} "
              f"{delta:+<15.4f} {better:<10}")
    
    print("-" * 75)
    
    # Calculate percentage improvements
    print("\n" + "=" * 70)
    print("PERCENTAGE IMPROVEMENTS (Bayesian vs Traditional)")
    print("=" * 70)
    
    r2_improvement = (bayes_test['r2'] - trad_test['r2']) / abs(trad_test['r2']) * 100
    rmse_improvement = (trad_test['rmse'] - bayes_test['rmse']) / trad_test['rmse'] * 100
    corr_improvement = (bayes_test['correlation'] - trad_test['correlation']) / abs(trad_test['correlation']) * 100
    
    print(f"\nR² improvement: {r2_improvement:+.2f}%")
    print(f"RMSE improvement: {rmse_improvement:+.2f}%")
    print(f"Correlation improvement: {corr_improvement:+.2f}%")
    
    # Overall assessment
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    
    wins = sum([
        bayes_test['r2'] > trad_test['r2'],
        bayes_test['rmse'] < trad_test['rmse'],
        bayes_test['correlation'] > trad_test['correlation']
    ])
    
    print(f"\nBayesian EVC wins on {wins}/3 metrics")
    
    if wins >= 2:
        print("\n✓ Bayesian EVC shows SUPERIOR performance!")
        print("  The uncertainty component improves model predictions.")
    elif wins == 1:
        print("\n≈ Models show MIXED performance.")
        print("  Consider examining specific conditions where each excels.")
    else:
        print("\n✗ Traditional EVC shows better performance.")
        print("  The uncertainty component may need refinement.")
    
    # Save comparison
    print("\nSaving comparison results...")
    comparison_df = pd.DataFrame({
        'Model': ['Traditional EVC', 'Bayesian EVC'],
        'R²': [trad_test['r2'], bayes_test['r2']],
        'RMSE': [trad_test['rmse'], bayes_test['rmse']],
        'Correlation': [trad_test['correlation'], bayes_test['correlation']]
    })
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✓ MODEL COMPARISON COMPLETE!")
    print("=" * 70)
    print("\nSaved to: results/model_comparison.csv")
    print("\nNext step: Run 'python3 step6_visualize.py'")


if __name__ == '__main__':
    main()

