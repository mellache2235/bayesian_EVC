"""
Step 5: Compare Models

PURPOSE:
--------
This step compares Traditional EVC (Step 3) vs Bayesian EVC (Step 4) performance.
This is where you determine if your Bayesian EVC hypothesis is supported.

🎯 THIS IS THE CRITICAL STEP FOR YOUR PHD PROJECT
--------------------------------------------------
This comparison tests your core hypothesis:
"Do people invest cognitive control to reduce uncertainty, beyond just getting rewards?"

⚠️ WHAT THE COMPARISON MEANS:
------------------------------
Traditional EVC Performance:
    - Based on: EVC = reward - effort
    - Assumes: Uncertainty doesn't matter
    - If this fits well → People only care about reward/effort

Bayesian EVC Performance:
    - Based on: EVC = reward - effort + uncertainty_reduction
    - Assumes: Uncertainty reduction IS valued
    - If this fits better → People DO value uncertainty reduction ← SUPPORTS HYPOTHESIS

📊 INTERPRETATION:
------------------
If Bayesian R² > Traditional R²:
    → Uncertainty reduction improves predictions
    → Your hypothesis is SUPPORTED ✓
    → People invest control to reduce uncertainty

If Traditional R² ≥ Bayesian R²:
    → Uncertainty doesn't improve predictions
    → Hypothesis not supported
    → May need to refine uncertainty components

WHAT THIS STEP DOES:
-------------------
1. Loads results from both models (Steps 3 and 4)
2. Compares performance metrics (R², RMSE, correlation)
3. Calculates percentage improvements
4. Determines which model performs better
5. Saves comparison results

WHY THIS STEP MATTERS:
--------------------
- This is the MAIN RESULT for your PhD project
- If Bayesian EVC > Traditional EVC → Your hypothesis is supported!
- Shows whether uncertainty reduction improves predictions
- Provides quantitative evidence for your thesis

KEY CONCEPTS:
------------
1. Model Comparison Metrics:
   - R²: Proportion of variance explained (higher is better)
     Bayesian R² > Traditional R² → Better fit
   - RMSE: Root mean squared error (lower is better)
     Bayesian RMSE < Traditional RMSE → More accurate predictions
   - Correlation: Pearson correlation (higher is better)
     Shows how well predictions track observed control

2. Percentage Improvement:
   - R² improvement: (Bayesian R² - Traditional R²) / Traditional R² × 100%
   - RMSE improvement: (Traditional RMSE - Bayesian RMSE) / Traditional RMSE × 100%
   - Positive improvement → Bayesian is better

3. Hypothesis Testing:
   - H₀: Uncertainty doesn't matter (Traditional = Bayesian)
   - H₁: Uncertainty matters (Bayesian > Traditional)
   - If Bayesian wins on ≥2/3 metrics → Support for H₁

OUTPUT FILES:
------------
- results/model_comparison.csv: Comparison table with metrics for both models
"""

import pandas as pd
import numpy as np


def main():
    """
    Main function to compare Traditional vs Bayesian EVC models.
    
    WORKFLOW:
    ---------
    1. Load results from both models
    2. Extract test set metrics
    3. Compare each metric
    4. Calculate percentage improvements
    5. Determine overall winner
    6. Save comparison results
    """
    print("=" * 70)
    print("STEP 5: COMPARE MODELS")
    print("=" * 70)
    
    # LOAD RESULTS FROM BOTH MODELS
    # These CSV files were created in Steps 3 and 4
    # Format: metric | train | test
    print("\nLoading model results...")
    try:
        trad_results = pd.read_csv('results/traditional_evc_results.csv')
        bayes_results = pd.read_csv('results/bayesian_evc_results.csv')
        print("✓ Loaded both model results")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please run steps 3 and 4 first.")
        return
    
    # EXTRACT TEST SET METRICS
    # .set_index('metric') makes 'metric' column the index
    # ['test'] gets the 'test' column values
    # Result: Series with metric names as index, test values as values
    # Example: trad_test['r2'] = test R² value for Traditional EVC
    trad_test = trad_results.set_index('metric')['test']
    bayes_test = bayes_results.set_index('metric')['test']
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON - TEST SET PERFORMANCE")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'Traditional':<15} {'Bayesian':<15} {'Δ':<15} {'Better':<10}")
    print("-" * 75)
    
    # COMPARE EACH METRIC
    metrics = ['r2', 'rmse', 'correlation']
    improvements = {}
    
    for metric in metrics:
        # Get values for both models
        trad_val = trad_test[metric]   # Traditional EVC value
        bayes_val = bayes_test[metric] # Bayesian EVC value
        
        # CALCULATE DELTA (difference)
        # For RMSE: lower is better, so delta = trad - bayes (positive = Bayesian better)
        # For R² and correlation: higher is better, so delta = bayes - trad (positive = Bayesian better)
        if metric == 'rmse':
            # RMSE: Lower is better
            delta = trad_val - bayes_val  # Positive delta = Bayesian RMSE is lower = Bayesian better
            better = "Bayesian ✓" if delta > 0 else "Traditional"
        else:
            # R² and correlation: Higher is better
            delta = bayes_val - trad_val  # Positive delta = Bayesian is higher = Bayesian better
            better = "Bayesian ✓" if delta > 0 else "Traditional"
        
        improvements[metric] = delta
        
        # Display comparison
        print(f"{metric.upper():<20} {trad_val:<15.4f} {bayes_val:<15.4f} "
              f"{delta:+<15.4f} {better:<10}")
        # {delta:+} means: show + sign for positive, - sign for negative
    
    print("-" * 75)
    
    # CALCULATE PERCENTAGE IMPROVEMENTS
    # Shows relative improvement of Bayesian over Traditional
    print("\n" + "=" * 70)
    print("PERCENTAGE IMPROVEMENTS (Bayesian vs Traditional)")
    print("=" * 70)
    
    # R² improvement: (Bayesian - Traditional) / |Traditional| × 100%
    # abs() in denominator handles negative R² values
    r2_improvement = (bayes_test['r2'] - trad_test['r2']) / abs(trad_test['r2']) * 100
    
    # RMSE improvement: (Traditional - Bayesian) / Traditional × 100%
    # Positive = Bayesian RMSE is lower = improvement
    rmse_improvement = (trad_test['rmse'] - bayes_test['rmse']) / trad_test['rmse'] * 100
    
    # Correlation improvement: (Bayesian - Traditional) / |Traditional| × 100%
    corr_improvement = (bayes_test['correlation'] - trad_test['correlation']) / abs(trad_test['correlation']) * 100
    
    print(f"\nR² improvement: {r2_improvement:+.2f}%")
    print(f"  Interpretation: Bayesian explains {abs(r2_improvement):.2f}% {'more' if r2_improvement > 0 else 'less'} variance")
    print(f"\nRMSE improvement: {rmse_improvement:+.2f}%")
    print(f"  Interpretation: Bayesian has {abs(rmse_improvement):.2f}% {'lower' if rmse_improvement > 0 else 'higher'} error")
    print(f"\nCorrelation improvement: {corr_improvement:+.2f}%")
    print(f"  Interpretation: Bayesian predictions track observed {abs(corr_improvement):.2f}% {'better' if corr_improvement > 0 else 'worse'}")
    
    # OVERALL ASSESSMENT
    # Count how many metrics Bayesian wins on
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    
    # Count wins: Bayesian wins if it's better on each metric
    wins = sum([
        bayes_test['r2'] > trad_test['r2'],           # Bayesian R² higher?
        bayes_test['rmse'] < trad_test['rmse'],       # Bayesian RMSE lower?
        bayes_test['correlation'] > trad_test['correlation']  # Bayesian correlation higher?
    ])
    
    print(f"\nBayesian EVC wins on {wins}/3 metrics")
    
    # INTERPRETATION
    if wins >= 2:
        print("\n✓ Bayesian EVC shows SUPERIOR performance!")
        print("  The uncertainty component improves model predictions.")
        print("  → Your hypothesis is SUPPORTED!")
        print("  → People DO value uncertainty reduction")
    elif wins == 1:
        print("\n≈ Models show MIXED performance.")
        print("  Consider examining specific conditions where each excels.")
        print("  → Partial support for hypothesis")
    else:
        print("\n✗ Traditional EVC shows better performance.")
        print("  The uncertainty component may need refinement.")
        print("  → Hypothesis not supported (may need to adjust model)")
    
    # SAVE COMPARISON RESULTS
    # Save to CSV for easy analysis/reporting
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
    print("  (Create visualizations of results)")


if __name__ == '__main__':
    main()
