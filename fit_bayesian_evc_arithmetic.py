"""
Fit Bayesian EVC to Arithmetic Task Data

Analyzes how children allocate cognitive control during math problems.

Research Questions:
1. Does problem difficulty (uncertainty) increase control? (λ > 0?)
2. Does past performance affect current control? (temporal effects?)
3. Do children differ in uncertainty sensitivity? (individual differences in λ?)
4. Do older children allocate control more efficiently?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.bayesian_evc import BayesianEVC
from models.bayesian_evc_temporal import BayesianEVC_Temporal

sns.set_style("whitegrid")


def main():
    print("=" * 70)
    print("FIT BAYESIAN EVC TO ARITHMETIC TASK DATA")
    print("=" * 70)
    
    # Load data
    print("\nLoading arithmetic task data...")
    try:
        data = pd.read_csv('data/arithmetic/arithmetic_task_data.csv')
        print(f"✓ Loaded {len(data)} trials from {data['child_id'].nunique()} children")
    except FileNotFoundError:
        print("✗ Error: data/arithmetic/arithmetic_task_data.csv not found!")
        print("  Please run 'python3 generate_arithmetic_data.py' first.")
        return
    
    # ============================================
    # FIT NON-TEMPORAL MODEL
    # ============================================
    
    print("\n" + "=" * 70)
    print("MODEL 1: NON-TEMPORAL BAYESIAN EVC")
    print("=" * 70)
    
    print("\nFitting non-temporal model...")
    model = BayesianEVC()
    
    results = model.fit(
        data,
        observed_control_col='control_signal',
        reward_col='reward',
        accuracy_col='expected_accuracy'
    )
    
    print("\nFitted parameters:")
    print(f"  Baseline: {results['baseline']:.4f}")
    print(f"  Reward weight: {results['reward_weight']:.4f}")
    print(f"  Uncertainty weight (λ): {results['uncertainty_weight']:.4f}")
    
    if results['uncertainty_weight'] > 0.1:
        print(f"\n→ Children DO increase control when uncertain (λ = {results['uncertainty_weight']:.4f})")
    else:
        print(f"\n→ Uncertainty has minimal effect on control (λ = {results['uncertainty_weight']:.4f})")
    
    print(f"\nModel performance:")
    print(f"  R²: {results['r2']:.4f}")
    print(f"  RMSE: {results['rmse']:.4f}")
    
    # ============================================
    # FIT TEMPORAL MODEL
    # ============================================
    
    print("\n" + "=" * 70)
    print("MODEL 2: TEMPORAL BAYESIAN EVC (WITH HGF)")
    print("=" * 70)
    
    print("\nFitting temporal model with trial history...")
    temporal_model = BayesianEVC_Temporal()
    
    temporal_results = temporal_model.fit(
        data,
        observed_control_col='control_signal',
        reward_col='reward',
        accuracy_col='expected_accuracy',
        outcome_col='correct',
        subject_col='child_id'
    )
    
    print("\nFitted parameters:")
    print(f"  Baseline: {temporal_results['baseline']:.4f}")
    print(f"  Reward weight: {temporal_results['reward_weight']:.4f}")
    print(f"  Uncertainty weight (λ): {temporal_results['uncertainty_weight']:.4f}")
    print(f"  Volatility weight (γ): {temporal_results['volatility_weight']:.4f} ← NEW!")
    
    print(f"\nModel performance:")
    print(f"  R²: {temporal_results['r2']:.4f}")
    print(f"  RMSE: {temporal_results['rmse']:.4f}")
    
    # Compare models
    improvement = temporal_results['r2'] - results['r2']
    print(f"\nImprovement from adding trial history: {improvement:+.4f} R²")
    if improvement > 0.05:
        print("→ Trial history significantly improves predictions!")
    
    # ============================================
    # INDIVIDUAL DIFFERENCES ANALYSIS
    # ============================================
    
    print("\n" + "=" * 70)
    print("INDIVIDUAL DIFFERENCES ANALYSIS")
    print("=" * 70)
    
    print("\nFitting model for each child...")
    
    child_results = []
    for child_id in data['child_id'].unique():
        child_data = data[data['child_id'] == child_id]
        
        child_model = BayesianEVC()
        child_res = child_model.fit(child_data)
        
        child_results.append({
            'child_id': child_id,
            'age': child_data['age'].iloc[0],
            'math_ability': child_data['math_ability'].iloc[0],
            'lambda': child_res['uncertainty_weight'],
            'r2': child_res['r2']
        })
    
    child_df = pd.DataFrame(child_results)
    
    print(f"\nUncertainty weight (λ) across children:")
    print(f"  Mean: {child_df['lambda'].mean():.3f}")
    print(f"  Std: {child_df['lambda'].std():.3f}")
    print(f"  Range: [{child_df['lambda'].min():.3f}, {child_df['lambda'].max():.3f}]")
    
    # Correlation with age
    corr_age = child_df['lambda'].corr(child_df['age'])
    print(f"\nCorrelation(λ, age): r = {corr_age:.3f}")
    if abs(corr_age) > 0.3:
        if corr_age < 0:
            print("→ Older children LESS sensitive to uncertainty (more efficient)")
        else:
            print("→ Older children MORE sensitive to uncertainty (more cautious)")
    
    # Correlation with ability
    corr_ability = child_df['lambda'].corr(child_df['math_ability'])
    print(f"Correlation(λ, ability): r = {corr_ability:.3f}")
    if abs(corr_ability) > 0.3:
        if corr_ability < 0:
            print("→ Higher ability children LESS affected by uncertainty")
        else:
            print("→ Higher ability children MORE affected by uncertainty")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    
    import os
    os.makedirs('results/arithmetic', exist_ok=True)
    
    # Plot 1: Individual differences
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(child_df['age'], child_df['lambda'], alpha=0.6, s=100)
    axes[0].set_xlabel('Age (years)', fontsize=12)
    axes[0].set_ylabel('Uncertainty Weight (λ)', fontsize=12)
    axes[0].set_title(f'Uncertainty Sensitivity by Age\nr = {corr_age:.3f}', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(child_df['age'], child_df['lambda'], 1)
    p = np.poly1d(z)
    ages = np.linspace(child_df['age'].min(), child_df['age'].max(), 100)
    axes[0].plot(ages, p(ages), "r--", alpha=0.8, linewidth=2)
    
    axes[1].scatter(child_df['math_ability'], child_df['lambda'], alpha=0.6, s=100)
    axes[1].set_xlabel('Math Ability', fontsize=12)
    axes[1].set_ylabel('Uncertainty Weight (λ)', fontsize=12)
    axes[1].set_title(f'Uncertainty Sensitivity by Ability\nr = {corr_ability:.3f}', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    z = np.polyfit(child_df['math_ability'], child_df['lambda'], 1)
    p = np.poly1d(z)
    abilities = np.linspace(child_df['math_ability'].min(), child_df['math_ability'].max(), 100)
    axes[1].plot(abilities, p(abilities), "r--", alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('results/arithmetic/individual_differences.png', dpi=300)
    print("✓ Saved: results/arithmetic/individual_differences.png")
    
    # Plot 2: Control by difficulty
    fig, ax = plt.subplots(figsize=(10, 6))
    
    difficulty_control = data.groupby('difficulty')['control_signal'].agg(['mean', 'std', 'count'])
    
    ax.errorbar(difficulty_control.index, difficulty_control['mean'], 
                yerr=difficulty_control['std'] / np.sqrt(difficulty_control['count']),
                marker='o', markersize=10, linewidth=2, capsize=5)
    ax.set_xlabel('Problem Difficulty', fontsize=12)
    ax.set_ylabel('Control Allocation', fontsize=12)
    ax.set_title('Control Increases with Problem Difficulty', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 6))
    
    plt.tight_layout()
    plt.savefig('results/arithmetic/control_by_difficulty.png', dpi=300)
    print("✓ Saved: results/arithmetic/control_by_difficulty.png")
    
    # ============================================
    # SUMMARY
    # ============================================
    
    print("\n" + "=" * 70)
    print("✓ ARITHMETIC TASK ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print("\nKey Findings:")
    print(f"\n1. Uncertainty Weight (λ):")
    print(f"   Population: λ = {results['uncertainty_weight']:.3f}")
    print(f"   Individual range: {child_df['lambda'].min():.3f} to {child_df['lambda'].max():.3f}")
    
    if results['uncertainty_weight'] > 0.1:
        print(f"   → Children DO allocate more control when uncertain")
    
    print(f"\n2. Trial History:")
    print(f"   Improvement from temporal model: {improvement:+.4f} R²")
    if improvement > 0.05:
        print(f"   → Past performance DOES affect current control")
    
    print(f"\n3. Age Effects:")
    print(f"   Correlation(λ, age): r = {corr_age:.3f}")
    if abs(corr_age) > 0.3:
        print(f"   → Uncertainty sensitivity changes with age")
    
    print(f"\n4. Ability Effects:")
    print(f"   Correlation(λ, ability): r = {corr_ability:.3f}")
    if abs(corr_ability) > 0.3:
        print(f"   → Math ability relates to control allocation strategy")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

