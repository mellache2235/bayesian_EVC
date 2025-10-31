"""
Step 5: Compare All Three EVC Models

Comprehensive comparison of:
1. Traditional EVC (no uncertainty)
2. Bayesian EVC (with uncertainty, no temporal dynamics)
3. Temporal Bayesian EVC (with uncertainty + trial history via HGF)

This script tests the progressive improvements from adding:
- Uncertainty consideration (Traditional → Bayesian)
- Temporal dynamics (Bayesian → Temporal)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from models.traditional_evc import TraditionalEVC
from models.bayesian_evc import BayesianEVC
from models.bayesian_evc_temporal import BayesianEVC_Temporal
from sklearn.metrics import r2_score, mean_squared_error
import os

sns.set_style("whitegrid")


def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 80)
    print("\nComparing three models:")
    print("  1. Traditional EVC (baseline)")
    print("  2. Bayesian EVC (+ uncertainty)")
    print("  3. Temporal Bayesian EVC (+ uncertainty + trial history)")
    
    # Load data
    print("\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)
    
    try:
        data = pd.read_csv('data/behavioral_data.csv')
        print(f"✓ Loaded {len(data)} trials from {data['subject_id'].nunique()} subjects")
    except FileNotFoundError:
        print("✗ Error: data/behavioral_data.csv not found!")
        print("  Please run 'python3 step1_generate_data.py' first.")
        return
    
    # Train/test split (same as previous steps)
    print("\nSplitting data (70/30 train/test)...")
    subjects = data['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    n_train = int(len(subjects) * 0.7)
    train_subjects = subjects[:n_train]
    test_subjects = subjects[n_train:]
    
    train_data = data[data['subject_id'].isin(train_subjects)].copy()
    test_data = data[data['subject_id'].isin(test_subjects)].copy()
    
    # Sort for temporal model
    train_data = train_data.sort_values(['subject_id', 'trial']).reset_index(drop=True)
    test_data = test_data.sort_values(['subject_id', 'trial']).reset_index(drop=True)
    
    print(f"  Training: {len(train_subjects)} subjects, {len(train_data)} trials")
    print(f"  Test: {len(test_subjects)} subjects, {len(test_data)} trials")
    
    # ============================================
    # MODEL 1: TRADITIONAL EVC
    # ============================================
    
    print("\n" + "=" * 80)
    print("MODEL 1: TRADITIONAL EVC (No Uncertainty)")
    print("=" * 80)
    print("\nFormula: Control = (Reward × Accuracy) / (2 × Cost)")
    print("         → Does NOT consider uncertainty")
    
    print("\nFitting Traditional EVC...")
    model1 = TraditionalEVC()
    
    train_results1 = model1.fit(
        train_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity'
    )
    
    test_results1 = model1.evaluate(
        test_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity'
    )
    
    print("\n" + "-" * 80)
    print("TRADITIONAL EVC RESULTS")
    print("-" * 80)
    print("Parameters:")
    print(f"  Baseline: {train_results1['baseline']:.4f}")
    print(f"  Reward weight: {train_results1['reward_weight']:.4f}")
    print(f"  Effort cost weight: {train_results1['effort_cost_weight']:.4f}")
    
    print(f"\nTraining: R² = {train_results1['r2']:.4f}, RMSE = {train_results1['rmse']:.4f}")
    print(f"Test:     R² = {test_results1['r2']:.4f}, RMSE = {test_results1['rmse']:.4f}")
    
    # ============================================
    # MODEL 2: BAYESIAN EVC (NON-TEMPORAL)
    # ============================================
    
    print("\n" + "=" * 80)
    print("MODEL 2: BAYESIAN EVC (With Uncertainty, No Temporal)")
    print("=" * 80)
    print("\nFormula: Control = (Reward × Accuracy + λ × Uncertainty) / (2 × Cost)")
    print("         → DOES consider uncertainty")
    print("         → Each trial is INDEPENDENT")
    
    print("\nFitting Bayesian EVC...")
    model2 = BayesianEVC()
    
    train_results2 = model2.fit(
        train_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity'
    )
    
    test_results2 = model2.evaluate(
        test_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity'
    )
    
    print("\n" + "-" * 80)
    print("BAYESIAN EVC RESULTS")
    print("-" * 80)
    print("Parameters:")
    print(f"  Baseline: {train_results2['baseline']:.4f}")
    print(f"  Reward weight: {train_results2['reward_weight']:.4f}")
    print(f"  Effort cost weight: {train_results2['effort_cost_weight']:.4f}")
    print(f"  Uncertainty weight (λ): {train_results2['uncertainty_weight']:.4f} ← KEY!")
    
    if train_results2['uncertainty_weight'] > 0.1:
        print(f"     → Uncertainty DOES matter (λ significantly > 0)")
    
    print(f"\nTraining: R² = {train_results2['r2']:.4f}, RMSE = {train_results2['rmse']:.4f}")
    print(f"Test:     R² = {test_results2['r2']:.4f}, RMSE = {test_results2['rmse']:.4f}")
    
    # ============================================
    # MODEL 3: TEMPORAL BAYESIAN EVC
    # ============================================
    
    print("\n" + "=" * 80)
    print("MODEL 3: TEMPORAL BAYESIAN EVC (With Uncertainty + History)")
    print("=" * 80)
    print("\nFormula: Control = (Reward × Accuracy + λ × Uncertainty_HGF + γ × Volatility) / (2 × Cost)")
    print("         → DOES consider uncertainty")
    print("         → Trials are CONNECTED via HGF recurrent state")
    print("         → Uncertainty evolves based on past outcomes")
    
    print("\nFitting Temporal Bayesian EVC...")
    print("  (This may take 2-3 minutes due to sequential processing...)")
    model3 = BayesianEVC_Temporal()
    
    train_results3 = model3.fit(
        train_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity',
        outcome_col='accuracy',
        subject_col='subject_id'
    )
    
    test_results3 = model3.evaluate(
        test_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity',
        outcome_col='accuracy',
        subject_col='subject_id'
    )
    
    print("\n" + "-" * 80)
    print("TEMPORAL BAYESIAN EVC RESULTS")
    print("-" * 80)
    print("Parameters:")
    print(f"  Baseline: {train_results3['baseline']:.4f}")
    print(f"  Reward weight: {train_results3['reward_weight']:.4f}")
    print(f"  Effort cost weight: {train_results3['effort_cost_weight']:.4f}")
    print(f"  Uncertainty weight (λ): {train_results3['uncertainty_weight']:.4f}")
    print(f"  Volatility weight (γ): {train_results3['volatility_weight']:.4f} ← NEW!")
    
    if train_results3['volatility_weight'] > 0.05:
        print(f"     → Environmental volatility DOES affect control")
    
    print(f"\nTraining: R² = {train_results3['r2']:.4f}, RMSE = {train_results3['rmse']:.4f}")
    print(f"Test:     R² = {test_results3['r2']:.4f}, RMSE = {test_results3['rmse']:.4f}")
    
    # ============================================
    # COMPARISON SUMMARY
    # ============================================
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON")
    print("=" * 80)
    
    # Create comparison table
    comparison_data = {
        'Model': ['Traditional EVC', 'Bayesian EVC', 'Temporal Bayesian EVC'],
        'Uncertainty': ['No', 'Yes (λ)', 'Yes (λ)'],
        'Temporal': ['No', 'No', 'Yes (HGF)'],
        'Train R²': [
            train_results1['r2'],
            train_results2['r2'],
            train_results3['r2']
        ],
        'Test R²': [
            test_results1['r2'],
            test_results2['r2'],
            test_results3['r2']
        ],
        'Test RMSE': [
            test_results1['rmse'],
            test_results2['rmse'],
            test_results3['rmse']
        ],
        'Test Correlation': [
            test_results1['correlation'],
            test_results2['correlation'],
            test_results3['correlation']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    print("\n" + "-" * 80)
    print("PERFORMANCE COMPARISON TABLE")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    
    # Compute improvements
    print("\n" + "-" * 80)
    print("PROGRESSIVE IMPROVEMENTS")
    print("-" * 80)
    
    # Traditional → Bayesian
    r2_improvement_1_to_2 = test_results2['r2'] - test_results1['r2']
    rmse_improvement_1_to_2 = test_results1['rmse'] - test_results2['rmse']
    
    print("\n1. Adding Uncertainty (Traditional → Bayesian):")
    print(f"   R² change: {r2_improvement_1_to_2:+.4f}")
    print(f"   RMSE change: {rmse_improvement_1_to_2:+.4f} (negative = better)")
    
    if r2_improvement_1_to_2 > 0.01:
        print("   → Significant improvement! Uncertainty matters!")
    elif r2_improvement_1_to_2 < -0.01:
        print("   → Performance decreased. Uncertainty may not matter or model overfits.")
    else:
        print("   → Minimal change. Uncertainty has small effect.")
    
    # Bayesian → Temporal
    r2_improvement_2_to_3 = test_results3['r2'] - test_results2['r2']
    rmse_improvement_2_to_3 = test_results2['rmse'] - test_results3['rmse']
    
    print("\n2. Adding Trial History (Bayesian → Temporal):")
    print(f"   R² change: {r2_improvement_2_to_3:+.4f}")
    print(f"   RMSE change: {rmse_improvement_2_to_3:+.4f} (negative = better)")
    
    if r2_improvement_2_to_3 > 0.05:
        print("   → Large improvement! Trial history matters a lot!")
    elif r2_improvement_2_to_3 > 0.01:
        print("   → Moderate improvement. Trial history helps.")
    elif r2_improvement_2_to_3 < -0.01:
        print("   → Performance decreased. Temporal effects weak or overfitting.")
    else:
        print("   → Minimal change. Temporal effects are small.")
    
    # Overall improvement
    r2_improvement_total = test_results3['r2'] - test_results1['r2']
    rmse_improvement_total = test_results1['rmse'] - test_results3['rmse']
    
    print("\n3. Overall (Traditional → Temporal):")
    print(f"   R² change: {r2_improvement_total:+.4f}")
    print(f"   RMSE change: {rmse_improvement_total:+.4f}")
    
    if test_results1['r2'] < 0 and test_results3['r2'] > 0:
        print("   → Major breakthrough! Went from negative to positive R²!")
    elif r2_improvement_total > 0.2:
        print("   → Excellent improvement! Full model much better!")
    elif r2_improvement_total > 0.1:
        print("   → Good improvement. Extensions are valuable.")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    os.makedirs('results/comparison', exist_ok=True)
    
    # Plot 1: Model comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    models = ['Traditional\nEVC', 'Bayesian\nEVC', 'Temporal\nBayesian EVC']
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    # R² comparison
    test_r2s = [test_results1['r2'], test_results2['r2'], test_results3['r2']]
    bars1 = axes[0].bar(models, test_r2s, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Test R²', fontsize=12)
    axes[0].set_title('R² Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, test_r2s)):
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # RMSE comparison
    test_rmses = [test_results1['rmse'], test_results2['rmse'], test_results3['rmse']]
    bars2 = axes[1].bar(models, test_rmses, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Test RMSE', fontsize=12)
    axes[1].set_title('RMSE Comparison\n(Lower is Better)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars2, test_rmses)):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Correlation comparison
    test_corrs = [test_results1['correlation'], test_results2['correlation'], test_results3['correlation']]
    bars3 = axes[2].bar(models, test_corrs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[2].set_ylabel('Test Correlation', fontsize=12)
    axes[2].set_title('Correlation Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars3, test_corrs)):
        axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/comparison/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/comparison/model_comparison.png")
    
    # Plot 2: Predictions comparison for one subject
    subject_id = test_data['subject_id'].iloc[0]
    subject_data = test_data[test_data['subject_id'] == subject_id].copy()
    
    # Get predictions from each model
    pred1 = model1.predict_control(subject_data)
    pred2 = model2.predict_control(subject_data)
    pred3, _, _ = model3.predict_control_sequential(subject_data)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    trials = range(len(subject_data))
    ax.plot(trials, subject_data['control_signal'].values, 
            linewidth=3, label='Observed Control', color='black', alpha=0.7)
    ax.plot(trials, pred1, linewidth=2, label='Traditional EVC', 
            color=colors[0], linestyle='--', alpha=0.8)
    ax.plot(trials, pred2, linewidth=2, label='Bayesian EVC', 
            color=colors[1], linestyle='--', alpha=0.8)
    ax.plot(trials, pred3, linewidth=2, label='Temporal Bayesian EVC', 
            color=colors[2], linestyle='--', alpha=0.8)
    
    ax.set_xlabel('Trial', fontsize=12)
    ax.set_ylabel('Control Signal', fontsize=12)
    ax.set_title(f'Subject {subject_id}: Model Predictions Comparison', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Calculate R² for this subject
    r2_trad = 1 - np.sum((subject_data['control_signal'].values - pred1)**2) / \
              np.sum((subject_data['control_signal'].values - subject_data['control_signal'].mean())**2)
    r2_bayes = 1 - np.sum((subject_data['control_signal'].values - pred2)**2) / \
               np.sum((subject_data['control_signal'].values - subject_data['control_signal'].mean())**2)
    r2_temp = 1 - np.sum((subject_data['control_signal'].values - pred3)**2) / \
              np.sum((subject_data['control_signal'].values - subject_data['control_signal'].mean())**2)
    
    textstr = f'Subject R²:\nTraditional: {r2_trad:.3f}\nBayesian: {r2_bayes:.3f}\nTemporal: {r2_temp:.3f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/comparison/predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/comparison/predictions_comparison.png")
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    
    print("\n" + "-" * 80)
    print("SAVING RESULTS")
    print("-" * 80)
    
    # Save comparison table
    comparison_df.to_csv('results/comparison/model_comparison.csv', index=False)
    print("✓ Saved: results/comparison/model_comparison.csv")
    
    # Save detailed results
    detailed_results = {
        'traditional': {
            'train': train_results1,
            'test': test_results1
        },
        'bayesian': {
            'train': train_results2,
            'test': test_results2
        },
        'temporal': {
            'train': train_results3,
            'test': test_results3
        },
        'improvements': {
            'uncertainty_effect': r2_improvement_1_to_2,
            'temporal_effect': r2_improvement_2_to_3,
            'total_improvement': r2_improvement_total
        }
    }
    
    with open('results/comparison/detailed_results.pkl', 'wb') as f:
        pickle.dump(detailed_results, f)
    print("✓ Saved: results/comparison/detailed_results.pkl")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    
    print("\n" + "=" * 80)
    print("✓ COMPREHENSIVE MODEL COMPARISON COMPLETE!")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print("SUMMARY OF FINDINGS")
    print("-" * 80)
    
    print("\n1. Does Uncertainty Matter?")
    print(f"   Uncertainty weight (λ): {train_results2['uncertainty_weight']:.4f}")
    print(f"   Performance improvement: R² {r2_improvement_1_to_2:+.4f}")
    if train_results2['uncertainty_weight'] > 0.1 and r2_improvement_1_to_2 > 0.01:
        print("   → YES! Uncertainty significantly improves predictions ✓")
    elif train_results2['uncertainty_weight'] > 0.1:
        print("   → Uncertainty has positive weight but improvement is modest")
    else:
        print("   → Uncertainty effect is small in this dataset")
    
    print("\n2. Does Trial History Matter?")
    print(f"   Volatility weight (γ): {train_results3['volatility_weight']:.4f}")
    print(f"   Performance improvement: R² {r2_improvement_2_to_3:+.4f}")
    if r2_improvement_2_to_3 > 0.05:
        print("   → YES! Temporal dynamics significantly improve predictions ✓")
    elif r2_improvement_2_to_3 > 0.01:
        print("   → Temporal effects provide modest improvement")
    else:
        print("   → Temporal effects are small or negligible")
    
    print("\n3. Best Model:")
    best_r2 = max(test_results1['r2'], test_results2['r2'], test_results3['r2'])
    if best_r2 == test_results3['r2']:
        print("   → Temporal Bayesian EVC (full model) ✓")
    elif best_r2 == test_results2['r2']:
        print("   → Bayesian EVC (uncertainty but no temporal)")
    else:
        print("   → Traditional EVC (baseline)")
    
    print(f"   Best Test R²: {best_r2:.4f}")
    
    print("\n" + "-" * 80)
    print("SAVED FILES:")
    print("  - results/comparison/model_comparison.csv")
    print("  - results/comparison/model_comparison.png")
    print("  - results/comparison/predictions_comparison.png")
    print("  - results/comparison/detailed_results.pkl")
    print("-" * 80)


if __name__ == '__main__':
    main()

