"""
Step 4b: Fit Temporal Bayesian EVC Model (With Trial History via HGF)

This is an ADVANCED version of Step 4 that incorporates trial history through
the Hierarchical Gaussian Filter (HGF).

KEY DIFFERENCES FROM STEP 4:
- Step 4: Each trial is independent
- Step 4b: Trials are connected via HGF recurrent state
- Expected improvement: R² from -0.02 → 0.25-0.40

HOW IT WORKS:
1. HGF maintains recurrent state across trials (memory of past)
2. HGF estimates uncertainty that evolves over time
3. Bayesian EVC uses HGF uncertainty to predict control
4. Control depends on trial history through HGF state

RECURRENT ARCHITECTURE:
Trial t-1 outcome → HGF update → State[t-1]
                                      ↓ (carried forward)
Trial t outcome → HGF update → State[t] → Uncertainty[t] → Control[t]
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from models.bayesian_evc_temporal import BayesianEVC_Temporal

sns.set_style("whitegrid")


def main():
    print("=" * 70)
    print("STEP 4b: FIT TEMPORAL BAYESIAN EVC MODEL (WITH HGF)")
    print("=" * 70)
    print("\nThis model incorporates trial history through HGF recurrent dynamics.")
    print("Expected improvement over non-temporal model: R² +0.2 to +0.4")
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv('data/behavioral_data.csv')
        print(f"✓ Loaded {len(data)} trials")
    except FileNotFoundError:
        print("✗ Error: data/behavioral_data.csv not found!")
        print("  Please run 'python3 step1_generate_data.py' first.")
        return
    
    # Split data by subject (70/30 train/test)
    print("\nSplitting data into train/test sets...")
    subjects = data['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    n_train = int(len(subjects) * 0.7)
    train_subjects = subjects[:n_train]
    test_subjects = subjects[n_train:]
    
    train_data = data[data['subject_id'].isin(train_subjects)].copy()
    test_data = data[data['subject_id'].isin(test_subjects)].copy()
    
    # IMPORTANT: Sort by subject and trial for temporal modeling!
    train_data = train_data.sort_values(['subject_id', 'trial']).reset_index(drop=True)
    test_data = test_data.sort_values(['subject_id', 'trial']).reset_index(drop=True)
    
    print(f"  Training: {len(train_subjects)} subjects, {len(train_data)} trials")
    print(f"  Test: {len(test_subjects)} subjects, {len(test_data)} trials")
    print(f"  (Same split as Step 3 and 4 for fair comparison)")
    
    # ============================================
    # FIT TEMPORAL BAYESIAN EVC MODEL
    # ============================================
    
    print("\n" + "=" * 70)
    print("TEMPORAL MODEL (WITH RECURRENT HGF)")
    print("=" * 70)
    
    print("\nInitializing Temporal Bayesian EVC model...")
    print("  - HGF parameters: κ₂=1.0, ω₂=-4.0, ω₃=-6.0")
    print("  - Recurrent dynamics: State evolves over trials")
    print("  - Uncertainty adapts to trial outcomes")
    print("  - Volatility is learned from data")
    
    model = BayesianEVC_Temporal(
        reward_weight=1.0,
        effort_cost_weight=1.0,
        uncertainty_weight=0.5,
        volatility_weight=0.2,  # NEW: Volatility affects control
        baseline=0.5,
        # HGF parameters
        kappa_2=1.0,
        omega_2=-4.0,
        omega_3=-6.0
    )
    
    print("\nFitting model to training data...")
    print("  (This may take 2-3 minutes due to sequential processing...)")
    
    train_results = model.fit(
        train_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity',
        outcome_col='accuracy',
        subject_col='subject_id'
    )
    
    print("\n" + "-" * 70)
    print("TRAINING RESULTS")
    print("-" * 70)
    print("Fitted parameters:")
    print(f"  - Baseline: {train_results['baseline']:.4f}")
    print(f"  - Reward weight: {train_results['reward_weight']:.4f}")
    print(f"  - Effort cost weight: {train_results['effort_cost_weight']:.4f}")
    print(f"  - Uncertainty weight (λ): {train_results['uncertainty_weight']:.4f} ← KEY!")
    print(f"  - Volatility weight (γ): {train_results['volatility_weight']:.4f} ← NEW!")
    print(f"    Interpretation: Higher volatility → more control needed")
    
    print(f"\nTraining performance:")
    print(f"  - R²: {train_results['r2']:.4f}")
    print(f"  - RMSE: {train_results['rmse']:.4f}")
    print(f"  - Correlation: {train_results['correlation']:.4f}")
    
    # ============================================
    # EVALUATE ON TEST SET
    # ============================================
    
    print("\nEvaluating on test set...")
    test_results = model.evaluate(
        test_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity',
        outcome_col='accuracy',
        subject_col='subject_id'
    )
    
    print("\n" + "-" * 70)
    print("TEST RESULTS")
    print("-" * 70)
    print(f"Test performance:")
    print(f"  - R²: {test_results['r2']:.4f}")
    print(f"  - RMSE: {test_results['rmse']:.4f}")
    print(f"  - Correlation: {test_results['correlation']:.4f}")
    
    # ============================================
    # COMPARISON WITH NON-TEMPORAL MODEL
    # ============================================
    
    print("\n" + "=" * 70)
    print("COMPARISON WITH STEP 4 (NON-TEMPORAL MODEL)")
    print("=" * 70)
    
    try:
        # Load Step 4 results for comparison
        step4_results = pd.read_csv('results/bayesian_evc_results.csv')
        step4_test_r2 = step4_results[step4_results['metric'] == 'r2']['test'].values[0]
        step4_test_rmse = step4_results[step4_results['metric'] == 'rmse']['test'].values[0]
        
        print("\nStep 4 (Non-temporal):")
        print(f"  - Test R²: {step4_test_r2:.4f}")
        print(f"  - Test RMSE: {step4_test_rmse:.4f}")
        
        print("\nStep 4b (Temporal with HGF):")
        print(f"  - Test R²: {test_results['r2']:.4f}")
        print(f"  - Test RMSE: {test_results['rmse']:.4f}")
        
        print("\nImprovement:")
        if step4_test_r2 < 0 and test_results['r2'] < 0:
            # Both negative - compare absolute distance from 0
            r2_improvement = abs(step4_test_r2) - abs(test_results['r2'])
            print(f"  - R² improvement: {r2_improvement:+.4f} (closer to 0 is better)")
        else:
            r2_improvement = test_results['r2'] - step4_test_r2
            print(f"  - R² improvement: {r2_improvement:+.4f}")
        
        rmse_improvement = step4_test_rmse - test_results['rmse']
        print(f"  - RMSE improvement: {rmse_improvement:+.4f} (lower is better)")
        
        if test_results['r2'] > step4_test_r2:
            print("\n✓ Temporal model outperforms non-temporal model!")
            print("  → Trial history matters for control allocation")
        
    except FileNotFoundError:
        print("\n⚠ Could not load Step 4 results for comparison")
        print("  Run 'python3 step4_fit_bayesian_evc.py' first for comparison")
    
    # ============================================
    # VISUALIZE TEMPORAL DYNAMICS
    # ============================================
    
    print("\n" + "=" * 70)
    print("VISUALIZING TEMPORAL DYNAMICS")
    print("=" * 70)
    
    # Get predictions and trajectories for one subject
    subject_id = test_data['subject_id'].iloc[0]
    subject_data = test_data[test_data['subject_id'] == subject_id].copy()
    
    print(f"\nAnalyzing Subject {subject_id} ({len(subject_data)} trials)...")
    
    predictions, uncertainty_traj, volatility_traj = model.predict_control_sequential(
        subject_data,
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity',
        outcome_col='accuracy',
        subject_col='subject_id'
    )
    
    # Create visualization
    import os
    os.makedirs('results/temporal', exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Uncertainty evolution (from HGF recurrent state)
    axes[0].plot(uncertainty_traj, linewidth=2, color='blue', label='HGF State Uncertainty')
    axes[0].set_ylabel('Uncertainty', fontsize=12)
    axes[0].set_title(f'Subject {subject_id}: HGF Uncertainty Evolution (Recurrent Dynamics)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].text(0.02, 0.95, 'Uncertainty adapts over trials based on outcomes', 
                transform=axes[0].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Plot 2: Volatility evolution (from HGF)
    axes[1].plot(volatility_traj, linewidth=2, color='orange', label='HGF Volatility Estimate')
    axes[1].set_ylabel('Volatility', fontsize=12)
    axes[1].set_title('Volatility Tracking (Learns Environmental Change Rate)', 
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].text(0.02, 0.95, 'Volatility increases when environment is changing', 
                transform=axes[1].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    # Plot 3: Control predictions vs. observed
    axes[2].plot(subject_data['control_signal'].values, 
                linewidth=2, alpha=0.7, label='Observed Control', color='green')
    axes[2].plot(predictions, linewidth=2, alpha=0.7, 
                label='Predicted Control (Temporal)', color='red', linestyle='--')
    axes[2].set_xlabel('Trial', fontsize=12)
    axes[2].set_ylabel('Control', fontsize=12)
    axes[2].set_title('Control Allocation: Observed vs. Predicted (with History)', 
                     fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Calculate R² for this subject
    subject_r2 = 1 - np.sum((subject_data['control_signal'].values - predictions)**2) / \
                 np.sum((subject_data['control_signal'].values - 
                        subject_data['control_signal'].mean())**2)
    axes[2].text(0.02, 0.95, f'Subject R² = {subject_r2:.3f}', 
                transform=axes[2].transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/temporal/temporal_dynamics_subject.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: results/temporal/temporal_dynamics_subject.png")
    
    # ============================================
    # SAVE MODEL AND RESULTS
    # ============================================
    
    print("\n" + "-" * 70)
    print("SAVING MODEL AND RESULTS")
    print("-" * 70)
    
    # Save model
    with open('results/temporal/bayesian_evc_temporal_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✓ Saved: results/temporal/bayesian_evc_temporal_model.pkl")
    
    # Save results
    results_df = pd.DataFrame({
        'metric': ['r2', 'rmse', 'correlation'],
        'train': [train_results['r2'], train_results['rmse'], train_results['correlation']],
        'test': [test_results['r2'], test_results['rmse'], test_results['correlation']]
    })
    results_df.to_csv('results/temporal/bayesian_evc_temporal_results.csv', index=False)
    print("✓ Saved: results/temporal/bayesian_evc_temporal_results.csv")
    
    # Save parameters
    params_df = pd.DataFrame({
        'parameter': ['baseline', 'reward_weight', 'effort_cost_weight', 
                     'uncertainty_weight', 'volatility_weight'],
        'value': [
            train_results['baseline'],
            train_results['reward_weight'],
            train_results['effort_cost_weight'],
            train_results['uncertainty_weight'],
            train_results['volatility_weight']
        ]
    })
    params_df.to_csv('results/temporal/bayesian_evc_temporal_parameters.csv', index=False)
    print("✓ Saved: results/temporal/bayesian_evc_temporal_parameters.csv")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    
    print("\n" + "=" * 70)
    print("✓ TEMPORAL BAYESIAN EVC FITTING COMPLETE!")
    print("=" * 70)
    
    print("\nKey Findings:")
    print(f"  1. Uncertainty weight (λ): {train_results['uncertainty_weight']:.4f}")
    print(f"     → People {'' if train_results['uncertainty_weight'] > 0 else 'do NOT '}value uncertainty reduction")
    
    print(f"  2. Volatility weight (γ): {train_results['volatility_weight']:.4f}")
    print(f"     → Environmental change rate {'DOES' if train_results['volatility_weight'] > 0.05 else 'does NOT'} affect control")
    
    print(f"  3. Test R²: {test_results['r2']:.4f}")
    if test_results['r2'] > 0:
        print(f"     → Model explains {test_results['r2']*100:.1f}% of variance ✓")
    else:
        print(f"     → Model still needs improvement (negative R²)")
    
    print("\nSaved files:")
    print("  - results/temporal/bayesian_evc_temporal_model.pkl")
    print("  - results/temporal/bayesian_evc_temporal_results.csv")
    print("  - results/temporal/bayesian_evc_temporal_parameters.csv")
    print("  - results/temporal/temporal_dynamics_subject.png")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE")
    print("=" * 70)
    
    print("\nThe Temporal model incorporates trial history through HGF:")
    print("  - Uncertainty evolves over trials (not independent!)")
    print("  - Recent outcomes affect current uncertainty")
    print("  - Volatility is learned from prediction errors")
    print("  - Control depends on accumulated uncertainty")
    
    print("\nCompared to non-temporal model (Step 4):")
    print("  - Non-temporal: Each trial independent")
    print("  - Temporal: Trials connected via HGF recurrent state")
    print("  - Expected: Temporal should have higher R² (better prediction)")
    
    print("\nNext step: Run 'python3 step5_compare_models.py'")
    print("  (Will include temporal model in comparison)")


if __name__ == '__main__':
    main()

