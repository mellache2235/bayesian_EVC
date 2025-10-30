"""
Step 3: Fit Traditional EVC Model

Fits the baseline EVC model without uncertainty.
"""

import pandas as pd
import numpy as np
import pickle
from models.traditional_evc import TraditionalEVC


def main():
    print("=" * 70)
    print("STEP 3: FIT TRADITIONAL EVC MODEL")
    print("=" * 70)
    
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
    
    print(f"  Training: {len(train_subjects)} subjects, {len(train_data)} trials")
    print(f"  Test: {len(test_subjects)} subjects, {len(test_data)} trials")
    
    # Initialize model
    print("\nInitializing Traditional EVC model...")
    model = TraditionalEVC(
        reward_weight=1.0,
        effort_cost_weight=1.0,
        effort_exponent=2.0
    )
    
    # Fit model
    print("\nFitting model to training data...")
    print("  (This may take a minute...)")
    
    train_results = model.fit(
        train_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity'
    )
    
    print("\n" + "-" * 70)
    print("TRAINING RESULTS")
    print("-" * 70)
    print(f"Fitted parameters:")
    print(f"  - Baseline: {train_results['baseline']:.4f}")
    print(f"  - Reward weight: {train_results['reward_weight']:.4f}")
    print(f"  - Effort cost weight: {train_results['effort_cost_weight']:.4f}")
    print(f"  - Effort exponent: {train_results['effort_exponent']:.4f}")
    print(f"\nTraining performance:")
    print(f"  - R²: {train_results['r2']:.4f}")
    print(f"  - RMSE: {train_results['rmse']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(
        test_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity'
    )
    
    print("\n" + "-" * 70)
    print("TEST RESULTS")
    print("-" * 70)
    print(f"Test performance:")
    print(f"  - R²: {test_results['r2']:.4f}")
    print(f"  - RMSE: {test_results['rmse']:.4f}")
    print(f"  - Correlation: {test_results['correlation']:.4f}")
    
    # Save model and results
    print("\nSaving model and results...")
    
    # Save model
    with open('results/traditional_evc_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save results
    results_df = pd.DataFrame({
        'metric': ['r2', 'rmse', 'correlation'],
        'train': [train_results['r2'], train_results['rmse'], np.nan],
        'test': [test_results['r2'], test_results['rmse'], test_results['correlation']]
    })
    results_df.to_csv('results/traditional_evc_results.csv', index=False)
    
    # Save predictions
    test_data['traditional_pred'] = model.predict_control(
        test_data,
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity'
    )
    test_data.to_csv('results/traditional_evc_predictions.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✓ TRADITIONAL EVC FITTING COMPLETE!")
    print("=" * 70)
    print("\nSaved files:")
    print("  - results/traditional_evc_model.pkl")
    print("  - results/traditional_evc_results.csv")
    print("  - results/traditional_evc_predictions.csv")
    
    print("\nNext step: Run 'python3 step4_fit_bayesian_evc.py'")


if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)
    main()

