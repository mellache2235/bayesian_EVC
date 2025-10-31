"""
Step 4: Fit Bayesian EVC Model

PURPOSE:
--------
This step fits the Bayesian EVC model, which extends Traditional EVC by including
uncertainty reduction as a benefit. This is the KEY MODEL for your PhD project.

⚠️ KEY DIFFERENCE FROM STEP 3 (Traditional EVC):
-------------------------------------------------
Traditional EVC (Step 3):
    EVC = reward_benefit - effort_cost
    Assumes: People only care about reward and effort
    NO uncertainty component

Bayesian EVC (THIS STEP):
    EVC = reward_benefit - effort_cost + uncertainty_reduction_benefit
    Assumes: People ALSO value reducing uncertainty
    INCLUDES uncertainty component ← KEY INNOVATION

🎯 THE HYPOTHESIS YOU'RE TESTING:
----------------------------------
If Bayesian EVC fits better than Traditional EVC:
→ People DO value uncertainty reduction
→ Your hypothesis is SUPPORTED
→ Uncertainty reduction is a meaningful driver of control allocation

WHAT THIS STEP DOES:
-------------------
1. Loads behavioral data from Step 1
2. Uses same train/test split as Step 3 (for fair comparison)
3. Initializes Bayesian EVC model with uncertainty components
4. Fits model to training data (optimizes parameters including uncertainty_weight)
5. Evaluates fitted model on test set
6. Saves model, results, and predictions

WHY THIS STEP MATTERS:
--------------------
- This is the MAIN MODEL you're testing in your PhD project
- Bayesian EVC: EVC = β_r × E[Reward|c] - c_e × c^α + λ × Uncertainty_Reduction(c)
- The uncertainty_weight (λ) parameter is KEY: How much does uncertainty reduction matter?
- Comparison with Step 3 will show if uncertainty improves predictions

KEY CONCEPTS:
------------
1. Bayesian EVC Formula:
   EVC(c) = reward_benefit - effort_cost + uncertainty_reduction_benefit
   
   Uncertainty benefit = λ × η × c × U_total × (1 - τ)
   Where:
   - λ (uncertainty_weight): How much uncertainty reduction matters ← KEY PARAMETER
   - η (control_efficiency): How well control reduces uncertainty
   - c: Control level
   - U_total: Total uncertainty (decision + state)
   - τ (uncertainty_tolerance): How comfortable with uncertainty

2. Model Fitting:
   - Optimizes: reward_weight, effort_cost_weight, uncertainty_weight
   - uncertainty_weight is the KEY parameter we want to estimate
   - Higher uncertainty_weight → people value uncertainty reduction more

3. Comparison with Traditional EVC:
   - If Bayesian R² > Traditional R² → uncertainty improves predictions
   - uncertainty_weight > 0 → uncertainty reduction is valued
   - This supports your hypothesis that people care about uncertainty

OUTPUT FILES:
------------
- results/bayesian_evc_model.pkl: Saved fitted model
- results/bayesian_evc_results.csv: Performance metrics
- results/bayesian_evc_predictions.csv: Test data with predictions
"""

import pandas as pd
import numpy as np
import pickle
from models.bayesian_evc import BayesianEVC


def main():
    """
    Main function to fit Bayesian EVC model.
    
    WORKFLOW:
    ---------
    1. Load behavioral data
    2. Split into train/test (same split as Step 3)
    3. Initialize Bayesian EVC model
    4. Fit model (optimize parameters including uncertainty_weight)
    5. Evaluate on test set
    6. Save model, results, and predictions
    """
    print("=" * 70)
    print("STEP 4: FIT BAYESIAN EVC MODEL")
    print("=" * 70)
    
    # LOAD BEHAVIORAL DATA
    # Same data as Step 3, but now using uncertainty columns
    print("\nLoading data...")
    try:
        data = pd.read_csv('data/behavioral_data.csv')
        print(f"✓ Loaded {len(data)} trials")
    except FileNotFoundError:
        print("✗ Error: data/behavioral_data.csv not found!")
        print("  Please run 'python3 step1_generate_data.py' first.")
        return
    
    # SPLIT DATA (SAME AS STEP 3)
    # CRITICAL: Use same split for fair comparison
    # Same subjects in train/test → models trained/tested on same data
    # This allows direct comparison of performance metrics
    print("\nSplitting data into train/test sets...")
    subjects = data['subject_id'].unique()
    np.random.seed(42)  # Same seed = same shuffle = same split as Step 3
    np.random.shuffle(subjects)
    
    n_train = int(len(subjects) * 0.7)
    train_subjects = subjects[:n_train]
    test_subjects = subjects[n_train:]
    
    train_data = data[data['subject_id'].isin(train_subjects)].copy()
    test_data = data[data['subject_id'].isin(test_subjects)].copy()
    
    print(f"  Training: {len(train_subjects)} subjects, {len(train_data)} trials")
    print(f"  Test: {len(test_subjects)} subjects, {len(test_data)} trials")
    print("  (Same split as Step 3 for fair comparison)")
    
    # INITIALIZE BAYESIAN EVC MODEL
    # Bayesian EVC extends Traditional EVC with uncertainty reduction benefit
    # Formula: EVC = reward - effort + uncertainty_reduction
    #
    # Parameters:
    # - reward_weight=1.0: β_r - Weight for reward benefits (same as Traditional)
    # - effort_cost_weight=1.0: c_e - Weight for effort costs (same as Traditional)
    # - uncertainty_weight=0.5: λ ← KEY PARAMETER
    #   How much does uncertainty reduction matter?
    #   Higher = uncertainty reduction is more valuable
    #   This is what we're trying to estimate!
    # - effort_exponent=2.0: α - Effort cost exponent (quadratic)
    # - n_states=2: Number of task states for uncertainty estimation
    # - learning_rate=0.1: Rate of belief updating
    # - uncertainty_tolerance=0.5: τ - Individual tolerance for uncertainty
    # - control_efficiency=1.0: η - How well control reduces uncertainty
    #
    # Note: fit() will optimize these parameters (especially uncertainty_weight)
    print("\nInitializing Bayesian EVC model...")
    model = BayesianEVC(
        reward_weight=1.0,        # β_r: Reward sensitivity
        effort_cost_weight=1.0,   # c_e: Effort cost scaling
        uncertainty_weight=0.5,   # λ ← KEY: How much uncertainty reduction matters
        effort_exponent=2.0,      # α: Effort cost exponent
        baseline=0.5,             # Baseline control level
        n_states=2,               # Number of task states
        learning_rate=0.1         # Rate of belief updating
    )
    
    # FIT MODEL TO TRAINING DATA
    # This optimizes parameters including uncertainty_weight (λ)
    #
    # What does fit() do?
    # 1. For each trial, computes predicted control from Bayesian EVC
    # 2. Compares predicted vs observed control
    # 3. Optimizes: reward_weight, effort_cost_weight, uncertainty_weight
    # 4. The uncertainty_weight parameter is KEY result!
    #
    # Parameters:
    # - train_data: Training data
    # - observed_control_col: Column with observed control signals
    # - reward_col: Column with reward values
    # - accuracy_col: Column with accuracy/probability
    # - uncertainty_col: Column with total uncertainty ← NEW!
    # - confidence_col: Column with confidence levels ← NEW!
    #
    # Returns: Dictionary with fitted parameters and metrics
    #   KEY: uncertainty_weight shows if uncertainty reduction matters
    print("\nFitting model to training data...")
    print("  (This may take a minute...)")
    print("  (Optimizing parameters including uncertainty_weight...)")
    
    train_results = model.fit(
        train_data,
        observed_control_col='control_signal',    # What we're predicting
        reward_col='reward_magnitude',            # Reward values
        accuracy_col='evidence_clarity',          # Accuracy/probability
        uncertainty_col='total_uncertainty',      # ← NEW: Total uncertainty
        confidence_col='confidence'               # ← NEW: Confidence levels
    )
    
    # DISPLAY TRAINING RESULTS
    # Pay special attention to uncertainty_weight!
    print("\n" + "-" * 70)
    print("TRAINING RESULTS")
    print("-" * 70)
    print(f"Fitted parameters:")
    print(f"  - Baseline: {train_results['baseline']:.4f}")
    print(f"  - Reward weight: {train_results['reward_weight']:.4f}")
    print(f"  - Effort cost weight: {train_results['effort_cost_weight']:.4f}")
    print(f"  - Uncertainty weight: {train_results['uncertainty_weight']:.4f} ← KEY PARAMETER!")
    print(f"    Interpretation: λ - How much uncertainty reduction matters")
    print(f"    If > 0: People value reducing uncertainty")
    print(f"    If ≈ 0: Uncertainty doesn't matter (Traditional EVC is sufficient)")
    print(f"  - Effort exponent: {train_results['effort_exponent']:.4f}")
    print(f"\nTraining performance:")
    print(f"  - R²: {train_results['r2']:.4f}")
    print(f"  - RMSE: {train_results['rmse']:.4f}")
    
    # EVALUATE ON TEST SET
    # Test set shows generalization performance
    print("\nEvaluating on test set...")
    test_results = model.evaluate(
        test_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity',
        uncertainty_col='total_uncertainty',  # ← NEW: Needs uncertainty
        confidence_col='confidence'           # ← NEW: Needs confidence
    )
    
    print("\n" + "-" * 70)
    print("TEST RESULTS")
    print("-" * 70)
    print(f"Test performance:")
    print(f"  - R²: {test_results['r2']:.4f}")
    print(f"    Compare to Traditional EVC R² from Step 3")
    print(f"  - RMSE: {test_results['rmse']:.4f}")
    print(f"    Lower is better (compare to Traditional EVC)")
    print(f"  - Correlation: {test_results['correlation']:.4f}")
    
    # SAVE MODEL AND RESULTS
    print("\nSaving model and results...")
    
    import os
    os.makedirs('results', exist_ok=True)
    
    # SAVE MODEL
    with open('results/bayesian_evc_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # SAVE RESULTS
    results_df = pd.DataFrame({
        'metric': ['r2', 'rmse', 'correlation'],
        'train': [train_results['r2'], train_results['rmse'], np.nan],
        'test': [test_results['r2'], test_results['rmse'], test_results['correlation']]
    })
    results_df.to_csv('results/bayesian_evc_results.csv', index=False)
    
    # SAVE PREDICTIONS
    # predict_control() uses fitted model including uncertainty components
    test_data['bayesian_pred'] = model.predict_control(
        test_data,
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity',
        uncertainty_col='total_uncertainty',  # ← Uses uncertainty
        confidence_col='confidence'
    )
    test_data.to_csv('results/bayesian_evc_predictions.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✓ BAYESIAN EVC FITTING COMPLETE!")
    print("=" * 70)
    print("\nSaved files:")
    print("  - results/bayesian_evc_model.pkl")
    print("  - results/bayesian_evc_results.csv")
    print("  - results/bayesian_evc_predictions.csv")
    
    print("\nKEY RESULT:")
    print(f"  Uncertainty weight (λ) = {train_results['uncertainty_weight']:.4f}")
    if train_results['uncertainty_weight'] > 0.1:
        print("  → People DO value uncertainty reduction!")
    else:
        print("  → Uncertainty reduction has minimal impact")
    
    print("\nNext step: Run 'python3 step5_compare_models.py'")
    print("  (Compare Bayesian vs Traditional EVC performance)")


if __name__ == '__main__':
    main()
