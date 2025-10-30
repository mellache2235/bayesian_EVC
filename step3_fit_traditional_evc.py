"""
Step 3: Fit Traditional EVC Model

PURPOSE:
--------
This step fits the BASELINE Traditional EVC model to behavioral data.
This is the control/comparison model - it does NOT include uncertainty components.

⚠️ KEY POINT: This model ASSUMES uncertainty doesn't matter.
Later, Step 4 will fit Bayesian EVC (which includes uncertainty) for comparison.
The difference shows whether uncertainty reduction improves predictions.

WHAT THIS STEP DOES:
-------------------
1. Loads behavioral data from Step 1
2. Splits data into training (70%) and test (30%) sets by subject
3. Initializes Traditional EVC model with default parameters
4. Fits model to training data (optimizes parameters to match observed control)
5. Evaluates fitted model on test data
6. Saves model, results, and predictions

WHY THIS STEP MATTERS:
--------------------
- Establishes baseline performance for comparison
- Traditional EVC: EVC = β_r × E[Reward|c] - c_e × c^α
- This model does NOT include uncertainty reduction as a benefit
- Later, Step 4 will fit Bayesian EVC (WITH uncertainty) for comparison
- The comparison shows whether uncertainty reduction improves predictions

KEY CONCEPTS:
------------
1. Train/Test Split: Split data to evaluate generalization
   - Training set: Used to fit parameters (find best model)
   - Test set: Used to evaluate performance (simulates new data)
   - Split by subject: Prevents data leakage (same subject in train and test)

2. Model Fitting: Find parameters that best explain observed control
   - Observed: actual control_signal from data
   - Predicted: control from EVC model
   - Optimization: Minimize prediction error (mean squared error)
   - Parameters fit: reward_weight (β_r), effort_cost_weight (c_e)

3. Performance Metrics:
   - R²: Proportion of variance explained (higher is better, max=1)
   - RMSE: Root mean squared error (lower is better)
   - Correlation: How well predictions track observed (higher is better)

OUTPUT FILES:
------------
- results/traditional_evc_model.pkl: Saved model (can reload later)
- results/traditional_evc_results.csv: Performance metrics (R², RMSE, correlation)
- results/traditional_evc_predictions.csv: Test data with predictions added
"""

import pandas as pd
import numpy as np
import pickle
from models.traditional_evc import TraditionalEVC


def main():
    """
    Main function to fit Traditional EVC model.
    
    WORKFLOW:
    ---------
    1. Load behavioral data
    2. Split into train/test sets (70/30 by subject)
    3. Initialize model with default parameters
    4. Fit model to training data (optimize parameters)
    5. Evaluate on test set
    6. Save model, results, and predictions
    """
    print("=" * 70)
    print("STEP 3: FIT TRADITIONAL EVC MODEL")
    print("=" * 70)
    
    # LOAD BEHAVIORAL DATA
    # Load the data created in Step 1
    # This data contains: subject_id, trial, reward_magnitude, control_signal, etc.
    print("\nLoading data...")
    try:
        data = pd.read_csv('data/behavioral_data.csv')
        print(f"✓ Loaded {len(data)} trials")
    except FileNotFoundError:
        print("✗ Error: data/behavioral_data.csv not found!")
        print("  Please run 'python3 step1_generate_data.py' first.")
        return
    
    # SPLIT DATA INTO TRAIN/TEST SETS
    # Why split? To evaluate model generalization (how well it works on new data)
    # Why by subject? To prevent data leakage (same subject shouldn't be in both sets)
    #
    # Process:
    # 1. Get all unique subject IDs
    # 2. Randomly shuffle them (with fixed seed for reproducibility)
    # 3. Split 70% for training, 30% for testing
    # 4. Separate trials based on subject IDs
    print("\nSplitting data into train/test sets...")
    
    # Get all unique subject IDs
    # data['subject_id'].unique() returns array of unique subject IDs
    subjects = data['subject_id'].unique()
    
    # Set random seed for reproducibility
    # Same seed = same random shuffle = same train/test split every time
    np.random.seed(42)
    np.random.shuffle(subjects)  # Shuffle subjects randomly
    
    # Calculate number of subjects for training (70%)
    # int() rounds down: e.g., 30 subjects → 21 train, 9 test
    n_train = int(len(subjects) * 0.7)
    
    # Split subjects: first n_train go to training, rest to test
    train_subjects = subjects[:n_train]   # First 70% of subjects
    test_subjects = subjects[n_train:]    # Last 30% of subjects
    
    # Filter data by subject IDs
    # .isin() returns boolean mask: True if subject_id is in train_subjects
    # .copy() creates independent copy (avoids warning about views vs copies
    train_data = data[data['subject_id'].isin(train_subjects)].copy()
    test_data = data[data['subject_id'].isin(test_subjects)].copy()
    
    print(f"  Training: {len(train_subjects)} subjects, {len(train_data)} trials")
    print(f"  Test: {len(test_subjects)} subjects, {len(test_data)} trials")
    
    # INITIALIZE TRADITIONAL EVC MODEL
    # TraditionalEVC implements: EVC = β_r × E[Reward|c] - c_e × c^α
    #
    # Parameters:
    # - reward_weight=1.0: β_r - Weight for reward benefits
    #   Interpretation: "How much does reward matter?"
    #   Higher = reward is more important in control decisions
    #
    # - effort_cost_weight=1.0: c_e - Weight for effort costs
    #   Interpretation: "How costly is effort?"
    #   Higher = effort is more aversive
    #
    # - effort_exponent=2.0: α - Exponent for effort cost function
    #   Interpretation: "How quickly does effort cost increase?"
    #   α=2 means quadratic (doubling control quadruples cost)
    #
    # Note: These are initial values. The fit() method will optimize them.
    print("\nInitializing Traditional EVC model...")
    model = TraditionalEVC(
        reward_weight=1.0,        # β_r: Initial reward sensitivity
        effort_cost_weight=1.0,   # c_e: Initial effort cost scaling
        effort_exponent=2.0      # α: Effort cost exponent (quadratic)
    )
    
    # FIT MODEL TO TRAINING DATA
    # This optimizes model parameters to match observed control allocation
    #
    # What does fit() do?
    # 1. For each trial, computes predicted control from model
    # 2. Compares predicted vs observed control
    # 3. Optimizes parameters to minimize prediction error (MSE)
    # 4. Returns fitted parameters and performance metrics
    #
    # Parameters:
    # - train_data: Training data with observed control signals
    # - observed_control_col='control_signal': Column with observed control (what we're predicting)
    # - reward_col='reward_magnitude': Column with reward values
    # - accuracy_col='evidence_clarity': Column with accuracy/probability values
    #
    # Returns: Dictionary with:
    #   - Fitted parameters: reward_weight, effort_cost_weight, effort_exponent
    #   - Performance metrics: r2, rmse, correlation
    print("\nFitting model to training data...")
    print("  (This may take a minute...)")
    
    train_results = model.fit(
        train_data,                              # Training data
        observed_control_col='control_signal',    # What we're trying to predict
        reward_col='reward_magnitude',            # Reward values
        accuracy_col='evidence_clarity'           # Accuracy/probability values
    )
    
    # DISPLAY TRAINING RESULTS
    print("\n" + "-" * 70)
    print("TRAINING RESULTS")
    print("-" * 70)
    print(f"Fitted parameters:")
    print(f"  - Baseline: {train_results['baseline']:.4f}")
    print(f"  - Reward weight: {train_results['reward_weight']:.4f}")
    print(f"    Interpretation: Fitted value of β_r (reward sensitivity)")
    print(f"  - Effort cost weight: {train_results['effort_cost_weight']:.4f}")
    print(f"    Interpretation: Fitted value of c_e (effort cost scaling)")
    print(f"  - Effort exponent: {train_results['effort_exponent']:.4f}")
    print(f"    Interpretation: α (fixed at 2.0, quadratic cost)")
    print(f"\nTraining performance:")
    print(f"  - R²: {train_results['r2']:.4f}")
    print(f"    Interpretation: Proportion of variance explained (1.0 = perfect)")
    print(f"  - RMSE: {train_results['rmse']:.4f}")
    print(f"    Interpretation: Root mean squared error (lower is better)")
    
    # EVALUATE ON TEST SET
    # Test set evaluation shows how well model generalizes to new data
    # This is the TRUE test of model performance (not training performance)
    #
    # evaluate() uses the fitted parameters to predict test data
    # It does NOT re-fit the model (uses parameters from training)
    print("\nEvaluating on test set...")
    test_results = model.evaluate(
        test_data,                               # Test data (unseen during training)
        observed_control_col='control_signal',      # What we're predicting
        reward_col='reward_magnitude',            # Reward values
        accuracy_col='evidence_clarity'           # Accuracy values
    )
    
    print("\n" + "-" * 70)
    print("TEST RESULTS")
    print("-" * 70)
    print(f"Test performance:")
    print(f"  - R²: {test_results['r2']:.4f}")
    print(f"    Interpretation: Generalization performance")
    print(f"  - RMSE: {test_results['rmse']:.4f}")
    print(f"    Interpretation: Prediction error on new data")
    print(f"  - Correlation: {test_results['correlation']:.4f}")
    print(f"    Interpretation: How well predictions track observed (Pearson r)")
    
    # SAVE MODEL AND RESULTS
    # Save everything so we can:
    # 1. Reload model later without re-fitting
    # 2. Compare results with Bayesian EVC
    # 3. Analyze predictions
    print("\nSaving model and results...")
    
    # CREATE RESULTS DIRECTORY
    import os
    os.makedirs('results', exist_ok=True)
    
    # SAVE MODEL (pickle format)
    # pickle.dump() serializes Python object to file
    # 'wb' = write binary mode
    # This allows reloading the fitted model later without re-fitting
    with open('results/traditional_evc_model.pkl', 'wb') as f:
        pickle.dump(model, f)  # Save model object
    
    # SAVE RESULTS (CSV format)
    # Create DataFrame with performance metrics
    # Columns: metric (r2, rmse, correlation), train value, test value
    results_df = pd.DataFrame({
        'metric': ['r2', 'rmse', 'correlation'],
        'train': [train_results['r2'], train_results['rmse'], np.nan],  # No train correlation
        'test': [test_results['r2'], test_results['rmse'], test_results['correlation']]
    })
    results_df.to_csv('results/traditional_evc_results.csv', index=False)
    
    # SAVE PREDICTIONS
    # Add predicted control to test data for analysis
    # predict_control() uses fitted model to predict control for each trial
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
    print("  - results/traditional_evc_model.pkl (fitted model)")
    print("  - results/traditional_evc_results.csv (performance metrics)")
    print("  - results/traditional_evc_predictions.csv (test predictions)")
    
    print("\nNext step: Run 'python3 step4_fit_bayesian_evc.py'")
    print("  (This will fit Bayesian EVC model for comparison)")


if __name__ == '__main__':
    main()
