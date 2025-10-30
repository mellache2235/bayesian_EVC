"""
Step 2: Estimate Uncertainties Using Bayesian Methods

PURPOSE:
--------
This step processes behavioral data to add Bayesian uncertainty estimates.
It uses Bayesian inference to quantify two types of uncertainty:
1. Decision uncertainty (from evidence clarity)
2. State uncertainty (from belief updating about task rules)

WHAT THIS STEP DOES:
-------------------
1. Loads behavioral data created in Step 1
2. Initializes a Bayesian uncertainty estimator
3. Processes each trial sequentially using Bayesian inference
4. Estimates decision uncertainty from evidence clarity
5. Estimates state uncertainty from belief updating
6. Combines both types of uncertainty
7. Saves enhanced data with uncertainty measures

WHY THIS STEP MATTERS:
--------------------
- The original data has uncertainty measures, but this adds Bayesian estimates
- Bayesian uncertainty uses principled inference (Bayes' rule)
- Tracks how uncertainty evolves over trials
- Provides entropy-based uncertainty measures
- This step is OPTIONAL - you can skip it if using original uncertainty columns

KEY CONCEPTS:
------------
1. Decision Uncertainty: How uncertain you are about what the evidence means
   - Currently: COMPUTED from evidence_clarity (1 - evidence_clarity)
   - Evidence_clarity: Currently SET in simulation, but would be INFERRED in real experiments
     → From RT, accuracy, confidence ratings, or DDM parameters
   - Higher evidence clarity → lower decision uncertainty
   - Measured using entropy: H = -Σ P(x) × log₂(P(x))

2. State Uncertainty: How uncertain you are about which task rule is active
   - **INFERRED** from behavioral data (accuracy outcomes) using Bayesian updating
   - Uses Bayes' rule: P(rule|observation) ∝ P(observation|rule) × P(rule)
   - Tracks beliefs about multiple possible task states
   - This is TRUE INFERENCE from observations
   - Measured using entropy of belief distribution

3. Combined Uncertainty: Integrated measure of both types
   - Typically: combined = 0.5 × decision + 0.5 × state
   - Provides overall uncertainty level for EVC calculations

⚠️ PARAMETER SOURCES:
---------------------
- evidence_clarity: SET (simulation) → Would be INFERRED in real experiments
- decision_uncertainty: COMPUTED from evidence_clarity
- state_uncertainty: INFERRED from observations (Bayesian updating)
- confidence: COMPUTED from uncertainty
- entropy: COMPUTED from probabilities
- uncertainty_weight (λ): INFERRED from model fitting (Step 4)

OUTPUT FILES:
------------
- data/behavioral_data_with_uncertainties.csv: Original data + Bayesian uncertainty estimates
"""

import pandas as pd
from models.bayesian_uncertainty import SequentialBayesianEstimator


def main():
    """
    Main function to estimate uncertainties using Bayesian methods.
    
    WORKFLOW:
    ---------
    1. Load behavioral data from Step 1
    2. Initialize Bayesian uncertainty estimator
    3. Process data sequentially (trial by trial)
    4. Estimate decision uncertainty (from evidence)
    5. Estimate state uncertainty (from belief updating)
    6. Combine uncertainties
    7. Save enhanced dataset
    """
    print("=" * 70)
    print("STEP 2: ESTIMATE UNCERTAINTIES")
    print("=" * 70)
    
    # LOAD BEHAVIORAL DATA
    # This step requires data created in Step 1
    # We're reading the CSV file that contains trial-level behavioral data
    # Expected columns: subject_id, trial, evidence_clarity, accuracy, etc.
    print("\nLoading behavioral data...")
    try:
        # pd.read_csv() loads the CSV file into a pandas DataFrame
        # This DataFrame contains all trial-level data (subject, trial, behavioral measures)
        behavioral_data = pd.read_csv('data/behavioral_data.csv')
        print(f"✓ Loaded {len(behavioral_data)} trials")
    except FileNotFoundError:
        # Error handling: If the file doesn't exist, Step 1 hasn't been run
        print("✗ Error: data/behavioral_data.csv not found!")
        print("  Please run 'python3 step1_generate_data.py' first.")
        return
    
    # INITIALIZE BAYESIAN UNCERTAINTY ESTIMATOR
    # SequentialBayesianEstimator processes trials sequentially using Bayesian inference
    #
    # Parameters:
    # - n_states=2: Number of possible task states/rules
    #   Example: If task can have 2 rules (e.g., "respond to color" vs "respond to shape")
    #   This tells the estimator to track beliefs about 2 possible states
    #
    # - learning_rate=0.1: Rate of belief updating (0-1)
    #   - Higher learning_rate (e.g., 0.5): Beliefs update quickly with new evidence
    #   - Lower learning_rate (e.g., 0.1): Beliefs update slowly (more stable)
    #   - Formula: new_belief = (1 - α) × old_belief + α × posterior
    #   - α = learning_rate: how much to trust new observation
    #
    # What does the estimator do?
    # - Processes each trial sequentially
    # - Updates beliefs about task states using Bayes' rule
    # - Computes uncertainty measures (entropy-based)
    # - Tracks uncertainty evolution over time
    print("\nInitializing Bayesian uncertainty estimator...")
    estimator = SequentialBayesianEstimator(
        n_states=2,           # Number of possible task states/rules
        learning_rate=0.1     # Rate of belief updating (how quickly to learn)
    )
    
    # PROCESS DATA TO ESTIMATE UNCERTAINTIES
    # This method processes all subjects and trials to compute uncertainty estimates
    #
    # process_subject_data() does:
    # 1. Groups data by subject (each subject's trials are processed separately)
    # 2. For each subject, resets beliefs at the start
    # 3. Processes trials sequentially:
    #    - Estimates decision uncertainty from evidence_clarity
    #    - Updates state beliefs using Bayesian inference
    #    - Estimates state uncertainty from belief distribution
    #    - Combines both uncertainties
    # 4. Returns enhanced data with new uncertainty columns
    #
    # Parameters:
    # - behavioral_data: DataFrame with trial data
    # - subject_col='subject_id': Column identifying subjects (beliefs reset per subject)
    # - evidence_col='evidence_clarity': Column with evidence clarity (0-1)
    # - outcome_col='accuracy': Column with trial outcomes (for belief updating)
    #
    # Returns: DataFrame with added columns:
    #   - decision_uncertainty: Uncertainty from evidence (0-1)
    #   - state_uncertainty: Uncertainty from beliefs (0-1)
    #   - combined_uncertainty: Combined measure (0-1)
    #   - confidence: Overall confidence level (0-1)
    print("Processing trials to estimate uncertainties...")
    print("  (This adds Bayesian uncertainty estimates to each trial)")
    
    enhanced_data = estimator.process_subject_data(
        behavioral_data,                    # Input data with trials
        subject_col='subject_id',           # Column identifying subjects
        evidence_col='evidence_clarity',    # Column with evidence clarity
        outcome_col='accuracy'              # Column with trial outcomes (for belief updating)
    )
    
    # SAVE ENHANCED DATA
    # Save the data with added uncertainty estimates to a new CSV file
    # This file will be used in later steps (though original uncertainty columns can also be used)
    print("\nSaving enhanced data with uncertainty estimates...")
    enhanced_data.to_csv('data/behavioral_data_with_uncertainties.csv', index=False)
    # index=False means don't save row numbers as a column
    
    print("\n" + "=" * 70)
    print("✓ UNCERTAINTY ESTIMATION COMPLETE!")
    print("=" * 70)
    
    # DISPLAY SUMMARY OF ADDED UNCERTAINTY MEASURES
    print("\nAdded uncertainty measures:")
    print("  - decision_uncertainty: From evidence clarity")
    print("    Formula: decision_uncertainty = 1 - evidence_clarity")
    print("    Interpretation: How uncertain am I about what this evidence means?")
    
    print("  - state_uncertainty: From Bayesian belief updating")
    print("    Formula: state_uncertainty = entropy(beliefs) / max_entropy")
    print("    Interpretation: How uncertain am I about which task rule is active?")
    
    print("  - combined_uncertainty: Integrated measure")
    print("    Formula: combined = 0.5 × decision + 0.5 × state")
    print("    Interpretation: Overall uncertainty level")
    
    print("  - confidence: Overall confidence level")
    print("    Formula: confidence = 1 - combined_uncertainty")
    print("    Interpretation: How confident am I overall?")
    
    # SHOW STATISTICS
    # .describe() computes summary statistics: mean, std, min, max, quartiles
    # This helps verify that uncertainty values are in reasonable ranges (0-1)
    print("\nUncertainty statistics:")
    print(enhanced_data[['decision_uncertainty', 'state_uncertainty', 
                         'combined_uncertainty', 'confidence']].describe().round(3))
    
    print("\nSaved to: data/behavioral_data_with_uncertainties.csv")
    print("\nNOTE: This step is OPTIONAL. You can proceed to Step 3 using original data.")
    print("\nNext step: Run 'python3 step3_fit_traditional_evc.py'")


if __name__ == '__main__':
    main()

