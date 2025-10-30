"""
Step 2: Estimate Uncertainties Using Bayesian Methods

This step processes the behavioral data to add Bayesian uncertainty estimates.
"""

import pandas as pd
from models.bayesian_uncertainty import SequentialBayesianEstimator


def main():
    print("=" * 70)
    print("STEP 2: ESTIMATE UNCERTAINTIES")
    print("=" * 70)
    
    # Load data
    print("\nLoading behavioral data...")
    try:
        behavioral_data = pd.read_csv('data/behavioral_data.csv')
        print(f"✓ Loaded {len(behavioral_data)} trials")
    except FileNotFoundError:
        print("✗ Error: data/behavioral_data.csv not found!")
        print("  Please run 'python3 step1_generate_data.py' first.")
        return
    
    # Initialize estimator
    print("\nInitializing Bayesian uncertainty estimator...")
    estimator = SequentialBayesianEstimator(n_states=2, learning_rate=0.1)
    
    # Process data
    print("Processing trials to estimate uncertainties...")
    print("  (This adds Bayesian uncertainty estimates to each trial)")
    
    enhanced_data = estimator.process_subject_data(
        behavioral_data,
        subject_col='subject_id',
        evidence_col='evidence_clarity',
        outcome_col='accuracy'
    )
    
    # Save enhanced data
    print("\nSaving enhanced data with uncertainty estimates...")
    enhanced_data.to_csv('data/behavioral_data_with_uncertainties.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✓ UNCERTAINTY ESTIMATION COMPLETE!")
    print("=" * 70)
    
    print("\nAdded uncertainty measures:")
    print("  - decision_uncertainty: From evidence clarity")
    print("  - state_uncertainty: From Bayesian belief updating")
    print("  - combined_uncertainty: Integrated measure")
    print("  - confidence: Overall confidence level")
    
    print("\nUncertainty statistics:")
    print(enhanced_data[['decision_uncertainty', 'state_uncertainty', 
                         'combined_uncertainty', 'confidence']].describe().round(3))
    
    print("\nSaved to: data/behavioral_data_with_uncertainties.csv")
    print("\nNext step: Run 'python3 step3_fit_traditional_evc.py'")


if __name__ == '__main__':
    main()

