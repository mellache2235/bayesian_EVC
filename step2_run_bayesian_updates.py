"""
Step 2: Run Bayesian belief updates.

Maintains per-participant estimates of uncertainty about task rules
and evidence reliability using Bayesian updates.
"""

import sys
sys.path.insert(0, '/Users/hari/.cursor/worktrees/bayesian_EVC/645wx')

from src.pipeline import EVCPipeline, ModelConfig, BayesianEVCModel
import pandas as pd


def main():
    print("=" * 70)
    print("STEP 2: RUN BAYESIAN BELIEF UPDATES")
    print("=" * 70)
    
    data_path = "data/structured_evc_trials.csv"
    config = ModelConfig()
    
    # Initialize
    pipeline = EVCPipeline(data_path=data_path, config=config)
    df = pipeline.load_data()
    
    print(f"\nProcessing {len(df)} trials with Bayesian updates...")
    print(f"Configuration:")
    print(f"  - Rule alpha (prior): {config.belief_prior.rule_alpha}")
    print(f"  - Rule beta (prior): {config.belief_prior.rule_beta}")
    print(f"  - Evidence strength: {config.belief_prior.evidence_strength}")
    print(f"  - Volatility discount: {config.belief_prior.volatility_discount}")
    
    # Run pipeline
    results = pipeline.run()
    
    print("\n" + "-" * 70)
    print("BAYESIAN UPDATE RESULTS")
    print("-" * 70)
    
    # Show example of belief evolution for one child
    child_id = results['child_id'].iloc[0]
    child_data = results[results['child_id'] == child_id].head(10)
    
    print(f"\nExample: Belief evolution for child {child_id} (first 10 trials):")
    print(child_data[[
        'trial_id',
        'rule_stability',
        'evidence_clarity',
        'posterior_rule_confidence',
        'posterior_rule_uncertainty',
        'posterior_evidence_precision'
    ]])
    
    # Summary statistics
    print(f"\nOverall statistics:")
    print(f"  Mean rule confidence: {results['posterior_rule_confidence'].mean():.3f}")
    print(f"  Mean rule uncertainty: {results['posterior_rule_uncertainty'].mean():.3f}")
    print(f"  Mean evidence precision: {results['posterior_evidence_precision'].mean():.3f}")
    
    # Save intermediate results
    results.to_csv('results/bayesian_updates.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✓ BAYESIAN UPDATES COMPLETE!")
    print("=" * 70)
    print("\nResults saved to: results/bayesian_updates.csv")
    print("\nNext step: Run 'python step3_compute_evc.py'")


if __name__ == "__main__":
    main()

