"""
Step 3: Compute Expected Value of Control (EVC).

Combines uncertainty estimates with reward and effort-cost information
to compute an EVC signal that explicitly prices uncertainty reduction benefits.
"""

import sys
sys.path.insert(0, '/Users/hari/.cursor/worktrees/bayesian_EVC/645wx')

from src.pipeline import EVCPipeline, ModelConfig
import pandas as pd


def main():
    print("=" * 70)
    print("STEP 3: COMPUTE EVC SCORES")
    print("=" * 70)
    
    data_path = "data/structured_evc_trials.csv"
    config = ModelConfig()
    
    # Initialize and run
    pipeline = EVCPipeline(data_path=data_path, config=config)
    results = pipeline.run()
    
    print(f"\nEVC Configuration:")
    print(f"  - Reward weight: {config.evc_weights.reward}")
    print(f"  - Effort cost weight: {config.evc_weights.effort_cost}")
    print(f"  - Uncertainty reduction weight: {config.evc_weights.uncertainty_reduction}")
    print(f"  - Control temperature: {config.control_temperature}")
    
    print("\n" + "-" * 70)
    print("EVC COMPUTATION RESULTS")
    print("-" * 70)
    
    # Show EVC components
    print(f"\nEVC Components (first 10 trials):")
    print(results[[
        'child_id', 'trial_id',
        'reward', 'effort_cost',
        'expected_uncertainty_reduction',
        'evc_score',
        'predicted_control_allocation'
    ]].head(10))
    
    # Statistics
    print(f"\nEVC Statistics:")
    print(f"  Mean EVC score: {results['evc_score'].mean():.3f}")
    print(f"  Std EVC score: {results['evc_score'].std():.3f}")
    print(f"  Min EVC score: {results['evc_score'].min():.3f}")
    print(f"  Max EVC score: {results['evc_score'].max():.3f}")
    
    print(f"\nControl Allocation Statistics:")
    print(f"  Mean control: {results['predicted_control_allocation'].mean():.3f}")
    print(f"  Std control: {results['predicted_control_allocation'].std():.3f}")
    
    print(f"\nUncertainty Reduction Statistics:")
    print(f"  Mean reduction: {results['expected_uncertainty_reduction'].mean():.3f}")
    print(f"  Std reduction: {results['expected_uncertainty_reduction'].std():.3f}")
    
    # Save results
    results.to_csv('results/evc_scores.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✓ EVC COMPUTATION COMPLETE!")
    print("=" * 70)
    print("\nResults saved to: results/evc_scores.csv")
    print("\nNext step: Run 'python step4_visualize.py'")


if __name__ == "__main__":
    main()

