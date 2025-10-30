"""
Run the complete Bayesian EVC pipeline.

This is the main entry point that runs all steps in sequence.
"""

import sys
sys.path.insert(0, '/Users/hari/.cursor/worktrees/bayesian_EVC/645wx')

from src.pipeline import EVCPipeline, ModelConfig
import pandas as pd


def main():
    """Run the complete pipeline."""
    print("=" * 70)
    print("BAYESIAN EVC PIPELINE")
    print("=" * 70)
    
    # Configuration
    data_path = "data/structured_evc_trials.csv"
    config = ModelConfig()
    
    # Initialize pipeline
    print("\nInitializing pipeline...")
    pipeline = EVCPipeline(data_path=data_path, config=config)
    
    # Run pipeline
    print("Running pipeline...")
    results = pipeline.run()
    
    # Get summary
    summary = pipeline.summarise()
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n=== Example Results (first 10 rows) ===")
    print(results[[
        'child_id', 'trial_id', 
        'posterior_rule_uncertainty', 
        'evc_score',
        'predicted_control_allocation',
        'predicted_accuracy'
    ]].head(10))
    
    print("\n=== Per-Child Summary ===")
    print(summary)
    
    # Save results
    print("\nSaving results...")
    results.to_csv('results/pipeline_results.csv', index=False)
    summary.to_csv('results/pipeline_summary.csv', index=False)
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nResults saved to:")
    print("  - results/pipeline_results.csv")
    print("  - results/pipeline_summary.csv")


if __name__ == "__main__":
    main()

