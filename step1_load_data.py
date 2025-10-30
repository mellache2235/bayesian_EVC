"""
Step 1: Load and validate data.

Loads task-level observations with evidence clarity and rule stability.
"""

import sys
sys.path.insert(0, '/Users/hari/.cursor/worktrees/bayesian_EVC/645wx')

from src.pipeline import EVCPipeline, ModelConfig
import pandas as pd


def main():
    print("=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)
    
    data_path = "data/structured_evc_trials.csv"
    
    # Initialize pipeline
    pipeline = EVCPipeline(data_path=data_path, config=ModelConfig())
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pipeline.load_data()
    
    print(f"\n✓ Loaded {len(df)} trials")
    print(f"✓ {df['child_id'].nunique()} unique children")
    
    # Display data info
    print("\n" + "-" * 70)
    print("DATA SUMMARY")
    print("-" * 70)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nData statistics:")
    print(df.describe())
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠ Missing values:")
        print(missing[missing > 0])
    else:
        print(f"\n✓ No missing values")
    
    print("\n" + "=" * 70)
    print("✓ DATA LOADED SUCCESSFULLY!")
    print("=" * 70)
    print("\nNext step: Run 'python step2_run_bayesian_updates.py'")


if __name__ == "__main__":
    main()

