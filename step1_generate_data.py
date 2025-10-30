"""
Step 1: Generate Experimental Data

Run this first to create dummy data for the analysis.
"""

import os
from utils.data_generator import ExperimentalDataGenerator


def main():
    print("=" * 70)
    print("STEP 1: GENERATE EXPERIMENTAL DATA")
    print("=" * 70)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Initialize generator
    generator = ExperimentalDataGenerator(seed=42)
    
    # Generate data
    print("\nGenerating behavioral data...")
    behavioral_data = generator.generate_task_data(
        n_subjects=30,
        n_trials_per_subject=200,
        n_blocks=4
    )
    
    print("Generating neural data...")
    neural_data = generator.generate_neural_data(behavioral_data)
    
    # Save data
    print("\nSaving data to 'data/' directory...")
    generator.save_data(behavioral_data, neural_data, output_dir='data')
    
    print("\n" + "=" * 70)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - data/behavioral_data.csv ({len(behavioral_data)} trials)")
    print(f"  - data/neural_data.csv ({len(neural_data)} observations)")
    print(f"  - data/summary_statistics.csv")
    
    print("\nData summary by block:")
    print(behavioral_data.groupby('block').agg({
        'total_uncertainty': ['mean', 'std'],
        'control_signal': ['mean', 'std'],
        'accuracy': 'mean',
        'reaction_time': 'mean'
    }).round(3))
    
    print("\nNext step: Run 'python3 step2_estimate_uncertainty.py'")


if __name__ == '__main__':
    main()

