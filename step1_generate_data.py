"""
Step 1: Generate Experimental Data

PURPOSE:
--------
This script creates simulated experimental data for your Bayesian EVC analysis.
It generates behavioral data (trials with varying uncertainty, rewards, control signals)
and corresponding neural data (simulated brain activity).

WHAT THIS STEP DOES:
-------------------
1. Creates a dataset of experimental trials
2. Manipulates uncertainty levels across blocks
3. Simulates control allocation decisions
4. Generates behavioral outcomes (accuracy, reaction time)
5. Creates corresponding neural activity data
6. Saves everything to CSV files

WHY START HERE:
--------------
- You need data before you can fit models
- This creates a controlled dataset with known structure
- Allows you to test if your models can recover known patterns
- In real experiments, you'd replace this with loading actual data

KEY PARAMETERS:
--------------
- n_subjects: Number of participants (default: 30)
- n_trials_per_subject: Trials per participant (default: 200)
- n_blocks: Number of experimental blocks with different uncertainty levels (default: 4)

OUTPUT FILES:
------------
- data/behavioral_data.csv: Trial-level behavioral data
- data/neural_data.csv: Simulated neural activity
- data/summary_statistics.csv: Block-level summaries
"""

import os
from utils.data_generator import ExperimentalDataGenerator


def main():
    """
    Main function to generate experimental data.
    
    WORKFLOW:
    ---------
    1. Create data directory
    2. Initialize data generator (with random seed for reproducibility)
    3. Generate behavioral data (trials with varying conditions)
    4. Generate neural data (corresponding brain activity)
    5. Save all data to CSV files
    6. Display summary statistics
    """
    print("=" * 70)
    print("STEP 1: GENERATE EXPERIMENTAL DATA")
    print("=" * 70)
    
    # CREATE DATA DIRECTORY
    # Ensure the 'data/' folder exists to save output files
    # exist_ok=True means don't error if folder already exists
    os.makedirs('data', exist_ok=True)
    
    # INITIALIZE DATA GENERATOR
    # ExperimentalDataGenerator creates simulated experimental data
    # seed=42 ensures reproducibility (same random seed = same data)
    # 
    # What does the generator do?
    # - Creates trials with varying uncertainty levels
    # - Simulates control allocation decisions
    # - Generates behavioral outcomes (accuracy, RT)
    generator = ExperimentalDataGenerator(seed=42)
    
    # GENERATE BEHAVIORAL DATA
    # This creates the main dataset of experimental trials
    #
    # Parameters:
    # - n_subjects=30: 30 participants
    # - n_trials_per_subject=200: Each participant does 200 trials
    # - n_blocks=4: 4 experimental blocks with different uncertainty levels
    #
    # Returns: DataFrame with columns like:
    #   - subject_id: Which participant
    #   - trial: Trial number
    #   - block: Which experimental block
    #   - evidence_clarity: How clear the evidence is (0-1)
    #   - total_uncertainty: Combined uncertainty (0-1)
    #   - reward_magnitude: Reward value for this trial
    #   - control_signal: How much control was allocated (0-1)
    #   - accuracy: Whether response was correct (0 or 1)
    #   - reaction_time: Response time in milliseconds
    print("\nGenerating behavioral data...")
    behavioral_data = generator.generate_task_data(
        n_subjects=30,              # Number of participants
        n_trials_per_subject=200,   # Trials per participant
        n_blocks=4                  # Experimental blocks
    )
    
    # GENERATE NEURAL DATA
    # Creates simulated brain activity corresponding to behavioral trials
    #
    # Neural regions simulated:
    # - DLPFC (dorsolateral prefrontal cortex): Related to cognitive control
    # - ACC (anterior cingulate cortex): Related to conflict/uncertainty
    # - Striatum: Related to reward processing
    #
    # The neural activity is correlated with behavioral measures:
    # - DLPFC activity ∝ control_signal (more control → more DLPFC activity)
    # - ACC activity ∝ total_uncertainty (more uncertainty → more ACC activity)
    # - Striatal activity ∝ reward_magnitude (more reward → more striatal activity)
    print("Generating neural data...")
    neural_data = generator.generate_neural_data(behavioral_data)
    
    # SAVE DATA TO FILES
    # Saves all generated data to CSV files in the 'data/' directory
    #
    # Files created:
    # 1. behavioral_data.csv: All trial-level behavioral data
    # 2. neural_data.csv: All neural activity data
    # 3. summary_statistics.csv: Block-level averages
    print("\nSaving data to 'data/' directory...")
    generator.save_data(behavioral_data, neural_data, output_dir='data')
    
    # DISPLAY SUMMARY
    print("\n" + "=" * 70)
    print("✓ DATA GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - data/behavioral_data.csv ({len(behavioral_data)} trials)")
    print(f"  - data/neural_data.csv ({len(neural_data)} observations)")
    print(f"  - data/summary_statistics.csv")
    
    # SHOW SUMMARY STATISTICS BY BLOCK
    # This helps verify that uncertainty was manipulated correctly across blocks
    # You should see:
    # - Block 1: Low uncertainty (high clarity, high control)
    # - Block 2: High evidence uncertainty (low clarity)
    # - Block 3: High rule uncertainty (low stability)
    # - Block 4: High both uncertainties
    print("\nData summary by block:")
    print(behavioral_data.groupby('block').agg({
        'total_uncertainty': ['mean', 'std'],  # Average uncertainty per block
        'control_signal': ['mean', 'std'],      # Average control allocation
        'accuracy': 'mean',                     # Average accuracy
        'reaction_time': 'mean'                 # Average reaction time
    }).round(3))
    
    print("\nNext step: Run 'python3 step2_estimate_uncertainty.py'")


if __name__ == '__main__':
    main()

