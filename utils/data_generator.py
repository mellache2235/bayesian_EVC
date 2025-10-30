"""
Data generator for simulating executive function task data with varying uncertainty levels.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class ExperimentalDataGenerator:
    """
    Generates dummy data simulating an executive function task with:
    - Decision uncertainty (evidence clarity)
    - State/rule uncertainty (rule stability)
    - Control allocation decisions
    - Behavioral outcomes (accuracy, RT)
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
    
    def generate_task_data(
        self, 
        n_subjects: int = 30,
        n_trials_per_subject: int = 200,
        n_blocks: int = 4
    ) -> pd.DataFrame:
        """
        Generate experimental task data.
        
        Args:
            n_subjects: Number of participants
            n_trials_per_subject: Trials per participant
            n_blocks: Number of task blocks with varying uncertainty
            
        Returns:
            DataFrame with trial-level data
        """
        data = []
        
        for subject_id in range(1, n_subjects + 1):
            # Individual differences in uncertainty tolerance
            uncertainty_tolerance = np.random.beta(2, 2)  # 0-1 scale
            baseline_control = np.random.uniform(0.3, 0.7)
            
            for trial in range(n_trials_per_subject):
                block = trial // (n_trials_per_subject // n_blocks)
                
                # Manipulate uncertainty levels across blocks
                if block == 0:  # Low uncertainty block
                    evidence_clarity = np.random.uniform(0.7, 0.9)
                    rule_stability = 0.9
                elif block == 1:  # High evidence uncertainty
                    evidence_clarity = np.random.uniform(0.3, 0.5)
                    rule_stability = 0.9
                elif block == 2:  # High rule uncertainty
                    evidence_clarity = np.random.uniform(0.7, 0.9)
                    rule_stability = 0.4
                else:  # High both uncertainties
                    evidence_clarity = np.random.uniform(0.3, 0.5)
                    rule_stability = 0.4
                
                # Decision uncertainty (inverse of evidence clarity)
                decision_uncertainty = 1 - evidence_clarity
                
                # State uncertainty (inverse of rule stability)
                state_uncertainty = 1 - rule_stability
                
                # Combined uncertainty
                total_uncertainty = (decision_uncertainty + state_uncertainty) / 2
                
                # Reward magnitude (varies by trial)
                reward_magnitude = np.random.choice([1, 2, 5, 10])
                
                # Control allocation (influenced by reward, accuracy, AND uncertainty)
                # Traditional EVC component: reward * accuracy
                reward_benefit = (reward_magnitude / 10.0) * evidence_clarity * 0.3
                
                # Uncertainty component (Bayesian innovation)
                uncertainty_benefit = uncertainty_tolerance * total_uncertainty * 0.3
                
                # Combined control signal
                control_signal = baseline_control + reward_benefit + uncertainty_benefit + \
                                np.random.normal(0, 0.1)
                control_signal = np.clip(control_signal, 0, 1)
                
                # Effort cost (increases with control)
                effort_cost = control_signal ** 2
                
                # Accuracy (influenced by evidence clarity and control)
                accuracy_prob = evidence_clarity * (0.5 + 0.5 * control_signal)
                accuracy = np.random.random() < accuracy_prob
                
                # Reaction time (influenced by uncertainty and control)
                base_rt = 500  # milliseconds
                uncertainty_rt = total_uncertainty * 300
                control_rt = -control_signal * 100  # more control -> faster
                rt = base_rt + uncertainty_rt + control_rt + np.random.normal(0, 50)
                rt = max(rt, 200)  # minimum RT
                
                # Actual reward (only if correct)
                obtained_reward = reward_magnitude if accuracy else 0
                
                # Confidence (inverse of uncertainty, modulated by control)
                confidence = (1 - total_uncertainty) * (0.5 + 0.5 * control_signal)
                confidence = np.clip(confidence, 0, 1)
                
                data.append({
                    'subject_id': subject_id,
                    'trial': trial + 1,
                    'block': block + 1,
                    'evidence_clarity': evidence_clarity,
                    'rule_stability': rule_stability,
                    'decision_uncertainty': decision_uncertainty,
                    'state_uncertainty': state_uncertainty,
                    'total_uncertainty': total_uncertainty,
                    'reward_magnitude': reward_magnitude,
                    'control_signal': control_signal,
                    'effort_cost': effort_cost,
                    'accuracy': int(accuracy),
                    'reaction_time': rt,
                    'obtained_reward': obtained_reward,
                    'confidence': confidence,
                    'uncertainty_tolerance': uncertainty_tolerance
                })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_neural_data(self, behavioral_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate simulated neural data (fNIRS/fMRI) corresponding to behavioral trials.
        
        Args:
            behavioral_data: DataFrame with behavioral trial data
            
        Returns:
            DataFrame with neural activity measures
        """
        neural_data = []
        
        for _, row in behavioral_data.iterrows():
            # Simulate DLPFC activity (related to cognitive control)
            dlpfc_activity = row['control_signal'] * 2.5 + \
                           np.random.normal(0, 0.3)
            dlpfc_activity = max(dlpfc_activity, 0)
            
            # Simulate ACC activity (related to conflict/uncertainty)
            acc_activity = row['total_uncertainty'] * 2.0 + \
                          row['control_signal'] * 0.5 + \
                          np.random.normal(0, 0.3)
            acc_activity = max(acc_activity, 0)
            
            # Simulate striatal activity (related to reward)
            striatal_activity = (row['reward_magnitude'] / 10) * 1.5 + \
                               row['accuracy'] * 0.5 + \
                               np.random.normal(0, 0.2)
            striatal_activity = max(striatal_activity, 0)
            
            neural_data.append({
                'subject_id': row['subject_id'],
                'trial': row['trial'],
                'dlpfc_activity': dlpfc_activity,
                'acc_activity': acc_activity,
                'striatal_activity': striatal_activity
            })
        
        return pd.DataFrame(neural_data)
    
    def save_data(
        self, 
        behavioral_data: pd.DataFrame,
        neural_data: pd.DataFrame,
        output_dir: str = 'data'
    ):
        """
        Save generated data to CSV files.
        
        Args:
            behavioral_data: Behavioral trial data
            neural_data: Neural activity data
            output_dir: Directory to save files
        """
        behavioral_data.to_csv(f'{output_dir}/behavioral_data.csv', index=False)
        neural_data.to_csv(f'{output_dir}/neural_data.csv', index=False)
        
        # Save summary statistics
        summary = behavioral_data.groupby('block').agg({
            'accuracy': 'mean',
            'reaction_time': 'mean',
            'control_signal': 'mean',
            'total_uncertainty': 'mean',
            'confidence': 'mean'
        }).round(3)
        
        summary.to_csv(f'{output_dir}/summary_statistics.csv')
        
        print(f"Data saved to {output_dir}/")
        print(f"  - behavioral_data.csv: {len(behavioral_data)} trials")
        print(f"  - neural_data.csv: {len(neural_data)} trials")
        print(f"  - summary_statistics.csv")


def main():
    """Generate and save dummy experimental data."""
    generator = ExperimentalDataGenerator(seed=42)
    
    print("Generating experimental data...")
    behavioral_data = generator.generate_task_data(
        n_subjects=30,
        n_trials_per_subject=200,
        n_blocks=4
    )
    
    print("Generating neural data...")
    neural_data = generator.generate_neural_data(behavioral_data)
    
    print("\nSaving data...")
    generator.save_data(behavioral_data, neural_data)
    
    print("\nData generation complete!")
    print(f"\nSample data (first 5 trials):")
    print(behavioral_data.head())


if __name__ == '__main__':
    main()

