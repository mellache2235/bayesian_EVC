"""
Generate dummy data for Bayesian EVC modeling.

This module generates synthetic behavioral and neural data that simulates
an executive functioning task with varying levels of evidence uncertainty
and state/rule uncertainty.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def generate_behavioral_data(
    n_participants: int = 20,
    n_trials_per_participant: int = 200,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate dummy behavioral data for Bayesian EVC modeling.
    
    Parameters:
    -----------
    n_participants : int
        Number of participants
    n_trials_per_participant : int
        Number of trials per participant
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Behavioral data with columns:
        - participant_id: participant identifier
        - trial: trial number
        - evidence_clarity: clarity of evidence (higher = clearer)
        - rule_stability: stability of task rules (higher = more stable)
        - decision_uncertainty: computed decision uncertainty
        - state_uncertainty: computed state/rule uncertainty
        - reaction_time: reaction time in seconds
        - choice: choice made (0 or 1)
        - correct: whether choice was correct (0 or 1)
        - reward: reward received
        - difficulty: trial difficulty level
    """
    np.random.seed(seed)
    
    data = []
    
    for pid in range(n_participants):
        # Participant-level parameters
        base_rt = np.random.uniform(0.5, 1.5)  # Base reaction time
        accuracy_baseline = np.random.uniform(0.55, 0.85)  # Baseline accuracy
        
        # Generate trial-level parameters
        for trial in range(n_trials_per_participant):
            # Evidence clarity: varies trial-to-trial (higher = clearer evidence)
            evidence_clarity = np.random.beta(2, 2)  # Range [0, 1]
            
            # Rule stability: varies within blocks (higher = more stable rules)
            # Simulate rule changes periodically
            block = trial // 40
            rule_stability = 0.7 + 0.3 * np.sin(block * np.pi / 4)  # Oscillates
            
            # Compute uncertainties
            # Decision uncertainty: inversely related to evidence clarity
            decision_uncertainty = 1 - evidence_clarity
            
            # State uncertainty: inversely related to rule stability
            state_uncertainty = 1 - rule_stability
            
            # Difficulty: combination of both uncertainties
            difficulty = (decision_uncertainty + state_uncertainty) / 2
            
            # Generate reaction time: influenced by uncertainty and difficulty
            rt_noise = np.random.exponential(0.1)
            reaction_time = base_rt + difficulty * 0.5 + rt_noise
            
            # Generate choice and correctness
            # Higher evidence clarity -> higher accuracy
            prob_correct = accuracy_baseline * evidence_clarity + (1 - accuracy_baseline) * 0.3
            prob_correct = np.clip(prob_correct, 0.1, 0.95)
            
            correct = np.random.binomial(1, prob_correct)
            choice = correct if np.random.random() > 0.1 else 1 - correct  # 10% random errors
            
            # Generate reward: higher for correct choices, modulated by uncertainty
            base_reward = 10 if correct else 0
            uncertainty_bonus = state_uncertainty * 2  # Bonus for reducing uncertainty
            reward = base_reward + uncertainty_bonus
            
            data.append({
                'participant_id': pid,
                'trial': trial,
                'evidence_clarity': evidence_clarity,
                'rule_stability': rule_stability,
                'decision_uncertainty': decision_uncertainty,
                'state_uncertainty': state_uncertainty,
                'reaction_time': reaction_time,
                'choice': choice,
                'correct': correct,
                'reward': reward,
                'difficulty': difficulty
            })
    
    df = pd.DataFrame(data)
    return df


def generate_neural_data(
    behavioral_data: pd.DataFrame,
    n_channels: int = 20,
    sampling_rate: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate dummy neural data (fNIRS/fMRI-like) correlated with behavioral measures.
    
    Parameters:
    -----------
    behavioral_data : pd.DataFrame
        Behavioral data from generate_behavioral_data
    n_channels : int
        Number of neural channels/regions
    sampling_rate : int
        Sampling rate (Hz)
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Neural data with columns:
        - participant_id: participant identifier
        - trial: trial number
        - channel_0, channel_1, ...: neural activity for each channel
        - control_signal: aggregated control signal (correlated with effort)
    """
    np.random.seed(seed)
    
    neural_data = []
    
    for (pid, trial), row in behavioral_data.groupby(['participant_id', 'trial']).first().iterrows():
        # Generate control signal: correlated with uncertainty and effort
        # Higher uncertainty -> higher control signal
        control_signal = (
            row['decision_uncertainty'] * 0.4 +
            row['state_uncertainty'] * 0.5 +
            row['difficulty'] * 0.3 +
            np.random.normal(0, 0.1)
        )
        
        # Generate channel-specific signals
        channel_signals = {}
        for ch in range(n_channels):
            # Some channels more correlated with control
            if ch < 5:  # Control-related channels
                signal = control_signal + np.random.normal(0, 0.2)
            elif ch < 10:  # Uncertainty-related channels
                signal = row['state_uncertainty'] * 0.5 + np.random.normal(0, 0.3)
            else:  # Noise channels
                signal = np.random.normal(0, 0.5)
            
            channel_signals[f'channel_{ch}'] = signal
        
        neural_data.append({
            'participant_id': pid,
            'trial': trial,
            'control_signal': control_signal,
            **channel_signals
        })
    
    df_neural = pd.DataFrame(neural_data)
    return df_neural


def save_data(
    behavioral_data: pd.DataFrame,
    neural_data: pd.DataFrame,
    output_dir: str = 'data'
):
    """
    Save generated data to CSV files.
    
    Parameters:
    -----------
    behavioral_data : pd.DataFrame
        Behavioral data
    neural_data : pd.DataFrame
        Neural data
    output_dir : str
        Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    behavioral_data.to_csv(f'{output_dir}/behavioral_data.csv', index=False)
    neural_data.to_csv(f'{output_dir}/neural_data.csv', index=False)
    
    print(f"Data saved to {output_dir}/")
    print(f"  - behavioral_data.csv: {len(behavioral_data)} rows")
    print(f"  - neural_data.csv: {len(neural_data)} rows")


if __name__ == '__main__':
    # Generate and save dummy data
    print("Generating dummy behavioral data...")
    behavioral_data = generate_behavioral_data(n_participants=20, n_trials_per_participant=200)
    
    print("Generating dummy neural data...")
    neural_data = generate_neural_data(behavioral_data, n_channels=20)
    
    print("Saving data...")
    save_data(behavioral_data, neural_data)
    
    print("\nData summary:")
    print(behavioral_data.describe())
    print("\nNeural data summary:")
    print(neural_data[['control_signal'] + [f'channel_{i}' for i in range(5)]].describe())

