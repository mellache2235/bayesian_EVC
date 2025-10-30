"""Generate dummy data for Bayesian EVC modeling."""

import numpy as np
import pandas as pd
import os


def generate_behavioral_data(n_participants=20, n_trials_per_participant=200, seed=42):
    np.random.seed(seed)
    data = []
    for pid in range(n_participants):
        base_rt = np.random.uniform(0.5, 1.5)
        accuracy_baseline = np.random.uniform(0.55, 0.85)
        for trial in range(n_trials_per_participant):
            evidence_clarity = np.random.beta(2, 2)
            rule_stability = 0.7 + 0.3 * np.sin((trial // 40) * np.pi / 4)
            decision_uncertainty = 1 - evidence_clarity
            state_uncertainty = 1 - rule_stability
            difficulty = (decision_uncertainty + state_uncertainty) / 2
            reaction_time = base_rt + difficulty * 0.5 + np.random.exponential(0.1)
            prob_correct = np.clip(accuracy_baseline * evidence_clarity + (1 - accuracy_baseline) * 0.3, 0.1, 0.95)
            correct = np.random.binomial(1, prob_correct)
            choice = correct if np.random.random() > 0.1 else 1 - correct
            reward = (10 if correct else 0) + state_uncertainty * 2
            data.append({
                'participant_id': pid, 'trial': trial, 'evidence_clarity': evidence_clarity,
                'rule_stability': rule_stability, 'decision_uncertainty': decision_uncertainty,
                'state_uncertainty': state_uncertainty, 'reaction_time': reaction_time,
                'choice': choice, 'correct': correct, 'reward': reward, 'difficulty': difficulty
            })
    return pd.DataFrame(data)


def generate_neural_data(behavioral_data, n_channels=20, seed=42):
    np.random.seed(seed)
    neural_data = []
    for (pid, trial), row in behavioral_data.groupby(['participant_id', 'trial']).first().iterrows():
        control_signal = (row['decision_uncertainty'] * 0.4 + row['state_uncertainty'] * 0.5 +
                         row['difficulty'] * 0.3 + np.random.normal(0, 0.1))
        channel_signals = {}
        for ch in range(n_channels):
            if ch < 5:
                signal = control_signal + np.random.normal(0, 0.2)
            elif ch < 10:
                signal = row['state_uncertainty'] * 0.5 + np.random.normal(0, 0.3)
            else:
                signal = np.random.normal(0, 0.5)
            channel_signals[f'channel_{ch}'] = signal
        neural_data.append({'participant_id': pid, 'trial': trial,
                           'control_signal': control_signal, **channel_signals})
    return pd.DataFrame(neural_data)


def save_data(behavioral_data, neural_data, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    behavioral_data.to_csv(f'{output_dir}/behavioral_data.csv', index=False)
    neural_data.to_csv(f'{output_dir}/neural_data.csv', index=False)
    print(f"Data saved to {output_dir}/")


if __name__ == '__main__':
    print("Generating dummy data...")
    behavioral_data = generate_behavioral_data(n_participants=20, n_trials_per_participant=200)
    neural_data = generate_neural_data(behavioral_data, n_channels=20)
    save_data(behavioral_data, neural_data)
    print(f"Generated {len(behavioral_data)} behavioral trials")
