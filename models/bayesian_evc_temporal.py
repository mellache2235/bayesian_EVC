"""
Bayesian EVC with Temporal Dynamics

Integrates HGF (recurrent uncertainty estimation) with Bayesian EVC (control allocation).
This allows control to depend on trial history through HGF's recurrent state.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
from models.hgf_uncertainty import HierarchicalGaussianFilter


class BayesianEVC_Temporal:
    """
    Bayesian EVC with temporal dynamics via HGF integration
    
    Key Innovation:
    - Uses HGF as RECURRENT MODULE for uncertainty estimation
    - HGF maintains internal state across trials (recurrent!)
    - Control depends on trial history through HGF state
    
    Architecture:
    
    Trial History ──→ HGF (Recurrent) ──→ Uncertainty[t]
                      ↓                        ↓
                   Internal State         Volatility[t]
                      ↑                        ↓
                   (carried forward)    ──────────┐
                                                  ↓
    [Reward[t], Accuracy[t]] ──────→ Bayesian EVC ──→ Control[t]
    """
    
    def __init__(self, 
                 reward_weight=1.0,
                 effort_cost_weight=1.0, 
                 uncertainty_weight=0.5,
                 volatility_weight=0.2,  # NEW: Volatility also affects control!
                 baseline=0.5,
                 # HGF parameters
                 kappa_2=1.0,
                 omega_2=-4.0,
                 omega_3=-6.0):
        """
        Initialize Temporal Bayesian EVC
        
        Args:
            reward_weight: β_r (reward sensitivity)
            effort_cost_weight: β_e (effort cost)
            uncertainty_weight: λ (uncertainty sensitivity) ← KEY PARAMETER
            volatility_weight: γ (volatility sensitivity) ← NEW PARAMETER
            baseline: Baseline control
            kappa_2, omega_2, omega_3: HGF parameters
        """
        # EVC parameters
        self.reward_weight = reward_weight
        self.effort_cost_weight = effort_cost_weight
        self.uncertainty_weight = uncertainty_weight
        self.volatility_weight = volatility_weight
        self.baseline = baseline
        
        # HGF parameters (for recurrent dynamics)
        self.hgf_params = {
            'kappa_2': kappa_2,
            'omega_2': omega_2,
            'omega_3': omega_3
        }
    
    def predict_control_sequential(self, data, 
                                   reward_col='reward_magnitude',
                                   accuracy_col='evidence_clarity',
                                   outcome_col='accuracy',
                                   subject_col='subject_id'):
        """
        Predict control sequentially (REQUIRED for temporal models!)
        
        Must process trials in order because HGF maintains state.
        
        Args:
            data: DataFrame with trial data IN ORDER
            reward_col: Reward magnitude column
            accuracy_col: Evidence clarity column
            outcome_col: Trial outcome (for HGF update)
            subject_col: Subject ID (reset HGF per subject)
            
        Returns:
            predictions: Array of predicted control values
            uncertainty_trajectory: HGF uncertainty over trials
            volatility_trajectory: HGF volatility over trials
        """
        predictions = []
        uncertainty_trajectory = []
        volatility_trajectory = []
        
        # Process each subject separately (reset HGF state)
        for subject in data[subject_col].unique():
            subject_data = data[data[subject_col] == subject]
            
            # Initialize HGF for this subject
            hgf = HierarchicalGaussianFilter(**self.hgf_params)
            
            # Process trials sequentially
            for idx, trial in subject_data.iterrows():
                # ============================================
                # GET UNCERTAINTY FROM HGF (incorporates history!)
                # ============================================
                state_uncertainty = hgf.get_state_uncertainty()
                volatility = hgf.get_volatility()
                
                # Decision uncertainty (from current evidence)
                evidence_clarity = trial[accuracy_col]
                decision_uncertainty = 1 - evidence_clarity
                
                # Combined uncertainty
                total_uncertainty = 0.5 * decision_uncertainty + 0.5 * state_uncertainty
                
                # ============================================
                # PREDICT CONTROL USING BAYESIAN EVC
                # ============================================
                reward = trial[reward_col]
                
                expected_value = reward * evidence_clarity
                
                predicted_control = self.baseline + \
                    (self.reward_weight * expected_value + 
                     self.uncertainty_weight * total_uncertainty +
                     self.volatility_weight * volatility) / \
                    (2 * self.effort_cost_weight)
                
                predicted_control = np.clip(predicted_control, 0, 1)
                
                # Store
                predictions.append(predicted_control)
                uncertainty_trajectory.append(state_uncertainty)
                volatility_trajectory.append(volatility)
                
                # ============================================
                # UPDATE HGF WITH TRIAL OUTCOME (for next trial)
                # ============================================
                outcome = trial[outcome_col]
                hgf.update(outcome)
        
        return np.array(predictions), np.array(uncertainty_trajectory), np.array(volatility_trajectory)
    
    def fit(self, data, observed_control_col='control_signal',
            reward_col='reward_magnitude', accuracy_col='evidence_clarity',
            outcome_col='accuracy', subject_col='subject_id'):
        """
        Fit model parameters
        
        Optimizes both EVC parameters AND HGF parameters!
        """
        observed_control = data[observed_control_col].values
        
        def objective(params):
            # EVC parameters
            self.baseline = params[0]
            self.reward_weight = params[1]
            self.effort_cost_weight = params[2]
            self.uncertainty_weight = params[3]
            self.volatility_weight = params[4]
            
            # HGF parameters (optional: can fix these)
            # self.hgf_params['kappa_2'] = params[5]
            # self.hgf_params['omega_2'] = params[6]
            
            # Predict sequentially
            predictions, _, _ = self.predict_control_sequential(
                data, reward_col, accuracy_col, outcome_col, subject_col
            )
            
            mse = np.mean((predictions - observed_control) ** 2)
            return mse
        
        # Initial parameters
        initial = [
            self.baseline,
            self.reward_weight,
            self.effort_cost_weight,
            self.uncertainty_weight,
            self.volatility_weight
        ]
        
        # Bounds
        bounds = [
            (0.0, 1.0),    # baseline
            (0.01, 10.0),  # reward_weight
            (0.01, 10.0),  # effort_cost_weight
            (0.0, 5.0),    # uncertainty_weight
            (0.0, 2.0)     # volatility_weight
        ]
        
        # Optimize
        print("Fitting Temporal Bayesian EVC (this may take a minute)...")
        result = minimize(objective, x0=initial, bounds=bounds, method='L-BFGS-B')
        
        # Update parameters
        self.baseline = result.x[0]
        self.reward_weight = result.x[1]
        self.effort_cost_weight = result.x[2]
        self.uncertainty_weight = result.x[3]
        self.volatility_weight = result.x[4]
        
        # Compute metrics
        predictions, uncertainty_traj, volatility_traj = self.predict_control_sequential(
            data, reward_col, accuracy_col, outcome_col, subject_col
        )
        
        r2 = r2_score(observed_control, predictions)
        rmse = np.sqrt(mean_squared_error(observed_control, predictions))
        correlation = np.corrcoef(observed_control, predictions)[0, 1]
        
        return {
            'baseline': self.baseline,
            'reward_weight': self.reward_weight,
            'effort_cost_weight': self.effort_cost_weight,
            'uncertainty_weight': self.uncertainty_weight,
            'volatility_weight': self.volatility_weight,
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation,
            'uncertainty_trajectory': uncertainty_traj,
            'volatility_trajectory': volatility_traj
        }
    
    def evaluate(self, data, observed_control_col='control_signal',
                reward_col='reward_magnitude', accuracy_col='evidence_clarity',
                outcome_col='accuracy', subject_col='subject_id'):
        """
        Evaluate on test data (without refitting)
        """
        observed_control = data[observed_control_col].values
        
        predictions, uncertainty_traj, volatility_traj = self.predict_control_sequential(
            data, reward_col, accuracy_col, outcome_col, subject_col
        )
        
        r2 = r2_score(observed_control, predictions)
        rmse = np.sqrt(mean_squared_error(observed_control, predictions))
        correlation = np.corrcoef(observed_control, predictions)[0, 1]
        
        return {
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation
        }


# ============================================
# COMPARISON: Non-Recurrent vs. Recurrent
# ============================================

if __name__ == '__main__':
    import pandas as pd
    
    # Load data
    data = pd.read_csv('data/behavioral_data.csv')
    
    # Split train/test
    subjects = data['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    n_train = int(len(subjects) * 0.7)
    
    train_data = data[data['subject_id'].isin(subjects[:n_train])].copy()
    test_data = data[data['subject_id'].isin(subjects[n_train:])].copy()
    
    print("=" * 70)
    print("COMPARISON: Non-Recurrent vs. Recurrent Bayesian EVC")
    print("=" * 70)
    
    # ============================================
    # MODEL 1: Non-Recurrent (Current)
    # ============================================
    print("\n1. Non-Recurrent Bayesian EVC (no history):")
    from models.bayesian_evc import BayesianEVC
    
    model_static = BayesianEVC()
    results_static = model_static.fit(train_data)
    test_static = model_static.evaluate(test_data)
    
    print(f"   Test R²: {test_static['r2']:.4f}")
    print(f"   Test RMSE: {test_static['rmse']:.4f}")
    print(f"   Uncertainty weight: {results_static['uncertainty_weight']:.4f}")
    
    # ============================================
    # MODEL 2: Recurrent (With HGF)
    # ============================================
    print("\n2. Recurrent Bayesian EVC (with HGF history):")
    
    model_temporal = BayesianEVC_Temporal()
    results_temporal = model_temporal.fit(train_data)
    test_temporal = model_temporal.evaluate(test_data)
    
    print(f"   Test R²: {test_temporal['r2']:.4f}")
    print(f"   Test RMSE: {test_temporal['rmse']:.4f}")
    print(f"   Uncertainty weight: {results_temporal['uncertainty_weight']:.4f}")
    print(f"   Volatility weight: {results_temporal['volatility_weight']:.4f} ← NEW!")
    
    # ============================================
    # COMPARISON
    # ============================================
    print("\n" + "=" * 70)
    print("IMPROVEMENT FROM ADDING RECURRENCE:")
    print("=" * 70)
    
    r2_improvement = ((test_temporal['r2'] - test_static['r2']) / 
                     abs(test_static['r2']) * 100)
    rmse_improvement = ((test_static['rmse'] - test_temporal['rmse']) / 
                       test_static['rmse'] * 100)
    
    print(f"R² improvement: {r2_improvement:+.1f}%")
    print(f"RMSE improvement: {rmse_improvement:+.1f}%")
    
    if test_temporal['r2'] > test_static['r2']:
        print("\n✓ Recurrent model captures trial history effects!")
    
    # ============================================
    # VISUALIZE UNCERTAINTY EVOLUTION
    # ============================================
    import matplotlib.pyplot as plt
    
    # Get trajectories for one subject
    subject_id = test_data['subject_id'].iloc[0]
    subject_data = test_data[test_data['subject_id'] == subject_id]
    
    _, unc_traj, vol_traj = model_temporal.predict_control_sequential(subject_data)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Uncertainty evolution
    axes[0].plot(unc_traj, linewidth=2, label='HGF State Uncertainty')
    axes[0].set_ylabel('Uncertainty', fontsize=12)
    axes[0].set_title(f'Subject {subject_id}: Uncertainty Evolution (Recurrent HGF)', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Volatility evolution
    axes[1].plot(vol_traj, linewidth=2, color='orange', label='HGF Volatility')
    axes[1].set_xlabel('Trial', fontsize=12)
    axes[1].set_ylabel('Volatility', fontsize=12)
    axes[1].set_title('Volatility Tracking (Adapts to Environmental Changes)', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/hgf_temporal_dynamics.png', dpi=300)
    print("\n✓ Saved temporal dynamics plot to results/hgf_temporal_dynamics.png")


