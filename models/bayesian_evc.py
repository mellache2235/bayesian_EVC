"""Bayesian Expected Value of Control model with uncertainty components."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error


class BayesianEVC:
    """
    Bayesian Expected Value of Control model with uncertainty.
    
    EVC = β_r * Reward * Accuracy - β_e * Control^effort_exp + β_u * Uncertainty * Control
    
    The key innovation: uncertainty reduction is valuable and influences control allocation.
    """
    
    def __init__(self, reward_weight=1.0, effort_cost_weight=1.0, uncertainty_weight=0.5,
                 effort_exponent=2.0, baseline=0.5, n_states=2, learning_rate=0.1):
        """
        Initialize Bayesian EVC model.
        
        Args:
            reward_weight: Weight for reward benefit (β_r)
            effort_cost_weight: Weight for effort cost (β_e)
            uncertainty_weight: Weight for uncertainty reduction benefit (β_u)
            effort_exponent: Exponent for effort cost function
            baseline: Baseline control level (intercept)
            n_states: Number of hidden states (for Bayesian inference)
            learning_rate: Learning rate for belief updates
        """
        self.reward_weight = reward_weight
        self.effort_cost_weight = effort_cost_weight
        self.uncertainty_weight = uncertainty_weight
        self.effort_exponent = effort_exponent
        self.baseline = baseline
        self.n_states = n_states
        self.learning_rate = learning_rate
    
    def predict_control(self, data, reward_col='reward_magnitude', accuracy_col='evidence_clarity',
                       uncertainty_col='total_uncertainty', confidence_col='confidence'):
        """
        Predict optimal control allocation with uncertainty.
        
        Args:
            data: DataFrame with trial data
            reward_col: Column name for reward magnitude
            accuracy_col: Column name for accuracy/evidence clarity
            uncertainty_col: Column name for total uncertainty
            confidence_col: Column name for confidence
            
        Returns:
            Array of predicted control values
        """
        rewards = data[reward_col].values
        accuracy = data[accuracy_col].values
        uncertainty = data[uncertainty_col].values
        
        # Expected value of control
        expected_value = rewards * accuracy
        
        # Optimal control with uncertainty term
        # For quadratic cost: c* = (β_r * expected_value + β_u * uncertainty) / (2 * β_e)
        
        if self.effort_exponent == 2.0:
            # Closed form solution
            control = self.baseline + (self.reward_weight * expected_value + 
                      self.uncertainty_weight * uncertainty) / (2 * self.effort_cost_weight)
        else:
            # Numerical optimization
            control = np.zeros(len(rewards))
            for i in range(len(rewards)):
                ev = expected_value[i]
                unc = uncertainty[i]
                
                def neg_evc(c):
                    return -(self.reward_weight * ev * c + 
                            self.uncertainty_weight * unc * c -
                            self.effort_cost_weight * (c ** self.effort_exponent))
                
                result = minimize(neg_evc, x0=0.5, bounds=[(0, 1)], method='L-BFGS-B')
                control[i] = self.baseline + result.x[0]
        
        # Clip to valid range
        control = np.clip(control, 0, 1)
        
        return control
    
    def fit(self, data, observed_control_col='control_signal',
            reward_col='reward_magnitude', accuracy_col='evidence_clarity',
            uncertainty_col='total_uncertainty', confidence_col='confidence'):
        """
        Fit model parameters to observed data.
        
        Args:
            data: DataFrame with trial data
            observed_control_col: Column with observed control allocation
            reward_col: Column with reward magnitudes
            accuracy_col: Column with accuracy/evidence clarity
            uncertainty_col: Column with total uncertainty
            confidence_col: Column with confidence
            
        Returns:
            Dictionary with fitted parameters and performance metrics
        """
        observed_control = data[observed_control_col].values
        
        # Objective: minimize squared error
        def objective(params):
            self.baseline = params[0]
            self.reward_weight = params[1]
            self.effort_cost_weight = params[2]
            self.uncertainty_weight = params[3]
            self.effort_exponent = params[4]
            
            predicted = self.predict_control(data, reward_col, accuracy_col, 
                                            uncertainty_col, confidence_col)
            mse = np.mean((predicted - observed_control) ** 2)
            return mse
        
        # Optimize parameters
        initial_params = [self.baseline, self.reward_weight, self.effort_cost_weight, 
                         self.uncertainty_weight, self.effort_exponent]
        bounds = [(0.0, 1.0), (0.01, 10.0), (0.01, 10.0), (0.0, 5.0), (1.0, 3.0)]
        
        result = minimize(objective, x0=initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Update parameters
        self.baseline = result.x[0]
        self.reward_weight = result.x[1]
        self.effort_cost_weight = result.x[2]
        self.uncertainty_weight = result.x[3]
        self.effort_exponent = result.x[4]
        
        # Compute performance metrics
        predicted = self.predict_control(data, reward_col, accuracy_col,
                                        uncertainty_col, confidence_col)
        r2 = r2_score(observed_control, predicted)
        rmse = np.sqrt(mean_squared_error(observed_control, predicted))
        
        return {
            'baseline': self.baseline,
            'reward_weight': self.reward_weight,
            'effort_cost_weight': self.effort_cost_weight,
            'uncertainty_weight': self.uncertainty_weight,
            'effort_exponent': self.effort_exponent,
            'r2': r2,
            'rmse': rmse
        }
    
    def evaluate(self, data, observed_control_col='control_signal',
                reward_col='reward_magnitude', accuracy_col='evidence_clarity',
                uncertainty_col='total_uncertainty', confidence_col='confidence'):
        """
        Evaluate model on data (without fitting).
        
        Args:
            data: DataFrame with trial data
            observed_control_col: Column with observed control
            reward_col: Column with reward magnitudes
            accuracy_col: Column with accuracy/evidence clarity
            uncertainty_col: Column with total uncertainty
            confidence_col: Column with confidence
            
        Returns:
            Dictionary with performance metrics
        """
        observed_control = data[observed_control_col].values
        predicted = self.predict_control(data, reward_col, accuracy_col,
                                        uncertainty_col, confidence_col)
        
        r2 = r2_score(observed_control, predicted)
        rmse = np.sqrt(mean_squared_error(observed_control, predicted))
        correlation = np.corrcoef(observed_control, predicted)[0, 1]
        
        return {
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation
        }
