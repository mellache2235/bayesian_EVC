"""Traditional EVC model without uncertainty components."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error


class TraditionalEVC:
    """
    Traditional Expected Value of Control model.
    
    EVC = β_r * Reward * Accuracy - β_e * Control^effort_exp
    
    Control allocation maximizes EVC:
    Control* = argmax_c [β_r * Reward * Accuracy(c) - β_e * c^effort_exp]
    """
    
    def __init__(self, reward_weight=1.0, effort_cost_weight=1.0, effort_exponent=2.0, baseline=0.5):
        """
        Initialize Traditional EVC model.
        
        Args:
            reward_weight: Weight for reward benefit (β_r)
            effort_cost_weight: Weight for effort cost (β_e)
            effort_exponent: Exponent for effort cost function
            baseline: Baseline control level (intercept)
        """
        self.reward_weight = reward_weight
        self.effort_cost_weight = effort_cost_weight
        self.effort_exponent = effort_exponent
        self.baseline = baseline
    
    def predict_control(self, data, reward_col='reward_magnitude', accuracy_col='evidence_clarity'):
        """
        Predict optimal control allocation.
        
        Args:
            data: DataFrame with trial data
            reward_col: Column name for reward magnitude
            accuracy_col: Column name for accuracy/evidence clarity
            
        Returns:
            Array of predicted control values
        """
        rewards = data[reward_col].values
        accuracy = data[accuracy_col].values
        
        # Expected value of control: reward * accuracy
        expected_value = rewards * accuracy
        
        # Optimal control: derivative of EVC w.r.t. control = 0
        # For quadratic cost (exponent=2): c* = (β_r * expected_value) / (2 * β_e)
        # For general exponent: approximate numerically or use closed form when available
        
        if self.effort_exponent == 2.0:
            # Closed form solution for quadratic cost
            control = self.baseline + (self.reward_weight * expected_value) / (2 * self.effort_cost_weight)
        else:
            # Numerical optimization for other exponents
            control = np.zeros(len(rewards))
            for i in range(len(rewards)):
                ev = expected_value[i]
                
                def neg_evc(c):
                    return -(self.reward_weight * ev * c - 
                            self.effort_cost_weight * (c ** self.effort_exponent))
                
                result = minimize(neg_evc, x0=0.5, bounds=[(0, 1)], method='L-BFGS-B')
                control[i] = self.baseline + result.x[0]
        
        # Clip to valid range
        control = np.clip(control, 0, 1)
        
        return control
    
    def fit(self, data, observed_control_col='control_signal', 
            reward_col='reward_magnitude', accuracy_col='evidence_clarity'):
        """
        Fit model parameters to observed data.
        
        Args:
            data: DataFrame with trial data
            observed_control_col: Column with observed control allocation
            reward_col: Column with reward magnitudes
            accuracy_col: Column with accuracy/evidence clarity
            
        Returns:
            Dictionary with fitted parameters and performance metrics
        """
        observed_control = data[observed_control_col].values
        rewards = data[reward_col].values
        accuracy = data[accuracy_col].values
        
        # Objective: minimize squared error between predicted and observed control
        def objective(params):
            self.baseline = params[0]
            self.reward_weight = params[1]
            self.effort_cost_weight = params[2]
            self.effort_exponent = params[3]
            
            predicted = self.predict_control(data, reward_col, accuracy_col)
            mse = np.mean((predicted - observed_control) ** 2)
            return mse
        
        # Optimize parameters
        initial_params = [self.baseline, self.reward_weight, self.effort_cost_weight, self.effort_exponent]
        bounds = [(0.0, 1.0), (0.01, 10.0), (0.01, 10.0), (1.0, 3.0)]
        
        result = minimize(objective, x0=initial_params, bounds=bounds, method='L-BFGS-B')
        
        # Update parameters
        self.baseline = result.x[0]
        self.reward_weight = result.x[1]
        self.effort_cost_weight = result.x[2]
        self.effort_exponent = result.x[3]
        
        # Compute performance metrics
        predicted = self.predict_control(data, reward_col, accuracy_col)
        r2 = r2_score(observed_control, predicted)
        rmse = np.sqrt(mean_squared_error(observed_control, predicted))
        
        return {
            'baseline': self.baseline,
            'reward_weight': self.reward_weight,
            'effort_cost_weight': self.effort_cost_weight,
            'effort_exponent': self.effort_exponent,
            'r2': r2,
            'rmse': rmse
        }
    
    def evaluate(self, data, observed_control_col='control_signal',
                reward_col='reward_magnitude', accuracy_col='evidence_clarity'):
        """
        Evaluate model on data (without fitting).
        
        Args:
            data: DataFrame with trial data
            observed_control_col: Column with observed control
            reward_col: Column with reward magnitudes
            accuracy_col: Column with accuracy/evidence clarity
            
        Returns:
            Dictionary with performance metrics
        """
        observed_control = data[observed_control_col].values
        predicted = self.predict_control(data, reward_col, accuracy_col)
        
        r2 = r2_score(observed_control, predicted)
        rmse = np.sqrt(mean_squared_error(observed_control, predicted))
        correlation = np.corrcoef(observed_control, predicted)[0, 1]
        
        return {
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation
        }
