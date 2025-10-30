"""Bayesian Expected Value of Control model with uncertainty components."""

import numpy as np


class BayesianEVC:
    """EVC = β_r * E[Reward|Control] - c * Control² + λ * UncertaintyReduction"""
    
    def __init__(self, reward_sensitivity=1.0, effort_cost=0.5, uncertainty_reduction_weight=0.3,
                 uncertainty_tolerance=0.5, control_efficiency=1.0):
        self.reward_sensitivity = reward_sensitivity
        self.effort_cost = effort_cost
        self.uncertainty_reduction_weight = uncertainty_reduction_weight
        self.uncertainty_tolerance = uncertainty_tolerance
        self.control_efficiency = control_efficiency
    
    def compute_evc(self, expected_reward, control_level, decision_uncertainty, state_uncertainty):
        total_unc = decision_uncertainty + state_uncertainty
        reward_benefit = self.reward_sensitivity * expected_reward * control_level
        effort_cost_term = self.effort_cost * (control_level ** 2)
        uncertainty_benefit = (self.uncertainty_reduction_weight * self.control_efficiency *
                              control_level * total_unc * (1 - self.uncertainty_tolerance))
        return (reward_benefit - effort_cost_term + uncertainty_benefit,
                {'reward_benefit': reward_benefit, 'effort_cost': effort_cost_term,
                 'uncertainty_benefit': uncertainty_benefit, 'total_uncertainty': total_unc})
    
    def optimal_control(self, expected_reward, decision_uncertainty, state_uncertainty):
        total_unc = decision_uncertainty + state_uncertainty
        numerator = (self.reward_sensitivity * expected_reward +
                    self.uncertainty_reduction_weight * self.control_efficiency *
                    total_unc * (1 - self.uncertainty_tolerance))
        return np.clip(numerator / (2 * self.effort_cost), 0, 1)
    
    def predict_behavior(self, expected_reward, decision_uncertainty, state_uncertainty, add_noise=False):
        control = self.optimal_control(expected_reward, decision_uncertainty, state_uncertainty)
        if add_noise:
            control = np.clip(control + np.random.normal(0, 0.1, size=control.shape), 0, 1)
        performance = np.clip(control * (1 - (decision_uncertainty + state_uncertainty) * 0.5), 0, 1)
        return control, performance
