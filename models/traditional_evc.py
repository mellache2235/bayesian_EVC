"""Traditional EVC model without uncertainty components."""

import numpy as np


class TraditionalEVC:
    """EVC = β_r * E[Reward|Control] - c * Control²"""
    
    def __init__(self, reward_sensitivity=1.0, effort_cost=0.5):
        self.reward_sensitivity = reward_sensitivity
        self.effort_cost = effort_cost
    
    def compute_evc(self, expected_reward, control_level):
        reward_benefit = self.reward_sensitivity * expected_reward * control_level
        effort_cost_term = self.effort_cost * (control_level ** 2)
        return (reward_benefit - effort_cost_term,
                {'reward_benefit': reward_benefit, 'effort_cost': effort_cost_term})
    
    def optimal_control(self, expected_reward):
        return np.clip((self.reward_sensitivity * expected_reward) / (2 * self.effort_cost), 0, 1)
    
    def predict_behavior(self, expected_reward, add_noise=False):
        control = self.optimal_control(expected_reward)
        if add_noise:
            control = np.clip(control + np.random.normal(0, 0.1, size=control.shape), 0, 1)
        return control, np.clip(control * 0.8, 0, 1)
