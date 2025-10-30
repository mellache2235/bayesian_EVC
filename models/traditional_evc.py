"""
Traditional Expected Value of Control (EVC) Model.

This module implements a traditional EVC model without Bayesian uncertainty
components, for comparison purposes.
"""

import numpy as np
from typing import Dict, Tuple


class TraditionalEVC:
    """
    Traditional EVC model without uncertainty components.
    
    EVC = β_r * E[Reward|Control] - c * Control
    
    This model does not incorporate uncertainty reduction as a benefit.
    """
    
    def __init__(
        self,
        reward_sensitivity: float = 1.0,
        effort_cost: float = 0.5
    ):
        """
        Initialize traditional EVC parameters.
        
        Parameters:
        -----------
        reward_sensitivity : float
            Sensitivity to rewards (β_r)
        effort_cost : float
            Cost of exerting effort (c)
        """
        self.reward_sensitivity = reward_sensitivity
        self.effort_cost = effort_cost
    
    def compute_evc(
        self,
        expected_reward: np.ndarray,
        control_level: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute Expected Value of Control (traditional version).
        
        Parameters:
        -----------
        expected_reward : np.ndarray
            Expected reward for each trial
        control_level : np.ndarray
            Level of cognitive control allocated
        
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, np.ndarray]]
            EVC values and components breakdown
        """
        # Reward benefit
        reward_benefit = self.reward_sensitivity * expected_reward * control_level
        
        # Effort cost: quadratic cost of control
        effort_cost_term = self.effort_cost * (control_level ** 2)
        
        # Total EVC (no uncertainty component)
        evc = reward_benefit - effort_cost_term
        
        components = {
            'reward_benefit': reward_benefit,
            'effort_cost': effort_cost_term
        }
        
        return evc, components
    
    def optimal_control(
        self,
        expected_reward: np.ndarray
    ) -> np.ndarray:
        """
        Compute optimal control allocation.
        
        Control* = (β_r * E[Reward]) / (2*c)
        
        Parameters:
        -----------
        expected_reward : np.ndarray
            Expected reward
        
        Returns:
        --------
        np.ndarray
            Optimal control levels
        """
        optimal = (self.reward_sensitivity * expected_reward) / (2 * self.effort_cost)
        optimal = np.clip(optimal, 0, 1)
        
        return optimal
    
    def predict_behavior(
        self,
        expected_reward: np.ndarray,
        add_noise: bool = True,
        noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict behavioral outcomes.
        
        Parameters:
        -----------
        expected_reward : np.ndarray
            Expected reward
        add_noise : bool
            Whether to add noise to predictions
        noise_level : float
            Noise level if adding noise
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predicted control levels and performance
        """
        control = self.optimal_control(expected_reward)
        
        if add_noise:
            control += np.random.normal(0, noise_level, size=control.shape)
            control = np.clip(control, 0, 1)
        
        # Performance: higher control -> better performance
        performance = control * 0.8  # No uncertainty modulation
        performance = np.clip(performance, 0, 1)
        
        return control, performance
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current model parameters."""
        return {
            'reward_sensitivity': self.reward_sensitivity,
            'effort_cost': self.effort_cost
        }
    
    def set_parameters(self, **kwargs):
        """Set model parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

