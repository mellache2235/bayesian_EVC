"""
Bayesian Expected Value of Control (EVC) Model.

This module implements a Bayesian EVC model that incorporates uncertainty
into the control allocation decision. The model explicitly incorporates
decision uncertainty and state uncertainty into the EVC calculation.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import norm


class BayesianEVC:
    """
    Bayesian Expected Value of Control model.
    
    The EVC represents the expected value of allocating cognitive control,
    which is computed as the difference between expected benefits and costs.
    In the Bayesian version, this explicitly incorporates uncertainty reduction
    as a benefit component.
    """
    
    def __init__(
        self,
        reward_sensitivity: float = 1.0,
        effort_cost: float = 0.5,
        uncertainty_reduction_weight: float = 0.3,
        uncertainty_tolerance: float = 0.5,
        control_efficiency: float = 1.0
    ):
        """
        Initialize Bayesian EVC parameters.
        
        Parameters:
        -----------
        reward_sensitivity : float
            Sensitivity to rewards (β_r)
        effort_cost : float
            Cost of exerting effort (c)
        uncertainty_reduction_weight : float
            Weight for uncertainty reduction benefit (λ)
        uncertainty_tolerance : float
            Individual tolerance for uncertainty (τ)
        control_efficiency : float
            Efficiency of control in reducing uncertainty (η)
        """
        self.reward_sensitivity = reward_sensitivity
        self.effort_cost = effort_cost
        self.uncertainty_reduction_weight = uncertainty_reduction_weight
        self.uncertainty_tolerance = uncertainty_tolerance
        self.control_efficiency = control_efficiency
    
    def compute_evc(
        self,
        expected_reward: np.ndarray,
        control_level: np.ndarray,
        decision_uncertainty: np.ndarray,
        state_uncertainty: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute Expected Value of Control with Bayesian uncertainty.
        
        EVC = β_r * E[Reward|Control] - c * Control + λ * UncertaintyReduction
        
        Parameters:
        -----------
        expected_reward : np.ndarray
            Expected reward for each trial
        control_level : np.ndarray
            Level of cognitive control allocated
        decision_uncertainty : np.ndarray
            Decision uncertainty (from DDM)
        state_uncertainty : np.ndarray
            State/rule uncertainty
        
        Returns:
        --------
        Tuple[np.ndarray, Dict[str, np.ndarray]]
            EVC values and components breakdown
        """
        # Reward benefit: scaled by control level
        # Higher control -> better performance -> higher expected reward
        reward_benefit = self.reward_sensitivity * expected_reward * control_level
        
        # Effort cost: quadratic cost of control
        effort_cost_term = self.effort_cost * (control_level ** 2)
        
        # Uncertainty reduction benefit: Bayesian component
        # Control reduces uncertainty, which has value
        total_uncertainty = decision_uncertainty + state_uncertainty
        
        # Uncertainty reduction: control reduces uncertainty with efficiency η
        uncertainty_reduction = (
            self.control_efficiency * control_level * total_uncertainty
        )
        
        # Uncertainty reduction benefit: weighted by individual tolerance
        # Higher uncertainty tolerance -> less value from reduction
        uncertainty_benefit = (
            self.uncertainty_reduction_weight *
            uncertainty_reduction *
            (1 - self.uncertainty_tolerance)
        )
        
        # Total EVC
        evc = reward_benefit - effort_cost_term + uncertainty_benefit
        
        components = {
            'reward_benefit': reward_benefit,
            'effort_cost': effort_cost_term,
            'uncertainty_reduction': uncertainty_reduction,
            'uncertainty_benefit': uncertainty_benefit,
            'total_uncertainty': total_uncertainty
        }
        
        return evc, components
    
    def optimal_control(
        self,
        expected_reward: np.ndarray,
        decision_uncertainty: np.ndarray,
        state_uncertainty: np.ndarray
    ) -> np.ndarray:
        """
        Compute optimal control allocation that maximizes EVC.
        
        By taking derivative of EVC w.r.t. control and setting to 0:
        ∂EVC/∂Control = β_r * E[Reward] - 2*c*Control + λ*η*TotalUncertainty = 0
        
        Solving: Control* = (β_r * E[Reward] + λ*η*TotalUncertainty) / (2*c)
        
        Parameters:
        -----------
        expected_reward : np.ndarray
            Expected reward
        decision_uncertainty : np.ndarray
            Decision uncertainty
        state_uncertainty : np.ndarray
            State uncertainty
        
        Returns:
        --------
        np.ndarray
            Optimal control levels
        """
        total_uncertainty = decision_uncertainty + state_uncertainty
        
        numerator = (
            self.reward_sensitivity * expected_reward +
            self.uncertainty_reduction_weight * self.control_efficiency *
            total_uncertainty * (1 - self.uncertainty_tolerance)
        )
        
        optimal = numerator / (2 * self.effort_cost)
        
        # Clip to [0, 1] range
        optimal = np.clip(optimal, 0, 1)
        
        return optimal
    
    def predict_behavior(
        self,
        expected_reward: np.ndarray,
        decision_uncertainty: np.ndarray,
        state_uncertainty: np.ndarray,
        add_noise: bool = True,
        noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict behavioral outcomes (control allocation and performance).
        
        Parameters:
        -----------
        expected_reward : np.ndarray
            Expected reward
        decision_uncertainty : np.ndarray
            Decision uncertainty
        state_uncertainty : np.ndarray
            State uncertainty
        add_noise : bool
            Whether to add noise to predictions
        noise_level : float
            Noise level if adding noise
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Predicted control levels and performance
        """
        # Optimal control
        control = self.optimal_control(
            expected_reward,
            decision_uncertainty,
            state_uncertainty
        )
        
        if add_noise:
            control += np.random.normal(0, noise_level, size=control.shape)
            control = np.clip(control, 0, 1)
        
        # Performance: higher control -> better performance
        # Reduced uncertainty also improves performance
        total_uncertainty = decision_uncertainty + state_uncertainty
        performance = control * (1 - total_uncertainty * 0.5)
        performance = np.clip(performance, 0, 1)
        
        return control, performance
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current model parameters."""
        return {
            'reward_sensitivity': self.reward_sensitivity,
            'effort_cost': self.effort_cost,
            'uncertainty_reduction_weight': self.uncertainty_reduction_weight,
            'uncertainty_tolerance': self.uncertainty_tolerance,
            'control_efficiency': self.control_efficiency
        }
    
    def set_parameters(self, **kwargs):
        """Set model parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

