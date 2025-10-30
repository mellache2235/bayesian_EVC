"""
Example usage of Bayesian EVC models.

This script demonstrates how to use the Bayesian EVC framework
with dummy data.
"""

import numpy as np
import pandas as pd
from models.ddm import DriftDiffusionModel
from models.bayesian_evc import BayesianEVC
from models.traditional_evc import TraditionalEVC


def example_usage():
    """Example of using the Bayesian EVC models."""
    
    print("Example: Bayesian EVC Modeling")
    print("=" * 50)
    
    # Generate some dummy trial data
    n_trials = 50
    np.random.seed(42)
    
    # Trial parameters
    evidence_clarity = np.random.beta(2, 2, n_trials)
    rule_stability = np.random.beta(3, 2, n_trials)
    expected_reward = np.random.uniform(5, 15, n_trials)
    
    # Compute uncertainties
    decision_uncertainty = 1 - evidence_clarity
    state_uncertainty = 1 - rule_stability
    
    print(f"\nGenerated {n_trials} trials")
    print(f"Mean decision uncertainty: {decision_uncertainty.mean():.3f}")
    print(f"Mean state uncertainty: {state_uncertainty.mean():.3f}")
    
    # Initialize models
    print("\n1. Initializing models...")
    bayesian_model = BayesianEVC(
        reward_sensitivity=1.0,
        effort_cost=0.5,
        uncertainty_reduction_weight=0.3,
        uncertainty_tolerance=0.5,
        control_efficiency=1.0
    )
    
    traditional_model = TraditionalEVC(
        reward_sensitivity=1.0,
        effort_cost=0.5
    )
    
    # Compute optimal control
    print("\n2. Computing optimal control allocation...")
    bayesian_control = bayesian_model.optimal_control(
        expected_reward,
        decision_uncertainty,
        state_uncertainty
    )
    
    traditional_control = traditional_model.optimal_control(
        expected_reward
    )
    
    print(f"Bayesian EVC mean control: {bayesian_control.mean():.3f}")
    print(f"Traditional EVC mean control: {traditional_control.mean():.3f}")
    
    # Compute EVC
    print("\n3. Computing Expected Value of Control...")
    bayesian_evc, bayesian_components = bayesian_model.compute_evc(
        expected_reward,
        bayesian_control,
        decision_uncertainty,
        state_uncertainty
    )
    
    traditional_evc, traditional_components = traditional_model.compute_evc(
        expected_reward,
        traditional_control
    )
    
    print(f"Bayesian EVC mean: {bayesian_evc.mean():.3f}")
    print(f"Traditional EVC mean: {traditional_evc.mean():.3f}")
    print(f"\nBayesian EVC components:")
    print(f"  Reward benefit: {bayesian_components['reward_benefit'].mean():.3f}")
    print(f"  Effort cost: {bayesian_components['effort_cost'].mean():.3f}")
    print(f"  Uncertainty benefit: {bayesian_components['uncertainty_benefit'].mean():.3f}")
    
    # DDM confidence estimation
    print("\n4. Estimating confidence using DDM...")
    reaction_times = np.random.exponential(0.8, n_trials)
    choices = np.random.binomial(1, 0.7, n_trials)
    
    ddm = DriftDiffusionModel()
    confidence, ddm_uncertainty = ddm.compute_confidence(
        reaction_times,
        choices,
        evidence_clarity
    )
    
    print(f"Mean confidence: {confidence.mean():.3f}")
    print(f"Mean DDM uncertainty: {ddm_uncertainty.mean():.3f}")
    
    # Compare models
    print("\n5. Model comparison...")
    print("Bayesian EVC incorporates uncertainty reduction benefit:")
    print(f"  Improvement: {bayesian_evc.mean() - traditional_evc.mean():.3f}")
    
    # Show relationship between uncertainty and control
    print("\n6. Uncertainty-Control Relationship:")
    high_uncertainty = (decision_uncertainty + state_uncertainty) > 1.0
    print(f"High uncertainty trials: {high_uncertainty.sum()}")
    print(f"  Mean control (high uncertainty): {bayesian_control[high_uncertainty].mean():.3f}")
    print(f"  Mean control (low uncertainty): {bayesian_control[~high_uncertainty].mean():.3f}")
    
    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == '__main__':
    example_usage()

