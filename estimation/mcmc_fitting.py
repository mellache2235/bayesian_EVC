"""
MCMC parameter estimation for Bayesian EVC models.

This module uses PyMC to fit Bayesian EVC model parameters using
Markov Chain Monte Carlo sampling.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from typing import Dict, Tuple, Optional
from models.bayesian_evc import BayesianEVC
from models.ddm import DriftDiffusionModel


def fit_bayesian_evc_mcmc(
    behavioral_data: pd.DataFrame,
    participant_id: Optional[int] = None,
    n_samples: int = 1000,
    n_tune: int = 500,
    random_seed: int = 42
) -> Tuple[pm.Model, az.InferenceData]:
    """
    Fit Bayesian EVC model using MCMC.
    
    Parameters:
    -----------
    behavioral_data : pd.DataFrame
        Behavioral data with columns: reaction_time, choice, correct, reward,
        decision_uncertainty, state_uncertainty
    participant_id : Optional[int]
        If provided, fit model for specific participant
    n_samples : int
        Number of MCMC samples
    n_tune : int
        Number of tuning samples
    random_seed : int
        Random seed
    
    Returns:
    --------
    Tuple[pm.Model, az.InferenceData]
        Fitted PyMC model and inference data
    """
    # Filter data if participant_id provided
    if participant_id is not None:
        data = behavioral_data[behavioral_data['participant_id'] == participant_id].copy()
    else:
        data = behavioral_data.copy()
    
    # Prepare data
    n_trials = len(data)
    expected_reward = data['reward'].values.astype(float)
    decision_uncertainty = data['decision_uncertainty'].values.astype(float)
    state_uncertainty = data['state_uncertainty'].values.astype(float)
    observed_control = data['reaction_time'].values / data['reaction_time'].max()  # Proxy for control
    
    # Build PyMC model
    with pm.Model() as model:
        # Priors for Bayesian EVC parameters
        reward_sensitivity = pm.Normal('reward_sensitivity', mu=1.0, sigma=0.5)
        effort_cost = pm.HalfNormal('effort_cost', sigma=0.5)
        uncertainty_reduction_weight = pm.HalfNormal('uncertainty_reduction_weight', sigma=0.3)
        uncertainty_tolerance = pm.Beta('uncertainty_tolerance', alpha=2, beta=2)
        control_efficiency = pm.HalfNormal('control_efficiency', sigma=1.0)
        
        # Compute EVC components
        total_uncertainty = decision_uncertainty + state_uncertainty
        
        # Reward benefit
        reward_benefit = reward_sensitivity * expected_reward * observed_control
        
        # Effort cost
        effort_cost_term = effort_cost * (observed_control ** 2)
        
        # Uncertainty reduction benefit
        uncertainty_reduction = control_efficiency * observed_control * total_uncertainty
        uncertainty_benefit = (
            uncertainty_reduction_weight *
            uncertainty_reduction *
            (1 - uncertainty_tolerance)
        )
        
        # Expected EVC
        evc = reward_benefit - effort_cost_term + uncertainty_benefit
        
        # Likelihood: EVC should be positive (control is allocated when beneficial)
        # Model control allocation as a function of EVC
        evc_prob = pm.math.sigmoid(evc)  # Convert to probability
        
        # Observe control allocation (using RT as proxy)
        control_obs = pm.Normal(
            'control_obs',
            mu=evc_prob,
            sigma=0.1,
            observed=observed_control
        )
        
        # Sample
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            random_seed=random_seed,
            return_inferencedata=True
        )
    
    return model, trace


def extract_posterior_samples(
    trace: az.InferenceData,
    parameter_names: Optional[list] = None
) -> Dict[str, np.ndarray]:
    """
    Extract posterior samples from MCMC trace.
    
    Parameters:
    -----------
    trace : az.InferenceData
        ArviZ inference data
    parameter_names : Optional[list]
        List of parameter names to extract
    
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary of posterior samples
    """
    if parameter_names is None:
        parameter_names = [
            'reward_sensitivity',
            'effort_cost',
            'uncertainty_reduction_weight',
            'uncertainty_tolerance',
            'control_efficiency'
        ]
    
    posterior = trace.posterior
    
    samples = {}
    for param in parameter_names:
        if param in posterior:
            samples[param] = posterior[param].values.reshape(-1)
    
    return samples


def summarize_posterior(
    trace: az.InferenceData
) -> pd.DataFrame:
    """
    Summarize posterior distributions.
    
    Parameters:
    -----------
    trace : az.InferenceData
        ArviZ inference data
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    return az.summary(trace)


def fit_participants(
    behavioral_data: pd.DataFrame,
    n_samples: int = 1000,
    n_tune: int = 500
) -> Dict[int, Dict]:
    """
    Fit Bayesian EVC model for each participant.
    
    Parameters:
    -----------
    behavioral_data : pd.DataFrame
        Behavioral data
    n_samples : int
        Number of MCMC samples
    n_tune : int
        Number of tuning samples
    
    Returns:
    --------
    Dict[int, Dict]
        Dictionary mapping participant_id to fitted parameters
    """
    participants = behavioral_data['participant_id'].unique()
    results = {}
    
    for pid in participants:
        print(f"Fitting participant {pid}...")
        try:
            model, trace = fit_bayesian_evc_mcmc(
                behavioral_data,
                participant_id=pid,
                n_samples=n_samples,
                n_tune=n_tune
            )
            
            # Extract posterior means
            summary = summarize_posterior(trace)
            samples = extract_posterior_samples(trace)
            
            results[pid] = {
                'trace': trace,
                'summary': summary,
                'posterior_samples': samples,
                'posterior_means': {
                    k: np.mean(v) for k, v in samples.items()
                }
            }
        except Exception as e:
            print(f"Error fitting participant {pid}: {e}")
            results[pid] = None
    
    return results

