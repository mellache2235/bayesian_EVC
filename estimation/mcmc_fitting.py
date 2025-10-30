"""MCMC parameter estimation for Bayesian EVC models."""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def fit_bayesian_evc_mcmc(behavioral_data, participant_id=None, n_samples=1000, n_tune=500, random_seed=42):
    data = behavioral_data[behavioral_data['participant_id'] == participant_id].copy() if participant_id else behavioral_data.copy()
    
    reward = data['reward'].values.astype(float)
    dec_unc = data['decision_uncertainty'].values.astype(float)
    state_unc = data['state_uncertainty'].values.astype(float)
    obs_control = data['reaction_time'].values / data['reaction_time'].max()
    
    with pm.Model() as model:
        reward_sensitivity = pm.Normal('reward_sensitivity', mu=1.0, sigma=0.5)
        effort_cost = pm.HalfNormal('effort_cost', sigma=0.5)
        uncertainty_reduction_weight = pm.HalfNormal('uncertainty_reduction_weight', sigma=0.3)
        uncertainty_tolerance = pm.Beta('uncertainty_tolerance', alpha=2, beta=2)
        control_efficiency = pm.HalfNormal('control_efficiency', sigma=1.0)
        
        total_unc = dec_unc + state_unc
        reward_benefit = reward_sensitivity * reward * obs_control
        effort_cost_term = effort_cost * (obs_control ** 2)
        uncertainty_reduction = control_efficiency * obs_control * total_unc
        uncertainty_benefit = uncertainty_reduction_weight * uncertainty_reduction * (1 - uncertainty_tolerance)
        evc = reward_benefit - effort_cost_term + uncertainty_benefit
        
        pm.Normal('control_obs', mu=pm.math.sigmoid(evc), sigma=0.1, observed=obs_control)
        trace = pm.sample(draws=n_samples, tune=n_tune, random_seed=random_seed, return_inferencedata=True)
    
    return model, trace


def extract_posterior_samples(trace):
    posterior = trace.posterior
    params = ['reward_sensitivity', 'effort_cost', 'uncertainty_reduction_weight',
              'uncertainty_tolerance', 'control_efficiency']
    return {param: posterior[param].values.reshape(-1) for param in params if param in posterior}


def summarize_posterior(trace):
    return az.summary(trace)
