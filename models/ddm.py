"""Drift Diffusion Model for confidence estimation."""

import numpy as np


class DriftDiffusionModel:
    """DDM for estimating decision confidence and uncertainty."""
    
    def __init__(self, boundary=1.0, drift_rate=0.5, noise=1.0, non_decision_time=0.2):
        self.boundary = boundary
        self.drift_rate = drift_rate
        self.noise = noise
        self.non_decision_time = non_decision_time
    
    def compute_confidence(self, reaction_time, choice, evidence_clarity):
        confidence = np.clip(self.drift_rate * evidence_clarity * (1 - reaction_time / reaction_time.max()), 0, 1)
        return confidence, 1 - confidence
    
    def estimate_parameters(self, reaction_time, choice, correct):
        if correct.sum() > 0:
            return {'boundary': float(reaction_time.mean() * correct.mean()),
                   'drift_rate': float(correct.mean()),
                   'noise': float(self.noise),
                   'non_decision_time': float(self.non_decision_time)}
        return {'boundary': float(self.boundary), 'drift_rate': float(self.drift_rate),
               'noise': float(self.noise), 'non_decision_time': float(self.non_decision_time)}
    
    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
