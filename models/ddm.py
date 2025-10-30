"""
Drift Diffusion Model (DDM) for confidence estimation.

This module implements a DDM-based approach to estimate decision confidence
and uncertainty from behavioral data.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict


class DriftDiffusionModel:
    """
    Drift Diffusion Model for confidence estimation.
    
    The DDM models decision-making as a noisy accumulation of evidence
    toward a decision boundary. Confidence is estimated from the strength
    of evidence relative to the boundary.
    """
    
    def __init__(
        self,
        boundary: float = 1.0,
        drift_rate: float = 0.5,
        noise: float = 1.0,
        non_decision_time: float = 0.2
    ):
        """
        Initialize DDM parameters.
        
        Parameters:
        -----------
        boundary : float
            Decision boundary (a)
        drift_rate : float
            Average drift rate (v)
        noise : float
            Noise parameter (s)
        non_decision_time : float
            Non-decision time (t)
        """
        self.boundary = boundary
        self.drift_rate = drift_rate
        self.noise = noise
        self.non_decision_time = non_decision_time
    
    def compute_confidence(
        self,
        reaction_time: np.ndarray,
        choice: np.ndarray,
        evidence_clarity: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute confidence and decision uncertainty from behavioral data.
        
        Parameters:
        -----------
        reaction_time : np.ndarray
            Reaction times in seconds
        choice : np.ndarray
            Choices (0 or 1)
        evidence_clarity : np.ndarray
            Evidence clarity (0-1 scale)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Confidence scores and decision uncertainty
        """
        # Adjust drift rate based on evidence clarity
        adjusted_drift = self.drift_rate * evidence_clarity
        
        # Estimate confidence from reaction time and drift
        # Faster RTs with high drift -> higher confidence
        # Confidence is proportional to drift rate and inversely related to RT
        rt_normalized = reaction_time / reaction_time.max()
        
        # Confidence: higher drift + faster RT -> higher confidence
        confidence = adjusted_drift * (1 - rt_normalized)
        confidence = np.clip(confidence, 0, 1)
        
        # Decision uncertainty: inverse of confidence
        decision_uncertainty = 1 - confidence
        
        return confidence, decision_uncertainty
    
    def simulate_rt(
        self,
        drift_rate: np.ndarray,
        n_samples: int = 1000
    ) -> np.ndarray:
        """
        Simulate reaction times from drift rates using DDM.
        
        Parameters:
        -----------
        drift_rate : np.ndarray
            Drift rates for each trial
        n_samples : int
            Number of samples per trial
        
        Returns:
        --------
        np.ndarray
            Simulated reaction times
        """
        rts = []
        
        for v in drift_rate:
            # Simplified DDM simulation
            # RT = boundary / drift_rate + noise
            rt = self.boundary / (v + 1e-6) + self.non_decision_time
            rt += np.random.normal(0, self.noise * 0.1)
            rt = np.maximum(rt, self.non_decision_time)
            rts.append(rt)
        
        return np.array(rts)
    
    def estimate_parameters(
        self,
        reaction_time: np.ndarray,
        choice: np.ndarray,
        correct: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate DDM parameters from data.
        
        Parameters:
        -----------
        reaction_time : np.ndarray
            Reaction times
        choice : np.ndarray
            Choices
        correct : np.ndarray
            Correctness
        
        Returns:
        --------
        Dict[str, float]
            Estimated parameters
        """
        # Simple heuristic estimates
        # Boundary: estimated from RT distribution
        rt_correct = reaction_time[correct == 1]
        rt_error = reaction_time[correct == 0]
        
        if len(rt_correct) > 0 and len(rt_error) > 0:
            # Estimate drift from accuracy and RT
            accuracy = correct.mean()
            avg_rt = reaction_time.mean()
            
            # Heuristic: higher accuracy -> higher drift
            estimated_drift = accuracy * 1.0
            
            # Boundary: RT is proportional to boundary/drift
            estimated_boundary = avg_rt * estimated_drift
        else:
            estimated_drift = self.drift_rate
            estimated_boundary = self.boundary
        
        return {
            'boundary': float(estimated_boundary),
            'drift_rate': float(estimated_drift),
            'noise': float(self.noise),
            'non_decision_time': float(self.non_decision_time)
        }
    
    def set_parameters(self, **kwargs):
        """Set DDM parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current DDM parameters."""
        return {
            'boundary': self.boundary,
            'drift_rate': self.drift_rate,
            'noise': self.noise,
            'non_decision_time': self.non_decision_time
        }

