"""
Bayesian uncertainty estimation for cognitive control tasks.

Implements two types of uncertainty:
1. Decision uncertainty: From trial-to-trial variability in evidence clarity
2. State/rule uncertainty: From beliefs about task rules and conditions
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Optional


class BayesianUncertaintyEstimator:
    """
    Estimates uncertainty using Bayesian inference.
    
    Combines:
    - Decision uncertainty (evidence-based)
    - State uncertainty (rule-based)
    """
    
    def __init__(self, n_states: int = 2, learning_rate: float = 0.1):
        """
        Initialize the uncertainty estimator.
        
        Args:
            n_states: Number of possible task states/rules
            learning_rate: Rate of belief updating
        """
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.state_beliefs = None
        self.reset_beliefs()
    
    def reset_beliefs(self):
        """Reset state beliefs to uniform distribution."""
        self.state_beliefs = np.ones(self.n_states) / self.n_states
    
    def estimate_decision_uncertainty(
        self, 
        evidence_clarity: float,
        drift_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Estimate decision uncertainty based on evidence clarity.
        
        Inspired by drift-diffusion models where uncertainty relates to
        the difficulty of accumulating evidence.
        
        Args:
            evidence_clarity: Clarity of evidence (0-1, higher = clearer)
            drift_rate: Optional drift rate from DDM-like process
            
        Returns:
            Dictionary with uncertainty measures
        """
        # Decision uncertainty is inverse of clarity
        decision_uncertainty = 1 - evidence_clarity
        
        # Confidence based on evidence clarity
        # Using a sigmoid-like transformation
        confidence = 1 / (1 + np.exp(-5 * (evidence_clarity - 0.5)))
        
        # Entropy-based uncertainty measure
        # Treat as probability of correct response
        p_correct = evidence_clarity
        p_incorrect = 1 - p_correct
        
        # Avoid log(0)
        p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)
        p_incorrect = np.clip(p_incorrect, 1e-10, 1 - 1e-10)
        
        entropy = -(p_correct * np.log2(p_correct) + 
                   p_incorrect * np.log2(p_incorrect))
        
        return {
            'decision_uncertainty': decision_uncertainty,
            'confidence': confidence,
            'entropy': entropy
        }
    
    def update_state_beliefs(
        self,
        observation: int,
        likelihood_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update beliefs about task state using Bayesian inference.
        
        Args:
            observation: Observed outcome (0 or 1 for correct/incorrect)
            likelihood_matrix: P(observation|state), shape (n_states,)
                             If None, uses default likelihood
            
        Returns:
            Updated state beliefs (posterior)
        """
        if likelihood_matrix is None:
            # Default: different states have different accuracy rates
            likelihood_matrix = np.array([0.8, 0.6]) if self.n_states == 2 else \
                               np.linspace(0.9, 0.5, self.n_states)
        
        # Likelihood of observation given each state
        if observation == 1:  # Correct
            likelihoods = likelihood_matrix
        else:  # Incorrect
            likelihoods = 1 - likelihood_matrix
        
        # Bayesian update: posterior ∝ likelihood × prior
        posterior = likelihoods * self.state_beliefs
        posterior = posterior / (posterior.sum() + 1e-10)  # Normalize
        
        # Gradual belief updating (learning rate)
        self.state_beliefs = (1 - self.learning_rate) * self.state_beliefs + \
                            self.learning_rate * posterior
        
        # Ensure normalization
        self.state_beliefs = self.state_beliefs / self.state_beliefs.sum()
        
        return self.state_beliefs
    
    def estimate_state_uncertainty(self) -> Dict[str, float]:
        """
        Estimate state/rule uncertainty from current beliefs.
        
        Returns:
            Dictionary with state uncertainty measures
        """
        # Entropy of belief distribution
        beliefs_clipped = np.clip(self.state_beliefs, 1e-10, 1)
        entropy = -np.sum(beliefs_clipped * np.log2(beliefs_clipped))
        
        # Maximum entropy (uniform distribution)
        max_entropy = np.log2(self.n_states)
        
        # Normalized uncertainty (0-1)
        state_uncertainty = entropy / max_entropy if max_entropy > 0 else 0
        
        # Confidence is inverse of uncertainty
        state_confidence = 1 - state_uncertainty
        
        # Most likely state
        most_likely_state = np.argmax(self.state_beliefs)
        state_probability = self.state_beliefs[most_likely_state]
        
        return {
            'state_uncertainty': state_uncertainty,
            'state_confidence': state_confidence,
            'entropy': entropy,
            'most_likely_state': most_likely_state,
            'state_probability': state_probability,
            'belief_distribution': self.state_beliefs.copy()
        }
    
    def estimate_combined_uncertainty(
        self,
        evidence_clarity: float,
        observation: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate combined uncertainty from both decision and state sources.
        
        Args:
            evidence_clarity: Clarity of current evidence
            observation: Optional outcome to update state beliefs
            
        Returns:
            Dictionary with all uncertainty measures
        """
        # Decision uncertainty
        decision_metrics = self.estimate_decision_uncertainty(evidence_clarity)
        
        # Update state beliefs if observation provided
        if observation is not None:
            self.update_state_beliefs(observation)
        
        # State uncertainty
        state_metrics = self.estimate_state_uncertainty()
        
        # Combined uncertainty (weighted average)
        combined_uncertainty = (
            0.5 * decision_metrics['decision_uncertainty'] +
            0.5 * state_metrics['state_uncertainty']
        )
        
        # Combined confidence
        combined_confidence = (
            0.5 * decision_metrics['confidence'] +
            0.5 * state_metrics['state_confidence']
        )
        
        return {
            **decision_metrics,
            **state_metrics,
            'combined_uncertainty': combined_uncertainty,
            'combined_confidence': combined_confidence
        }


class SequentialBayesianEstimator:
    """
    Estimates uncertainty sequentially across trials.
    
    Useful for analyzing experimental data where uncertainty evolves over time.
    """
    
    def __init__(self, n_states: int = 2, learning_rate: float = 0.1):
        """
        Initialize sequential estimator.
        
        Args:
            n_states: Number of possible task states
            learning_rate: Rate of belief updating
        """
        self.estimator = BayesianUncertaintyEstimator(n_states, learning_rate)
    
    def process_trial_sequence(
        self,
        trial_data: pd.DataFrame,
        evidence_col: str = 'evidence_clarity',
        outcome_col: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Process a sequence of trials to estimate uncertainty over time.
        
        Args:
            trial_data: DataFrame with trial-level data
            evidence_col: Column name for evidence clarity
            outcome_col: Column name for trial outcomes
            
        Returns:
            DataFrame with added uncertainty estimates
        """
        results = []
        
        # Reset beliefs at start
        self.estimator.reset_beliefs()
        
        for idx, row in trial_data.iterrows():
            evidence = row[evidence_col]
            outcome = row[outcome_col] if outcome_col in row else None
            
            # Estimate uncertainty
            uncertainty_metrics = self.estimator.estimate_combined_uncertainty(
                evidence_clarity=evidence,
                observation=outcome
            )
            
            # Combine with original data
            result = {**row.to_dict(), **uncertainty_metrics}
            results.append(result)
        
        return pd.DataFrame(results)
    
    def process_subject_data(
        self,
        data: pd.DataFrame,
        subject_col: str = 'subject_id',
        evidence_col: str = 'evidence_clarity',
        outcome_col: str = 'accuracy'
    ) -> pd.DataFrame:
        """
        Process data for multiple subjects, resetting beliefs for each.
        
        Args:
            data: DataFrame with multi-subject data
            subject_col: Column identifying subjects
            evidence_col: Column for evidence clarity
            outcome_col: Column for outcomes
            
        Returns:
            DataFrame with uncertainty estimates for all subjects
        """
        all_results = []
        
        for subject_id in data[subject_col].unique():
            subject_data = data[data[subject_col] == subject_id].copy()
            subject_results = self.process_trial_sequence(
                subject_data,
                evidence_col=evidence_col,
                outcome_col=outcome_col
            )
            all_results.append(subject_results)
        
        return pd.concat(all_results, ignore_index=True)


def main():
    """Demonstrate Bayesian uncertainty estimation."""
    print("Bayesian Uncertainty Estimation Demo\n")
    
    # Create estimator
    estimator = BayesianUncertaintyEstimator(n_states=2, learning_rate=0.2)
    
    # Simulate a sequence of trials
    print("Simulating trial sequence with varying evidence clarity...\n")
    
    evidence_sequence = [0.8, 0.7, 0.5, 0.4, 0.6, 0.8, 0.9]
    outcomes = [1, 1, 0, 0, 1, 1, 1]  # 1=correct, 0=incorrect
    
    for trial, (evidence, outcome) in enumerate(zip(evidence_sequence, outcomes), 1):
        metrics = estimator.estimate_combined_uncertainty(evidence, outcome)
        
        print(f"Trial {trial}:")
        print(f"  Evidence clarity: {evidence:.2f}")
        print(f"  Outcome: {'Correct' if outcome else 'Incorrect'}")
        print(f"  Decision uncertainty: {metrics['decision_uncertainty']:.3f}")
        print(f"  State uncertainty: {metrics['state_uncertainty']:.3f}")
        print(f"  Combined uncertainty: {metrics['combined_uncertainty']:.3f}")
        print(f"  Combined confidence: {metrics['combined_confidence']:.3f}")
        print(f"  State beliefs: {metrics['belief_distribution']}")
        print()


if __name__ == '__main__':
    main()

