"""
Bayesian Uncertainty Estimation for Cognitive Control Tasks

This module implements Bayesian inference methods to estimate two types of uncertainty
that influence cognitive control allocation:

1. DECISION UNCERTAINTY (Evidence-based uncertainty)
   - Uncertainty about what the current evidence means
   - Based on evidence clarity/quality
   - Example: "How certain am I that this stimulus is a 'left' vs 'right'?"
   - Measured via entropy and inverse of evidence clarity

2. STATE/RULE UNCERTAINTY (Belief-based uncertainty)
   - Uncertainty about which task rule/state is currently active
   - Based on Bayesian belief updating over trials
   - Example: "How certain am I that the current rule is 'respond to color' vs 'respond to shape'?"
   - Measured via entropy of belief distribution over possible states

THEORETICAL FOUNDATION:
-----------------------
Bayesian inference provides a principled way to:
1. Update beliefs about task states based on observations
2. Quantify uncertainty using information-theoretic measures (entropy)
3. Track how uncertainty changes over time

Bayes' Rule: P(state|observation) ∝ P(observation|state) × P(state)
- Prior: P(state) - beliefs before seeing observation
- Likelihood: P(observation|state) - probability of observation given state
- Posterior: P(state|observation) - updated beliefs after seeing observation

Entropy measures uncertainty:
- H(P) = -Σ P(x) × log₂(P(x))
- Maximum entropy = uniform distribution (maximum uncertainty)
- Minimum entropy = point mass (no uncertainty, complete certainty)
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
        # DECISION UNCERTAINTY CALCULATION
        # Decision uncertainty is the inverse of evidence clarity
        # Formula: U_decision = 1 - evidence_clarity
        #
        # Explanation:
        # - evidence_clarity: 0 = completely ambiguous, 1 = completely clear
        # - If evidence is clear (1.0) → uncertainty is low (0.0)
        # - If evidence is ambiguous (0.0) → uncertainty is high (1.0)
        # - This captures: "How uncertain am I about what this evidence means?"
        decision_uncertainty = 1 - evidence_clarity
        
        # CONFIDENCE CALCULATION
        # Confidence is a sigmoid transformation of evidence clarity
        # Formula: confidence = 1 / (1 + exp(-k × (clarity - 0.5)))
        #
        # Explanation:
        # - Uses logistic/sigmoid function (S-shaped curve)
        # - k = 5: steepness parameter (how sharp the transition)
        # - (clarity - 0.5): centers the function at 0.5
        # - Result: confidence smoothly transitions from 0 to 1
        # - Higher clarity → higher confidence (but non-linear)
        # - This captures: "How confident am I in my decision?"
        confidence = 1 / (1 + np.exp(-5 * (evidence_clarity - 0.5)))
        
        # ENTROPY-BASED UNCERTAINTY MEASURE
        # Entropy quantifies uncertainty using information theory
        # Formula: H(P) = -Σ P(x) × log₂(P(x))
        #
        # Explanation:
        # - Treat evidence_clarity as probability of being correct
        # - p_correct = evidence_clarity (probability of correct response)
        # - p_incorrect = 1 - evidence_clarity (probability of incorrect response)
        # - Entropy measures "surprise" or "information content"
        # - Maximum entropy = 1 bit (when p_correct = 0.5, maximum uncertainty)
        # - Minimum entropy = 0 bits (when p_correct = 0 or 1, no uncertainty)
        #
        # Why log base 2? Entropy is measured in bits (information theory)
        p_correct = evidence_clarity
        p_incorrect = 1 - p_correct
        
        # Avoid log(0) which is undefined
        # Clip probabilities to small epsilon values to ensure numerical stability
        p_correct = np.clip(p_correct, 1e-10, 1 - 1e-10)
        p_incorrect = np.clip(p_incorrect, 1e-10, 1 - 1e-10)
        
        # Calculate entropy: H = -[p_correct × log₂(p_correct) + p_incorrect × log₂(p_incorrect)]
        # This measures the uncertainty in bits
        # Example: if p_correct = 0.5, entropy = 1 bit (maximum uncertainty)
        #          if p_correct = 0.9, entropy ≈ 0.47 bits (low uncertainty)
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
        
        # COMPUTE LIKELIHOOD: P(observation|state)
        # Likelihood matrix specifies: "If state X is active, what's the probability of this observation?"
        #
        # For correct observation (observation == 1):
        #   likelihoods = likelihood_matrix
        #   Example: if state 0 has accuracy 0.8, P(correct|state0) = 0.8
        #
        # For incorrect observation (observation == 0):
        #   likelihoods = 1 - likelihood_matrix
        #   Example: if state 0 has accuracy 0.8, P(incorrect|state0) = 0.2
        if observation == 1:  # Correct response observed
            likelihoods = likelihood_matrix  # P(correct|state)
        else:  # Incorrect response observed
            likelihoods = 1 - likelihood_matrix  # P(incorrect|state) = 1 - P(correct|state)
        
        # BAYESIAN UPDATE: Compute posterior using Bayes' rule
        # Bayes' Rule: P(state|observation) ∝ P(observation|state) × P(state)
        #
        # Formula: posterior = likelihood × prior
        #   - likelihood: P(observation|state) - probability of observation given state
        #   - prior: self.state_beliefs - current beliefs P(state)
        #   - posterior: P(state|observation) - updated beliefs
        #
        # Note: This is proportional (∝), not equal (=)
        # We normalize below to make it a proper probability distribution
        posterior = likelihoods * self.state_beliefs
        
        # NORMALIZE: Convert from proportional to actual probabilities
        # Sum of probabilities must equal 1
        # Formula: P(state|observation) = (likelihood × prior) / Σ(likelihood × prior)
        # This ensures the posterior is a valid probability distribution
        posterior = posterior / (posterior.sum() + 1e-10)  # +1e-10 prevents division by zero
        
        # GRADUAL BELIEF UPDATING (Learning Rate)
        # Instead of jumping directly to posterior, we gradually update beliefs
        # Formula: new_beliefs = (1 - α) × old_beliefs + α × posterior
        #
        # Explanation:
        # - α (learning_rate): how much to trust the new observation
        #   - α = 1: fully trust new observation → jump to posterior
        #   - α = 0: ignore new observation → keep old beliefs
        #   - α = 0.1: slowly update → smooth learning
        #
        # Why gradual updating?
        # - Accounts for potential noise in observations
        # - Prevents overreacting to single observations
        # - Mimics how humans learn: gradually updating beliefs
        self.state_beliefs = (1 - self.learning_rate) * self.state_beliefs + \
                            self.learning_rate * posterior
        
        # ENSURE NORMALIZATION (safety check)
        # Due to floating-point arithmetic, probabilities might not sum exactly to 1
        # Renormalize to ensure they sum to 1 (required for valid probability distribution)
        self.state_beliefs = self.state_beliefs / self.state_beliefs.sum()
        
        return self.state_beliefs
    
    def estimate_state_uncertainty(self) -> Dict[str, float]:
        """
        Estimate state/rule uncertainty from current beliefs.
        
        Returns:
            Dictionary with state uncertainty measures
        """
        # STATE UNCERTAINTY CALCULATION
        # Quantify uncertainty about which state/rule is active using entropy
        
        # Entropy of belief distribution
        # Formula: H(P) = -Σ P(state) × log₂(P(state))
        #
        # Explanation:
        # - beliefs_clipped: probability distribution over states
        # - Entropy measures "spread" of beliefs
        #   - High entropy = beliefs spread evenly = high uncertainty
        #   - Low entropy = beliefs concentrated on one state = low uncertainty
        #
        # Example with 2 states:
        #   - Beliefs [0.5, 0.5]: entropy = 1 bit (maximum uncertainty)
        #   - Beliefs [0.9, 0.1]: entropy ≈ 0.47 bits (low uncertainty)
        #   - Beliefs [1.0, 0.0]: entropy = 0 bits (no uncertainty, complete certainty)
        beliefs_clipped = np.clip(self.state_beliefs, 1e-10, 1)  # Clip to avoid log(0)
        entropy = -np.sum(beliefs_clipped * np.log2(beliefs_clipped))
        
        # MAXIMUM ENTROPY (uniform distribution)
        # Maximum entropy occurs when beliefs are uniform (equal probability for all states)
        # Formula: H_max = log₂(n_states)
        #
        # Example: with 2 states, max entropy = log₂(2) = 1 bit
        #          with 4 states, max entropy = log₂(4) = 2 bits
        max_entropy = np.log2(self.n_states)
        
        # NORMALIZED UNCERTAINTY (0-1 scale)
        # Convert entropy to 0-1 scale for easier interpretation
        # Formula: normalized_uncertainty = entropy / max_entropy
        #
        # Explanation:
        # - 0 = no uncertainty (complete certainty about which state is active)
        # - 1 = maximum uncertainty (completely uncertain, all states equally likely)
        # - This normalized measure allows comparison across different numbers of states
        state_uncertainty = entropy / max_entropy if max_entropy > 0 else 0
        
        # STATE CONFIDENCE
        # Confidence is the inverse of uncertainty
        # Formula: confidence = 1 - uncertainty
        #
        # Explanation:
        # - High uncertainty → low confidence
        # - Low uncertainty → high confidence
        # - Ranges from 0 (no confidence) to 1 (complete confidence)
        state_confidence = 1 - state_uncertainty
        
        # MOST LIKELY STATE
        # Identify which state has the highest probability
        # This is the "best guess" about which state is currently active
        most_likely_state = np.argmax(self.state_beliefs)  # Index of maximum probability
        state_probability = self.state_beliefs[most_likely_state]  # Probability of that state
        
        # Example: if beliefs = [0.2, 0.8], then:
        #   - most_likely_state = 1 (second state)
        #   - state_probability = 0.8 (80% confident it's state 1)
        
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

