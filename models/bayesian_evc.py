"""
Bayesian Expected Value of Control (EVC) Model

This module extends the traditional EVC model by incorporating uncertainty components.
This is the KEY INNOVATION for your PhD project.

⚠️ KEY DIFFERENCE FROM TRADITIONAL EVC:
---------------------------------------
Traditional EVC ASSUMES:
- People have perfect knowledge (no uncertainty)
- Decisions based ONLY on: Reward - Effort
- Formula: EVC = β_r × E[Reward|c] - c_e × c^α
- Uncertainty reduction is NOT valued

Bayesian EVC (THIS MODEL) EXTENDS by:
- Including uncertainty about task rules (state uncertainty)
- Including uncertainty about evidence (decision uncertainty)
- Adding uncertainty reduction AS A BENEFIT
- Formula: EVC = β_r × E[Reward|c] - c_e × c^α + λ × Uncertainty_Reduction(c)
- The +λ term is the KEY ADDITION

🎯 THE INNOVATION:
------------------
Unlike traditional EVC, this model assumes people VALUE reducing uncertainty,
not just maximizing reward. Uncertainty reduction provides intrinsic benefit.

This is the CORE HYPOTHESIS you're testing in your PhD:
"Do people invest cognitive control to reduce uncertainty, beyond just getting rewards?"

THEORETICAL FOUNDATION:
-----------------------
The Bayesian EVC framework extends traditional EVC by recognizing that:
1. People are uncertain about task rules/states (state uncertainty)
2. People are uncertain about evidence quality (decision uncertainty)
3. Cognitive control can reduce both types of uncertainty
4. People value uncertainty reduction AS A BENEFIT (not just reward)

This is based on the idea that uncertainty is aversive, and reducing uncertainty
provides intrinsic value beyond just increasing reward probability.

MATHEMATICAL FORMULA:
---------------------
Bayesian_EVC(c) = β_r × Confidence × E[Reward|c] - c_e × c^α + λ × Uncertainty_Reduction(c)

Where:
  c = control level (0-1)
  β_r = reward_weight (sensitivity to reward)
  Confidence = confidence in being correct (inverse of uncertainty)
  E[Reward|c] = expected reward given control
  c_e = effort_cost_weight (cost of effort)
  α = effort_exponent (typically 2)
  λ = uncertainty_weight (how much uncertainty reduction matters)
  Uncertainty_Reduction(c) = how much uncertainty is reduced by deploying control

Expanded form:
Bayesian_EVC(c) = β_r × E[Reward|c] × c 
                - c_e × c^α 
                + λ × η × c × U_total × (1 - τ)

Where:
  η = control_efficiency (how well control reduces uncertainty)
  U_total = total_uncertainty = decision_uncertainty + state_uncertainty
  τ = uncertainty_tolerance (how comfortable with uncertainty)

KEY CONCEPTS:
-------------
1. Reward Benefit: Similar to traditional EVC, but modulated by confidence
   - Higher confidence → reward is more valuable
   - Benefit = reward_weight × expected_reward × control_level
   
2. Effort Cost: Same as traditional EVC
   - Cost = effort_cost_weight × control_level^effort_exponent
   
3. Uncertainty Reduction Benefit: NEW COMPONENT
   - Control reduces uncertainty
   - People value this reduction
   - Benefit = uncertainty_weight × control_efficiency × control_level × total_uncertainty × (1 - uncertainty_tolerance)
   - Higher uncertainty → more benefit from reducing it
   - Higher uncertainty_tolerance → less bothered by uncertainty → less benefit
   
4. Optimal Control: Balances all three components
   - When uncertainty is high → invest more control (to reduce it)
   - When reward is high → invest more control (to get it)
   - When effort cost is high → invest less control
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import pandas as pd


class BayesianEVC:
    """
    Bayesian EVC model: EVC = β_r × E[Reward|c] - c_e × c^α + λ × Uncertainty_Reduction(c)
    
    ⚠️ THIS IS THE KEY MODEL FOR YOUR PHD PROJECT
    
    This model extends traditional EVC by adding uncertainty reduction as a benefit.
    
    📊 SIDE-BY-SIDE COMPARISON:
    ---------------------------
    
    Traditional EVC:
        EVC = β_r × E[Reward|c] × c - c_e × c^α
        Components: (1) Reward benefit, (2) Effort cost
        Assumption: People only care about reward and effort
    
    Bayesian EVC (THIS MODEL):
        EVC = β_r × E[Reward|c] × c - c_e × c^α + λ × η × c × U_total × (1 - τ)
        Components: (1) Reward benefit, (2) Effort cost, (3) Uncertainty reduction benefit
        Assumption: People ALSO value reducing uncertainty
    
    🔑 THE KEY PARAMETER: uncertainty_weight (λ)
    ---------------------------------------------
    This parameter answers: "How much does uncertainty reduction matter?"
    - If λ = 0 → Traditional EVC (uncertainty doesn't matter)
    - If λ > 0 → Uncertainty reduction IS valued (supports your hypothesis)
    - Higher λ → People strongly value reducing uncertainty
    
    This is what you're trying to ESTIMATE and TEST in your PhD project!
    
    Parameters:
    -----------
    reward_weight : float (default: 1.0)
        β_r - Weight for reward benefits
        Same interpretation as traditional EVC
        
    effort_cost_weight : float (default: 1.0)
        c_e - Weight for effort costs
        Same interpretation as traditional EVC
        
    uncertainty_weight : float (default: 0.5)
        λ - Weight for uncertainty reduction benefit
        KEY PARAMETER: How much does uncertainty reduction matter?
        Higher values mean people strongly value reducing uncertainty
        Lower values mean uncertainty reduction is less important
        Interpretation: "How much do I dislike uncertainty?"
        
    effort_exponent : float (default: 2.0)
        α - Exponent for effort cost function (typically 2.0)
        
    n_states : int (default: 2)
        Number of possible task states/rules
        Used for Bayesian uncertainty estimation
        
    learning_rate : float (default: 0.1)
        Rate of Bayesian belief updating
        How quickly beliefs change with new evidence
        Higher = faster learning, lower = more stable beliefs
        
    uncertainty_tolerance : float (default: 0.5)
        τ - Individual tolerance for uncertainty (0-1)
        Higher = more comfortable with uncertainty
        Lower = more averse to uncertainty
        Affects how much benefit uncertainty reduction provides
        
    control_efficiency : float (default: 1.0)
        η - How efficiently control reduces uncertainty
        Higher = control is more effective at reducing uncertainty
        Lower = control is less effective
    """
    
    def __init__(
        self,
        reward_weight: float = 1.0,
        effort_cost_weight: float = 1.0,
        uncertainty_weight: float = 0.5,
        effort_exponent: float = 2.0,
        n_states: int = 2,
        learning_rate: float = 0.1,
        uncertainty_tolerance: float = 0.5,
        control_efficiency: float = 1.0
    ):
        """
        Initialize Bayesian EVC model.
        
        Args:
            reward_weight: β_r - Reward sensitivity
            effort_cost_weight: c_e - Effort cost scaling
            uncertainty_weight: λ - Uncertainty reduction benefit weight (KEY PARAMETER)
            effort_exponent: α - Effort cost exponent
            n_states: Number of task states for Bayesian updating
            learning_rate: Rate of belief updating
            uncertainty_tolerance: τ - Tolerance for uncertainty
            control_efficiency: η - Efficiency of control at reducing uncertainty
        """
        # Store all parameters
        self.reward_weight = reward_weight
        self.effort_cost_weight = effort_cost_weight
        self.uncertainty_weight = uncertainty_weight  # λ - KEY PARAMETER
        self.effort_exponent = effort_exponent
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.uncertainty_tolerance = uncertainty_tolerance  # τ
        self.control_efficiency = control_efficiency  # η
    
    def compute_evc(
        self,
        expected_reward: float,
        control_level: float,
        decision_uncertainty: float,
        state_uncertainty: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Bayesian Expected Value of Control.
        
        Formula: EVC(c) = β_r × E[Reward|c] × c 
                        - c_e × c^α 
                        + λ × η × c × U_total × (1 - τ)
        
        Args:
            expected_reward: E[Reward|c] - Expected reward given control
            control_level: c - Level of cognitive control (0-1)
            decision_uncertainty: Uncertainty about evidence quality (0-1)
            state_uncertainty: Uncertainty about task rules/states (0-1)
            
        Returns:
            Tuple of:
            - EVC value (float)
            - Dictionary with component breakdown:
              {
                  'reward_benefit': reward component,
                  'effort_cost': effort cost component,
                  'uncertainty_benefit': uncertainty reduction benefit,
                  'total_uncertainty': combined uncertainty
              }
        
        Example:
        --------
        >>> model = BayesianEVC(uncertainty_weight=0.5)
        >>> evc, comp = model.compute_evc(
        ...     expected_reward=10.0,
        ...     control_level=0.5,
        ...     decision_uncertainty=0.3,
        ...     state_uncertainty=0.4
        ... )
        >>> print(f"EVC: {evc:.3f}")
        >>> print(f"Uncertainty benefit: {comp['uncertainty_benefit']:.3f}")
        """
        # TOTAL UNCERTAINTY
        # Combine decision uncertainty (from evidence) and state uncertainty (from rules)
        # U_total = U_decision + U_state
        #
        # Explanation:
        # - Decision uncertainty: How uncertain am I about what the evidence means?
        # - State uncertainty: How uncertain am I about which task rule is active?
        # - Total uncertainty: Combined uncertainty from both sources
        total_unc = decision_uncertainty + state_uncertainty
        
        # REWARD BENEFIT COMPONENT
        # Same as traditional EVC: benefit increases with reward and control
        # Formula: reward_benefit = β_r × E[Reward|c] × c
        #
        # Note: In a more sophisticated version, this could be modulated by confidence:
        # reward_benefit = β_r × Confidence × E[Reward|c] × c
        # But here we keep it simple
        reward_benefit = self.reward_weight * expected_reward * control_level
        
        # EFFORT COST COMPONENT
        # Same as traditional EVC: cost increases non-linearly with control
        # Formula: effort_cost = c_e × c^α
        effort_cost_term = self.effort_cost_weight * (control_level ** self.effort_exponent)
        
        # UNCERTAINTY REDUCTION BENEFIT (NEW COMPONENT)
        # This is the KEY INNOVATION: people value reducing uncertainty
        # Formula: uncertainty_benefit = λ × η × c × U_total × (1 - τ)
        #
        # Explanation of each term:
        # - λ (uncertainty_weight): How much uncertainty reduction matters
        #   Higher = uncertainty reduction is more valuable
        # - η (control_efficiency): How well control reduces uncertainty
        #   Higher = control is more effective
        # - c (control_level): More control → more uncertainty reduction
        # - U_total (total_uncertainty): Higher uncertainty → more to reduce → more benefit
        # - (1 - τ) (uncertainty_tolerance): Lower tolerance → more bothered by uncertainty
        #   Lower tolerance → (1 - τ) closer to 1 → more benefit from reducing uncertainty
        #
        # Example: If uncertainty_weight=0.5, control_efficiency=1.0, control=0.5,
        #          total_uncertainty=0.7, uncertainty_tolerance=0.3:
        #          benefit = 0.5 × 1.0 × 0.5 × 0.7 × (1 - 0.3) = 0.5 × 0.35 = 0.175
        uncertainty_benefit = (
            self.uncertainty_weight *           # λ: weight for uncertainty reduction
            self.control_efficiency *            # η: efficiency of control
            control_level *                      # c: control level
            total_unc *                          # U_total: total uncertainty
            (1 - self.uncertainty_tolerance)     # (1 - τ): inverse of tolerance
        )
        
        # TOTAL EVC
        # EVC = Reward Benefit - Effort Cost + Uncertainty Reduction Benefit
        # Note: Uncertainty reduction is ADDED (it's a benefit, not a cost)
        evc = reward_benefit - effort_cost_term + uncertainty_benefit
        
        return evc, {
            'reward_benefit': reward_benefit,
            'effort_cost': effort_cost_term,
            'uncertainty_benefit': uncertainty_benefit,
            'total_uncertainty': total_unc
        }
    
    def optimal_control(
        self,
        expected_reward: float,
        decision_uncertainty: float,
        state_uncertainty: float
    ) -> float:
        """
        Calculate optimal control level that maximizes Bayesian EVC.
        
        Mathematical derivation:
        -----------------------
        EVC(c) = β_r × E[Reward] × c - c_e × c^α + λ × η × c × U_total × (1 - τ)
        
        For optimal control, take derivative and set to zero:
        dEVC/dc = β_r × E[Reward] - c_e × α × c^(α-1) + λ × η × U_total × (1 - τ) = 0
        
        For α = 2 (quadratic cost):
        dEVC/dc = β_r × E[Reward] - 2 × c_e × c + λ × η × U_total × (1 - τ) = 0
        
        Solving for c:
        c* = (β_r × E[Reward] + λ × η × U_total × (1 - τ)) / (2 × c_e)
        
        Args:
            expected_reward: Expected reward magnitude
            decision_uncertainty: Uncertainty about evidence (0-1)
            state_uncertainty: Uncertainty about task rules (0-1)
            
        Returns:
            Optimal control level (0-1) that maximizes EVC
        
        Example:
        --------
        >>> model = BayesianEVC(uncertainty_weight=0.5)
        >>> optimal = model.optimal_control(
        ...     expected_reward=10.0,
        ...     decision_uncertainty=0.3,
        ...     state_uncertainty=0.4
        ... )
        >>> print(f"Optimal control: {optimal:.3f}")
        """
        # Total uncertainty
        total_unc = decision_uncertainty + state_uncertainty
        
        # Optimal control formula
        # Numerator: reward benefit + uncertainty reduction benefit
        #   β_r × E[Reward]: benefit from reward
        #   λ × η × U_total × (1 - τ): benefit from uncertainty reduction
        # Denominator: effort cost scaling
        #   2 × c_e: accounts for quadratic effort cost (α = 2)
        #
        # Interpretation:
        # - Higher reward → more control
        # - Higher uncertainty → more control (to reduce it)
        # - Higher uncertainty_weight → more control (uncertainty reduction matters more)
        # - Higher uncertainty_tolerance → less control (less bothered by uncertainty)
        # - Higher effort_cost → less control (effort is more expensive)
        numerator = (
            self.reward_weight * expected_reward +                    # Reward benefit
            self.uncertainty_weight * self.control_efficiency *       # Uncertainty benefit scaling
            total_unc * (1 - self.uncertainty_tolerance)               # Uncertainty benefit magnitude
        )
        
        # Denominator: effort cost scaling (for quadratic cost with α=2)
        denominator = 2 * self.effort_cost_weight
        
        # Optimal control
        optimal = numerator / denominator
        
        # Clip to valid range [0, 1]
        return np.clip(optimal, 0, 1)
    
    def predict_control(
        self,
        data: pd.DataFrame,
        reward_col: str = 'reward_magnitude',
        accuracy_col: str = 'evidence_clarity',
        uncertainty_col: str = 'total_uncertainty',
        confidence_col: str = 'confidence'
    ) -> np.ndarray:
        """
        Predict optimal control levels for a dataset.
        
        For each trial:
        1. Extract expected reward and uncertainty
        2. Compute optimal control = argmax_c Bayesian_EVC(c)
        
        Args:
            data: DataFrame with trial data
            reward_col: Column name for reward magnitude
            accuracy_col: Column name for accuracy/probability
            uncertainty_col: Column name for total uncertainty
            confidence_col: Column name for confidence (may not be used)
            
        Returns:
            Array of predicted optimal control levels
        """
        predictions = []
        
        for _, row in data.iterrows():
            # Get reward magnitude
            reward = row[reward_col]
            
            # Estimate expected reward
            # expected_reward = reward × P(success)
            expected_reward = reward * row[accuracy_col]
            
            # Get uncertainty measures
            # If total_uncertainty column exists, use it
            # Otherwise, estimate from decision and state uncertainty
            if uncertainty_col in row:
                total_unc = row[uncertainty_col]
                # Split equally between decision and state uncertainty
                # (This is a simplification - in practice, you'd have separate columns)
                decision_unc = total_unc / 2
                state_unc = total_unc / 2
            else:
                # Fallback: estimate from available columns
                decision_unc = 1 - row.get('evidence_clarity', 0.5)
                state_unc = row.get('state_uncertainty', 0.5)
            
            # Compute optimal control
            optimal_c = self.optimal_control(
                expected_reward,
                decision_unc,
                state_unc
            )
            predictions.append(optimal_c)
        
        return np.array(predictions)
    
    def fit(
        self,
        data: pd.DataFrame,
        observed_control_col: str = 'control_signal',
        reward_col: str = 'reward_magnitude',
        accuracy_col: str = 'evidence_clarity',
        uncertainty_col: str = 'total_uncertainty',
        confidence_col: str = 'confidence'
    ) -> Dict[str, float]:
        """
        Fit model parameters to observed control allocation data.
        
        This optimizes:
        - reward_weight (β_r)
        - effort_cost_weight (c_e)
        - uncertainty_weight (λ) ← KEY PARAMETER TO ESTIMATE
        
        Args:
            data: DataFrame with observed behavioral data
            observed_control_col: Column name for observed control signals
            reward_col: Column name for reward magnitudes
            accuracy_col: Column name for accuracy/probability
            uncertainty_col: Column name for total uncertainty
            confidence_col: Column name for confidence (optional)
            
        Returns:
            Dictionary with fitted parameters and performance metrics:
            {
                'reward_weight': fitted β_r,
                'effort_cost_weight': fitted c_e,
                'uncertainty_weight': fitted λ ← KEY RESULT,
                'effort_exponent': α (fixed),
                'r2': R-squared,
                'rmse': Root mean squared error,
                'correlation': Pearson correlation
            }
        """
        # Extract observed control
        observed_control = data[observed_control_col].values
        
        # Extract predictors
        rewards = data[reward_col].values
        accuracies = data[accuracy_col].values
        
        # Extract uncertainty
        if uncertainty_col in data.columns:
            total_uncertainties = data[uncertainty_col].values
        else:
            # Estimate uncertainty if not available
            total_uncertainties = 1 - accuracies
        
        # Split uncertainty equally between decision and state
        # (This is a simplification - ideally you'd have separate columns)
        decision_uncs = total_uncertainties / 2
        state_uncs = total_uncertainties / 2
        
        # Define objective function
        def objective(params):
            """
            Objective: minimize prediction error.
            
            Args:
                params: [reward_weight, effort_cost_weight, uncertainty_weight]
            
            Returns:
                Mean squared error
            """
            reward_w, effort_w, uncertainty_w = params
            
            # Temporarily update parameters
            old_reward_w = self.reward_weight
            old_effort_w = self.effort_cost_weight
            old_uncertainty_w = self.uncertainty_weight
            
            self.reward_weight = reward_w
            self.effort_cost_weight = effort_w
            self.uncertainty_weight = uncertainty_w
            
            # Predict control for all trials
            predictions = []
            for reward, accuracy, dec_unc, state_unc in zip(
                rewards, accuracies, decision_uncs, state_uncs
            ):
                expected_reward = reward * accuracy
                pred_control = self.optimal_control(
                    expected_reward,
                    dec_unc,
                    state_unc
                )
                predictions.append(pred_control)
            
            predictions = np.array(predictions)
            
            # Compute mean squared error
            mse = np.mean((observed_control - predictions) ** 2)
            
            # Restore parameters
            self.reward_weight = old_reward_w
            self.effort_cost_weight = old_effort_w
            self.uncertainty_weight = old_uncertainty_w
            
            return mse
        
        # Initial parameter guesses
        initial_params = [
            self.reward_weight,
            self.effort_cost_weight,
            self.uncertainty_weight
        ]
        
        # Bounds: all parameters must be positive
        bounds = [(0.1, 10.0), (0.1, 10.0), (0.0, 10.0)]  # uncertainty_weight can be 0
        
        # Optimize
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Update model with fitted parameters
        self.reward_weight = result.x[0]
        self.effort_cost_weight = result.x[1]
        self.uncertainty_weight = result.x[2]  # ← KEY PARAMETER
        
        # Compute predictions with fitted parameters
        predictions = self.predict_control(
            data,
            reward_col,
            accuracy_col,
            uncertainty_col,
            confidence_col
        )
        
        # Compute performance metrics
        ss_res = np.sum((observed_control - predictions) ** 2)
        ss_tot = np.sum((observed_control - np.mean(observed_control)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((observed_control - predictions) ** 2))
        correlation = np.corrcoef(observed_control, predictions)[0, 1]
        
        return {
            'reward_weight': self.reward_weight,
            'effort_cost_weight': self.effort_cost_weight,
            'uncertainty_weight': self.uncertainty_weight,  # ← KEY RESULT
            'effort_exponent': self.effort_exponent,
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation
        }
    
    def evaluate(
        self,
        data: pd.DataFrame,
        observed_control_col: str = 'control_signal',
        reward_col: str = 'reward_magnitude',
        accuracy_col: str = 'evidence_clarity',
        uncertainty_col: str = 'total_uncertainty',
        confidence_col: str = 'confidence'
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            data: DataFrame with test data
            observed_control_col: Column name for observed control
            reward_col: Column name for rewards
            accuracy_col: Column name for accuracy
            uncertainty_col: Column name for uncertainty
            confidence_col: Column name for confidence
            
        Returns:
            Dictionary with performance metrics: {'r2', 'rmse', 'correlation'}
        """
        observed = data[observed_control_col].values
        predictions = self.predict_control(
            data,
            reward_col,
            accuracy_col,
            uncertainty_col,
            confidence_col
        )
        
        # Compute metrics
        ss_res = np.sum((observed - predictions) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        rmse = np.sqrt(np.mean((observed - predictions) ** 2))
        correlation = np.corrcoef(observed, predictions)[0, 1]
        
        return {
            'r2': r2,
            'rmse': rmse,
            'correlation': correlation
        }
    
    def predict_behavior(
        self,
        expected_reward: float,
        decision_uncertainty: float,
        state_uncertainty: float,
        add_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict behavioral outcomes (control allocation and performance).
        
        Args:
            expected_reward: Expected reward for each trial
            decision_uncertainty: Decision uncertainty for each trial
            state_uncertainty: State uncertainty for each trial
            add_noise: Whether to add random noise to predictions
            
        Returns:
            Tuple of (control_levels, performance_scores)
        """
        # Compute optimal control
        control = self.optimal_control(
            expected_reward,
            decision_uncertainty,
            state_uncertainty
        )
        
        # Optionally add noise
        if add_noise:
            noise = np.random.normal(0, 0.1, size=control.shape)
            control = np.clip(control + noise, 0, 1)
        
        # Performance is influenced by uncertainty
        # Higher uncertainty → worse performance even with same control
        # Formula: performance = control × (1 - uncertainty_penalty)
        total_unc = decision_uncertainty + state_uncertainty
        performance = np.clip(control * (1 - total_unc * 0.5), 0, 1)
        
        return control, performance


# Example usage
if __name__ == '__main__':
    print("Bayesian EVC Model - Example Usage\n")
    print("=" * 70)
    
    # Create model
    model = BayesianEVC(
        reward_weight=1.0,
        effort_cost_weight=0.5,
        uncertainty_weight=0.5,  # ← KEY PARAMETER
        uncertainty_tolerance=0.3,
        control_efficiency=1.0
    )
    
    # Example: Compare optimal control with vs without uncertainty
    print("\nExample: Effect of uncertainty on optimal control")
    print("-" * 70)
    expected_reward = 10.0
    
    # Low uncertainty
    control_low = model.optimal_control(expected_reward, 0.1, 0.1)
    evc_low, comp_low = model.compute_evc(expected_reward, control_low, 0.1, 0.1)
    
    # High uncertainty
    control_high = model.optimal_control(expected_reward, 0.5, 0.5)
    evc_high, comp_high = model.compute_evc(expected_reward, control_high, 0.5, 0.5)
    
    print(f"Low uncertainty (U=0.2):")
    print(f"  Optimal control: {control_low:.3f}")
    print(f"  EVC: {evc_low:.3f}")
    print(f"  Uncertainty benefit: {comp_low['uncertainty_benefit']:.3f}")
    
    print(f"\nHigh uncertainty (U=1.0):")
    print(f"  Optimal control: {control_high:.3f}")
    print(f"  EVC: {evc_high:.3f}")
    print(f"  Uncertainty benefit: {comp_high['uncertainty_benefit']:.3f}")
    
    print(f"\nDifference: High uncertainty → {control_high - control_low:.3f} more control")
