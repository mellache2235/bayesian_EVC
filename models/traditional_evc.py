"""
Traditional Expected Value of Control (EVC) Model

This module implements the classic EVC model from Shenhav et al. (2013) without 
uncertainty components. The model assumes people allocate cognitive control 
based on a simple cost-benefit analysis: maximize expected reward while minimizing effort.

THEORETICAL FOUNDATION:
-----------------------
The Expected Value of Control (EVC) framework posits that:
1. People estimate the expected value of investing cognitive control
2. Control allocation is optimized to maximize: EVC = Benefit - Cost
3. Benefit comes from increased probability of reward
4. Cost comes from the effort required to deploy control

MATHEMATICAL FORMULA:
---------------------
EVC(c) = β_r × E[Reward|c] - c_e × c^α

Where:
  c = control level (0-1, where 0=no control, 1=maximal control)
  β_r = reward_weight (sensitivity to reward, how much reward matters)
  E[Reward|c] = expected reward given control level c
               = P(success|c) × reward_magnitude
  c_e = effort_cost_weight (how costly effort is)
  α = effort_exponent (typically 2, makes effort cost quadratic)

The optimal control level maximizes EVC:
  c* = argmax_c EVC(c)
  
Taking derivative and setting to zero:
  dEVC/dc = β_r × (dE[Reward]/dc) - c_e × α × c^(α-1) = 0
  
If E[Reward] is linear in control: E[Reward|c] = reward × (baseline + control × gain)
Then: c* = clip((β_r × reward × gain) / (c_e × α), 0, 1)

KEY CONCEPTS:
-------------
1. Reward Benefit: How much additional reward you get from investing control
   - Higher control → better performance → higher probability of reward
   - Benefit = reward_weight × expected_reward × control_level
   
2. Effort Cost: The "cost" of deploying control (mental effort)
   - Effort increases non-linearly with control (quadratic: cost ∝ c²)
   - Cost = effort_cost_weight × control_level^effort_exponent
   - Quadratic cost means: doubling control quadruples effort cost
   
3. Optimal Control: The control level that maximizes EVC
   - Found by balancing benefit vs cost
   - When reward is high → invest more control
   - When effort cost is high → invest less control
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Optional
import pandas as pd


class TraditionalEVC:
    """
    Traditional EVC model: EVC = β_r × E[Reward|c] - c_e × c^α
    
    This is the baseline model WITHOUT uncertainty components. It assumes:
    - Perfect knowledge of task rules
    - No uncertainty about evidence quality
    - Decisions based purely on reward-effort tradeoff
    
    Parameters:
    -----------
    reward_weight : float (default: 1.0)
        β_r - How much weight is given to reward benefits
        Higher values mean reward is more important
        Interpretation: "How much do I care about getting rewards?"
        
    effort_cost_weight : float (default: 1.0)
        c_e - How costly effort is perceived to be
        Higher values mean effort is more aversive
        Interpretation: "How much do I dislike expending mental effort?"
        
    effort_exponent : float (default: 2.0)
        α - Exponent for effort cost function
        Typically 2.0 (quadratic cost)
        Interpretation: "How quickly does effort cost increase with control?"
        - α = 1: Linear (doubling control doubles cost)
        - α = 2: Quadratic (doubling control quadruples cost) ← DEFAULT
        - α > 2: Even steeper (effort becomes very expensive quickly)
    """
    
    def __init__(
        self,
        reward_weight: float = 1.0,
        effort_cost_weight: float = 1.0,
        effort_exponent: float = 2.0
    ):
        """
        Initialize Traditional EVC model.
        
        Args:
            reward_weight: β_r - Reward sensitivity parameter
            effort_cost_weight: c_e - Effort cost scaling parameter
            effort_exponent: α - Exponent for effort cost function
        """
        # Store parameters
        # reward_weight (β_r): How much reward matters
        self.reward_weight = reward_weight
        
        # effort_cost_weight (c_e): How costly effort is
        self.effort_cost_weight = effort_cost_weight
        
        # effort_exponent (α): Shape of effort cost curve
        self.effort_exponent = effort_exponent
    
    def compute_evc(
        self,
        expected_reward: float,
        control_level: float
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Expected Value of Control for a given control level.
        
        Formula: EVC(c) = β_r × E[Reward|c] - c_e × c^α
        
        Args:
            expected_reward: E[Reward|c] - Expected reward given control level
                            This is typically: reward_magnitude × P(success|c)
            control_level: c - Level of cognitive control to deploy (0-1)
            
        Returns:
            Tuple of:
            - EVC value (float)
            - Dictionary with component breakdown:
              {
                  'reward_benefit': benefit from reward,
                  'effort_cost': cost of effort
              }
        
        Example:
        --------
        >>> model = TraditionalEVC(reward_weight=1.0, effort_cost_weight=0.5)
        >>> evc, components = model.compute_evc(expected_reward=10.0, control_level=0.5)
        >>> print(f"EVC: {evc:.2f}")
        >>> print(f"Reward benefit: {components['reward_benefit']:.2f}")
        >>> print(f"Effort cost: {components['effort_cost']:.2f}")
        """
        # REWARD BENEFIT COMPONENT
        # How much value do we get from deploying control?
        # Formula: reward_benefit = β_r × E[Reward|c] × c
        # 
        # Explanation:
        # - β_r (reward_weight): scaling factor for how much reward matters
        # - expected_reward: the expected reward given this control level
        # - control_level: how much control we're deploying
        #
        # The benefit increases linearly with both expected reward and control
        reward_benefit = self.reward_weight * expected_reward * control_level
        
        # EFFORT COST COMPONENT
        # How much does it "cost" to deploy this level of control?
        # Formula: effort_cost = c_e × c^α
        #
        # Explanation:
        # - c_e (effort_cost_weight): scaling factor for effort cost
        # - control_level^effort_exponent: non-linear cost function
        #   With α=2 (quadratic), doubling control quadruples cost
        #
        # This captures the idea that mental effort becomes increasingly
        # expensive as you deploy more control
        effort_cost_term = self.effort_cost_weight * (control_level ** self.effort_exponent)
        
        # TOTAL EVC
        # EVC = Benefit - Cost
        # Positive EVC means control is worth deploying
        # Negative EVC means control costs more than it's worth
        evc = reward_benefit - effort_cost_term
        
        # Return both the total EVC and component breakdown
        return evc, {
            'reward_benefit': reward_benefit,
            'effort_cost': effort_cost_term
        }
    
    def optimal_control(self, expected_reward: float) -> float:
        """
        Calculate the optimal control level that maximizes EVC.
        
        Mathematical derivation:
        -----------------------
        EVC(c) = β_r × E[Reward|c] × c - c_e × c^α
        
        For optimal control, take derivative and set to zero:
        dEVC/dc = β_r × E[Reward] - c_e × α × c^(α-1) = 0
        
        Solving for c:
        c* = (β_r × E[Reward]) / (c_e × α)
        
        However, this assumes E[Reward] is constant (doesn't depend on c).
        In practice, E[Reward|c] = reward × (baseline + control × gain)
        
        If we assume linear relationship: E[Reward|c] = reward × (a + b×c)
        Then: c* = clip((β_r × reward × b) / (c_e × α), 0, 1)
        
        Args:
            expected_reward: E[Reward] - Expected reward magnitude
                            This is the reward you'd get if successful
            
        Returns:
            Optimal control level (0-1) that maximizes EVC
        
        Example:
        --------
        >>> model = TraditionalEVC(reward_weight=1.0, effort_cost_weight=0.5)
        >>> optimal = model.optimal_control(expected_reward=10.0)
        >>> print(f"Optimal control: {optimal:.3f}")
        """
        # Optimal control formula (simplified version)
        # This assumes expected_reward is constant and doesn't depend on control
        # 
        # Formula: c* = (β_r × E[Reward]) / (c_e × α)
        #
        # Explanation:
        # - Numerator (β_r × expected_reward): Reward benefit scale
        # - Denominator (c_e × α): Effort cost scaling
        # - Higher reward → more control worth deploying
        # - Higher effort cost → less control worth deploying
        #
        # The division by α accounts for the non-linear cost function
        optimal = (self.reward_weight * expected_reward) / (2 * self.effort_cost_weight)
        
        # Clip to valid range [0, 1]
        # Control can't be negative or greater than 1
        return np.clip(optimal, 0, 1)
    
    def predict_control(
        self,
        data: pd.DataFrame,
        reward_col: str = 'reward_magnitude',
        accuracy_col: str = 'evidence_clarity'
    ) -> np.ndarray:
        """
        Predict optimal control levels for a dataset.
        
        For each trial, computes:
        1. Expected reward = reward_magnitude × P(success)
        2. Optimal control = argmax_c EVC(c)
        
        Args:
            data: DataFrame with trial data
            reward_col: Column name for reward magnitude
            accuracy_col: Column name for accuracy/probability of success
            
        Returns:
            Array of predicted optimal control levels
        """
        predictions = []
        
        for _, row in data.iterrows():
            # Get reward magnitude for this trial
            reward = row[reward_col]
            
            # Estimate expected reward
            # In traditional EVC, we assume expected reward ≈ reward × accuracy
            # accuracy_col could be evidence_clarity (probability of being correct)
            expected_reward = reward * row[accuracy_col]
            
            # Compute optimal control for this expected reward
            optimal_c = self.optimal_control(expected_reward)
            predictions.append(optimal_c)
        
        return np.array(predictions)
    
    def fit(
        self,
        data: pd.DataFrame,
        observed_control_col: str = 'control_signal',
        reward_col: str = 'reward_magnitude',
        accuracy_col: str = 'evidence_clarity'
    ) -> Dict[str, float]:
        """
        Fit model parameters to observed control allocation data.
        
        Optimization process:
        ---------------------
        1. Define objective function: minimize prediction error
        2. Use scipy.optimize to find best parameters
        3. Compute R², RMSE, correlation
        
        The optimization tries to find parameters that make predicted control
        match observed control as closely as possible.
        
        Args:
            data: DataFrame with observed behavioral data
            observed_control_col: Column name for observed control signals
            reward_col: Column name for reward magnitudes
            accuracy_col: Column name for accuracy/probability values
            
        Returns:
            Dictionary with fitted parameters and performance metrics:
            {
                'reward_weight': fitted β_r,
                'effort_cost_weight': fitted c_e,
                'effort_exponent': fitted α (may be fixed),
                'r2': R-squared (goodness of fit),
                'rmse': Root mean squared error,
                'correlation': Pearson correlation
            }
        """
        # Extract observed control signals
        observed_control = data[observed_control_col].values
        
        # Extract predictors
        rewards = data[reward_col].values
        accuracies = data[accuracy_col].values
        
        # Define objective function for optimization
        # We want to minimize the difference between predicted and observed control
        def objective(params):
            """
            Objective function: minimize prediction error.
            
            Args:
                params: [reward_weight, effort_cost_weight]
                        (effort_exponent is typically fixed at 2.0)
            
            Returns:
                Mean squared error between predicted and observed control
            """
            reward_w, effort_w = params
            
            # Temporarily update parameters
            old_reward_w = self.reward_weight
            old_effort_w = self.effort_cost_weight
            
            self.reward_weight = reward_w
            self.effort_cost_weight = effort_w
            
            # Predict control for all trials
            predictions = []
            for reward, accuracy in zip(rewards, accuracies):
                expected_reward = reward * accuracy
                pred_control = self.optimal_control(expected_reward)
                predictions.append(pred_control)
            
            predictions = np.array(predictions)
            
            # Compute mean squared error
            mse = np.mean((observed_control - predictions) ** 2)
            
            # Restore parameters
            self.reward_weight = old_reward_w
            self.effort_cost_weight = old_effort_w
            
            return mse
        
        # Initial parameter guesses
        initial_params = [self.reward_weight, self.effort_cost_weight]
        
        # Bounds: parameters must be positive
        bounds = [(0.1, 10.0), (0.1, 10.0)]
        
        # Optimize: find parameters that minimize prediction error
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Update model with fitted parameters
        self.reward_weight = result.x[0]
        self.effort_cost_weight = result.x[1]
        
        # Compute predictions with fitted parameters
        predictions = self.predict_control(data, reward_col, accuracy_col)
        
        # Compute performance metrics
        ss_res = np.sum((observed_control - predictions) ** 2)  # Sum of squared residuals
        ss_tot = np.sum((observed_control - np.mean(observed_control)) ** 2)  # Total sum of squares
        r2 = 1 - (ss_res / ss_tot)  # R-squared: proportion of variance explained
        
        rmse = np.sqrt(np.mean((observed_control - predictions) ** 2))  # Root mean squared error
        
        correlation = np.corrcoef(observed_control, predictions)[0, 1]  # Pearson correlation
        
        return {
            'reward_weight': self.reward_weight,
            'effort_cost_weight': self.effort_cost_weight,
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
        accuracy_col: str = 'evidence_clarity'
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            data: DataFrame with test data
            observed_control_col: Column name for observed control
            reward_col: Column name for rewards
            accuracy_col: Column name for accuracy
            
        Returns:
            Dictionary with performance metrics: {'r2', 'rmse', 'correlation'}
        """
        observed = data[observed_control_col].values
        predictions = self.predict_control(data, reward_col, accuracy_col)
        
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
        add_noise: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict behavioral outcomes (control allocation and performance).
        
        Args:
            expected_reward: Expected reward for each trial
            add_noise: Whether to add random noise to predictions
            
        Returns:
            Tuple of (control_levels, performance_scores)
        """
        # Compute optimal control
        control = self.optimal_control(expected_reward)
        
        # Optionally add noise to simulate behavioral variability
        if add_noise:
            # Add Gaussian noise with std=0.1
            noise = np.random.normal(0, 0.1, size=control.shape)
            control = np.clip(control + noise, 0, 1)
        
        # Performance is assumed to scale with control
        # Simple model: performance = 0.8 × control
        # (80% efficiency of control deployment)
        performance = np.clip(control * 0.8, 0, 1)
        
        return control, performance


# Example usage and testing
if __name__ == '__main__':
    print("Traditional EVC Model - Example Usage\n")
    print("=" * 70)
    
    # Create model
    model = TraditionalEVC(
        reward_weight=1.0,      # β_r: reward matters a lot
        effort_cost_weight=0.5, # c_e: effort is moderately costly
        effort_exponent=2.0     # α: quadratic effort cost
    )
    
    # Example 1: Compute EVC for different control levels
    print("\nExample 1: EVC at different control levels")
    print("-" * 70)
    expected_reward = 10.0
    for control in [0.2, 0.4, 0.6, 0.8]:
        evc, components = model.compute_evc(expected_reward, control)
        print(f"Control={control:.1f}: EVC={evc:.3f} "
              f"(Benefit={components['reward_benefit']:.3f}, "
              f"Cost={components['effort_cost']:.3f})")
    
    # Example 2: Find optimal control
    print("\nExample 2: Optimal control level")
    print("-" * 70)
    optimal = model.optimal_control(expected_reward)
    print(f"For expected_reward={expected_reward}, optimal control = {optimal:.3f}")
    
    # Example 3: Show how optimal control changes with reward
    print("\nExample 3: Optimal control vs reward")
    print("-" * 70)
    for reward in [1, 5, 10, 20]:
        optimal = model.optimal_control(reward)
        print(f"Reward={reward:2d}: optimal control = {optimal:.3f}")
