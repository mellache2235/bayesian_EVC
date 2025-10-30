# Literature Review: Bayesian Approaches to Uncertainty in EVC and DDM

## Overview

This document summarizes related approaches for incorporating uncertainty and Bayesian statistics into Expected Value of Control (EVC) calculations, Drift Diffusion Models (DDM), and related cognitive control frameworks.

---

## 1. Bayesian Drift Diffusion Models (DDM)

### 1.1 Mathematical Equivalence of DDM and Bayesian Models

**Key Finding:** DDM is mathematically equivalent to certain Bayesian models of evidence accumulation.

**Reference:** Bitzer et al. (2014) - Derived equations relating DDM parameters to Bayesian model parameters

**Implications for EVC:**
- DDM can be interpreted as optimal Bayesian inference
- Drift rate represents the quality of evidence
- Decision threshold represents the balance between speed and accuracy
- Confidence can be derived from the posterior distribution

**How to Implement:**
```python
# Bayesian interpretation of DDM
# Posterior belief about correct choice after time t:
# P(correct | evidence) = 1 / (1 + exp(-2 * drift_rate * evidence / diffusion_constant))

# Confidence = posterior probability of being correct
confidence = 1 / (1 + np.exp(-2 * drift_rate * accumulated_evidence / noise))
```

### 1.2 Hierarchical Bayesian DDM (HDDM)

**Key Approach:** Use hierarchical Bayesian models to estimate DDM parameters across individuals

**Benefits:**
- Accounts for individual differences
- Provides uncertainty estimates for parameters
- Handles missing data and outliers better
- Enables group-level inference

**Python Implementation:**
```python
# Using HDDM package
import hddm

# Hierarchical model with group differences
model = hddm.HDDM(data, depends_on={'v': 'condition'})
model.sample(2000, burn=200)

# Extract uncertainty estimates
v_samples = model.nodes_db.loc['v', 'node'].trace()
uncertainty = np.std(v_samples)
```

**Relevance to EVC:**
- Can estimate uncertainty about evidence quality (drift rate)
- Provides trial-by-trial confidence estimates
- Accounts for individual differences in uncertainty tolerance

---

## 2. Post-Decision Confidence in DDM

### 2.1 Confidence as Balance of Evidence

**Key Insight:** Confidence reflects the balance of accumulated evidence at decision time

**Approaches:**

**A. Balance of Evidence Model:**
```python
# Confidence based on evidence difference
confidence = abs(evidence_for - evidence_against) / total_time
```

**B. Posterior Probability Model:**
```python
# Confidence as posterior probability
log_odds = 2 * drift_rate * decision_time / diffusion_constant
confidence = 1 / (1 + np.exp(-log_odds))
```

**C. Time-dependent Confidence:**
```python
# Confidence decreases with longer RTs (more uncertainty)
confidence = base_confidence * np.exp(-lambda_param * reaction_time)
```

### 2.2 Metacognitive Uncertainty

**Key Concept:** Distinguish between:
1. **Type 1 uncertainty:** Uncertainty about the decision itself
2. **Type 2 uncertainty:** Uncertainty about one's confidence (metacognition)

**Implementation:**
```python
# Type 1: Decision uncertainty from evidence
decision_uncertainty = 1 - abs(evidence_difference) / threshold

# Type 2: Metacognitive uncertainty
metacognitive_noise = np.random.normal(0, meta_noise_std)
reported_confidence = true_confidence + metacognitive_noise
```

---

## 3. Bayesian Learning and Uncertainty

### 3.1 Volatility and Uncertainty Estimation

**Key Papers:** Behrens et al. (2007), Daw et al. (2005)

**Approach:** Track two types of uncertainty:
1. **Expected uncertainty (risk):** Known variability in outcomes
2. **Unexpected uncertainty (volatility):** Changes in the environment

**Bayesian Update:**
```python
# Hierarchical Bayesian learning
# Level 1: Estimate current value
mu_t = mu_t-1 + alpha * prediction_error

# Level 2: Estimate volatility
alpha_t = alpha_base + beta * abs(prediction_error)

# Uncertainty increases with volatility
uncertainty_t = base_uncertainty + gamma * volatility_estimate
```

**Neural Correlates:**
- Anterior Cingulate Cortex (ACC): Tracks volatility/uncertainty
- Dorsolateral Prefrontal Cortex (DLPFC): Implements control
- Striatum: Tracks reward prediction errors

### 3.2 State Uncertainty

**Approach:** Use Hidden Markov Models (HMM) with Bayesian inference

```python
# Belief about current state
belief_state_1 = P(state=1 | observations)
belief_state_2 = P(state=2 | observations)

# State uncertainty (entropy)
state_uncertainty = -sum(p * log(p) for p in beliefs)

# Update beliefs with Bayes rule
posterior = likelihood * prior / normalization
```

---

## 4. Expected Value of Sample Information (EVSI)

### 4.1 Bayesian Value of Information

**Key Concept:** Quantify the value of reducing uncertainty through information gathering

**Formula:**
```
EVSI = E[max(benefit with new info)] - max(benefit without new info)
```

**Application to EVC:**
- Information gathering = allocating cognitive control
- Benefit = improved decision accuracy
- Cost = cognitive effort

**Implementation:**
```python
# Value of allocating control to reduce uncertainty
def compute_evsi(current_uncertainty, control_allocation):
    # Expected uncertainty after control
    reduced_uncertainty = current_uncertainty * (1 - control_allocation * efficiency)
    
    # Expected benefit from reduced uncertainty
    benefit_with_control = expected_reward * (1 - reduced_uncertainty)
    benefit_without_control = expected_reward * (1 - current_uncertainty)
    
    # EVSI
    evsi = benefit_with_control - benefit_without_control
    
    # Net value (subtract effort cost)
    net_value = evsi - effort_cost(control_allocation)
    
    return net_value
```

---

## 5. Integrating Uncertainty into EVC

### 5.1 Traditional EVC Formula

```
EVC = Expected_Reward - Effort_Cost
    = P(success) × Reward - Cost(control)
```

### 5.2 Uncertainty-Aware EVC (Proposed Approaches)

**Approach 1: Confidence-Weighted Rewards**
```python
# Weight expected rewards by confidence
EVC = confidence × P(success) × Reward - Cost(control)
```

**Approach 2: Explicit Uncertainty Reduction Benefit**
```python
# Add value of reducing uncertainty
EVC = Expected_Reward - Effort_Cost + Uncertainty_Reduction_Benefit
    = P(success) × Reward - Cost(control) + w_u × Uncertainty × Control
```

**Approach 3: Risk-Sensitive EVC**
```python
# Account for uncertainty in reward prediction
mean_reward = P(success) × Reward
variance_reward = uncertainty × Reward^2

# Risk-sensitive value (CARA utility)
EVC = mean_reward - 0.5 × risk_aversion × variance_reward - Cost(control)
```

**Approach 4: Information-Theoretic EVC**
```python
# Value information gain
information_gain = KL_divergence(posterior || prior)
EVC = Expected_Reward - Effort_Cost + w_info × information_gain
```

---

## 6. Practical Implementation Strategies

### 6.1 Trial-by-Trial Uncertainty Estimation

```python
class UncertaintyEstimator:
    def __init__(self):
        self.belief_state = np.ones(n_states) / n_states
        
    def estimate_uncertainty(self, evidence_clarity, outcome):
        # Decision uncertainty from evidence
        decision_uncertainty = 1 - evidence_clarity
        
        # Update state beliefs (Bayesian)
        likelihood = self.get_likelihood(outcome)
        posterior = likelihood * self.belief_state
        posterior /= posterior.sum()
        
        # State uncertainty (entropy)
        state_uncertainty = -np.sum(posterior * np.log(posterior + 1e-10))
        
        # Combined uncertainty
        total_uncertainty = 0.5 * decision_uncertainty + 0.5 * state_uncertainty
        
        # Update beliefs
        self.belief_state = posterior
        
        return {
            'decision_uncertainty': decision_uncertainty,
            'state_uncertainty': state_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence': 1 - total_uncertainty
        }
```

### 6.2 Bayesian EVC with Uncertainty

```python
class BayesianEVC:
    def compute_evc(self, reward, baseline_acc, control, uncertainty, confidence):
        # Confidence-weighted expected reward
        success_prob = baseline_acc + (1 - baseline_acc) * control * 0.5
        expected_reward = reward * success_prob * confidence
        
        # Effort cost
        effort_cost = control ** 2
        
        # Uncertainty reduction benefit
        # More control reduces uncertainty more when uncertainty is high
        uncertainty_benefit = uncertainty * control
        
        # Combined EVC
        evc = (self.w_reward * expected_reward - 
               self.w_effort * effort_cost + 
               self.w_uncertainty * uncertainty_benefit)
        
        return evc
```

---

## 7. Key Papers and Resources

### Foundational Papers

1. **EVC Theory:**
   - Shenhav, A., et al. (2013). "The expected value of control: An integrative theory of anterior cingulate cortex function." *Neuron*, 79(2), 217-240.

2. **Drift Diffusion Model:**
   - Ratcliff, R., & McKoon, G. (2008). "The diffusion decision model: Theory and data for two-choice decision tasks." *Neural Computation*, 20(4), 873-922.

3. **Bayesian DDM:**
   - Bitzer, S., et al. (2014). "Perceptual decision making: Drift-diffusion model is equivalent to a Bayesian model." *Frontiers in Human Neuroscience*, 8, 102.

4. **Hierarchical Bayesian DDM:**
   - Wiecki, T. V., et al. (2013). "HDDM: Hierarchical Bayesian estimation of the drift-diffusion model in Python." *Frontiers in Neuroinformatics*, 7, 14.

5. **Bayesian Learning and Uncertainty:**
   - Behrens, T. E., et al. (2007). "Learning the value of information in an uncertain world." *Nature Neuroscience*, 10(9), 1214-1221.
   - Daw, N. D., et al. (2005). "Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control." *Nature Neuroscience*, 8(12), 1704-1711.

6. **Confidence and Metacognition:**
   - Fleming, S. M., & Dolan, R. J. (2012). "The neural basis of metacognitive ability." *Philosophical Transactions of the Royal Society B*, 367(1594), 1338-1349.

7. **Value of Information:**
   - Howard, R. A. (1966). "Information value theory." *IEEE Transactions on Systems Science and Cybernetics*, 2(1), 22-26.

### Software Tools

1. **HDDM (Hierarchical Drift Diffusion Model):**
   ```bash
   pip install hddm
   ```
   - Python package for Bayesian estimation of DDM parameters
   - Handles hierarchical models and group differences

2. **PyMC:**
   ```bash
   pip install pymc
   ```
   - General-purpose Bayesian modeling
   - Can implement custom EVC models with uncertainty

3. **Stan:**
   - Probabilistic programming language
   - Excellent for hierarchical Bayesian models

---

## 8. Recommendations for Implementation

### For Your Bayesian EVC Project:

1. **Decision Uncertainty:**
   - Use evidence clarity as proxy for drift rate quality
   - Compute confidence from DDM-inspired formula
   - Include trial-by-trial variability

2. **State Uncertainty:**
   - Implement Bayesian belief updating over task states
   - Use entropy as uncertainty measure
   - Track volatility in rule changes

3. **EVC Calculation:**
   - Weight expected rewards by confidence
   - Add explicit uncertainty reduction term
   - Fit uncertainty weight parameter from data

4. **Model Comparison:**
   - Compare traditional EVC vs. Bayesian EVC
   - Use cross-validation for robust comparison
   - Examine individual differences in uncertainty sensitivity

5. **Neural Validation:**
   - Correlate uncertainty with ACC activity
   - Correlate control with DLPFC activity
   - Test if uncertainty × control predicts ACC-DLPFC coupling

---

## 9. Advanced Extensions

### 9.1 Multi-Level Uncertainty

```python
# Level 1: Observation uncertainty (noise in evidence)
observation_uncertainty = 1 / evidence_quality

# Level 2: State uncertainty (which rule is active)
state_uncertainty = entropy(belief_distribution)

# Level 3: Volatility uncertainty (how fast rules change)
volatility_uncertainty = variance(state_transition_rate)

# Hierarchical uncertainty
total_uncertainty = (w1 * observation_uncertainty + 
                    w2 * state_uncertainty + 
                    w3 * volatility_uncertainty)
```

### 9.2 Active Inference Framework

```python
# Free energy minimization
# Control allocation to minimize expected free energy

def expected_free_energy(control, beliefs, preferences):
    # Pragmatic value (achieving goals)
    pragmatic_value = expected_utility(control, preferences)
    
    # Epistemic value (reducing uncertainty)
    epistemic_value = expected_information_gain(control, beliefs)
    
    # Total expected free energy
    EFE = pragmatic_value + epistemic_value
    
    return -EFE  # Minimize free energy = maximize value
```

### 9.3 Thompson Sampling for Exploration

```python
# Bayesian approach to exploration-exploitation
# Sample from posterior distribution of values

def thompson_sampling_control(beliefs_about_control_effectiveness):
    # Sample from posterior
    sampled_effectiveness = np.random.normal(
        beliefs_about_control_effectiveness['mean'],
        beliefs_about_control_effectiveness['std']
    )
    
    # Choose control based on sample
    optimal_control = optimize_evc(sampled_effectiveness)
    
    return optimal_control
```

---

## 10. Summary and Conclusions

### Key Insights:

1. **DDM and Bayesian models are equivalent** - Can use DDM framework with Bayesian interpretation

2. **Confidence can be derived from evidence** - Use balance of evidence or posterior probability

3. **Multiple types of uncertainty** - Decision uncertainty, state uncertainty, volatility

4. **Uncertainty has value** - Reducing uncertainty provides benefit beyond reward

5. **Hierarchical models are powerful** - Account for individual differences and provide uncertainty estimates

### Best Practices for Bayesian EVC:

✅ **Explicitly model uncertainty** - Don't just assume it's implicit  
✅ **Use hierarchical models** - Account for individual differences  
✅ **Validate with neural data** - Check ACC/DLPFC correlations  
✅ **Compare models rigorously** - Use cross-validation and multiple metrics  
✅ **Interpret parameters** - Uncertainty weights reveal cognitive strategies  

### Future Directions:

1. **Computational psychiatry** - Individual differences in uncertainty processing
2. **Developmental studies** - How uncertainty sensitivity changes with age
3. **Neural network models** - Deep learning approaches to uncertainty
4. **Real-world applications** - Educational interventions, clinical treatments

---

## References

See Section 7 for complete list of key papers and resources.

---

**Last Updated:** 2025  
**For Implementation:** See `models/bayesian_evc.py` and `models/bayesian_uncertainty.py`

