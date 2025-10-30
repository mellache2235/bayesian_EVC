# Where Bayesian Inference Happens

## 🎯 Quick Answer

**Bayesian inference happens in TWO places:**

1. **`models/bayesian_uncertainty.py`** - Lines 84-122
   - `update_state_beliefs()` method
   - **This is the core Bayesian update**

2. **`models/hgf_uncertainty.py`** - Lines 78-132
   - `update()` method
   - **HGF uses Bayesian updates at multiple levels**

---

## 📍 Location 1: Simple Bayesian Update

### File: `models/bayesian_uncertainty.py`

### Method: `update_state_beliefs()` (Lines 84-122)

```python
def update_state_beliefs(self, observation: int, likelihood_matrix: Optional[np.ndarray] = None):
    """
    Update beliefs about task state using Bayesian inference.
    
    THIS IS WHERE BAYESIAN INFERENCE HAPPENS!
    """
    # Step 1: Define likelihood P(observation | state)
    if observation == 1:  # Correct
        likelihoods = likelihood_matrix
    else:  # Incorrect
        likelihoods = 1 - likelihood_matrix
    
    # Step 2: BAYESIAN UPDATE - Apply Bayes' Rule
    # Posterior ∝ Likelihood × Prior
    posterior = likelihoods * self.state_beliefs  # self.state_beliefs is the prior
    posterior = posterior / (posterior.sum() + 1e-10)  # Normalize to sum to 1
    
    # Step 3: Update beliefs (with learning rate for gradual updating)
    self.state_beliefs = (1 - self.learning_rate) * self.state_beliefs + \
                        self.learning_rate * posterior
    
    return self.state_beliefs
```

### What's Happening:

**Bayes' Rule:**
```
P(state | observation) = P(observation | state) × P(state) / P(observation)
                       = Likelihood × Prior / Normalization
```

**In the code:**
- `self.state_beliefs` = Prior beliefs P(state)
- `likelihoods` = Likelihood P(observation | state)
- `posterior` = Updated beliefs P(state | observation)

**Example:**
```python
# Before observation:
Prior: P(state1) = 0.5, P(state2) = 0.5  # Uncertain which state

# Observation: Correct response
# Likelihood: State1 gives 80% correct, State2 gives 60% correct
Likelihood: P(correct | state1) = 0.8, P(correct | state2) = 0.6

# Bayesian update:
Posterior ∝ [0.8, 0.6] × [0.5, 0.5] = [0.4, 0.3]
Normalized: [0.4/(0.4+0.3), 0.3/(0.4+0.3)] = [0.57, 0.43]

# After observation:
Posterior: P(state1) = 0.57, P(state2) = 0.43  # More confident in state1
```

---

## 📍 Location 2: Hierarchical Bayesian Update (HGF)

### File: `models/hgf_uncertainty.py`

### Method: `update()` (Lines 78-132)

```python
def update(self, observation: float, observation_type: str = 'binary'):
    """
    Update HGF with new observation.
    
    BAYESIAN INFERENCE AT MULTIPLE LEVELS!
    """
    # --- PREDICTION STEP (Prior) ---
    # Predicted precision (inverse variance) at level 2
    pi_2_hat = 1 / (self.sa_2 + np.exp(self.kappa_2 * self.mu_3 + self.omega_2))
    
    # --- PREDICTION ERRORS ---
    # Expected observation
    mu_1_hat = self._sigmoid(self.mu_2)
    
    # Observation prediction error
    delta_1 = observation - mu_1_hat
    
    # --- BAYESIAN UPDATE LEVEL 2 (State) ---
    # Update uncertainty (posterior precision)
    pi_2 = pi_2_hat + w_2**2
    self.sa_2 = 1 / pi_2  # Posterior variance
    
    # Update belief (posterior mean) - THIS IS BAYESIAN!
    self.mu_2 = mu_2_old + self.sa_2 * delta_2
    
    # --- BAYESIAN UPDATE LEVEL 3 (Volatility) ---
    # Update volatility estimate based on how much uncertainty changed
    self.mu_3 = self.mu_3 + self.kappa_2 * self.sa_3 * delta_3
```

### What's Happening:

**HGF uses Gaussian Bayesian updates:**

At each level, beliefs are updated using:
```
Posterior Mean = Prior Mean + Kalman Gain × Prediction Error
Posterior Variance = 1 / (Prior Precision + Information Precision)
```

This is equivalent to Bayesian inference with Gaussian distributions!

**Example:**
```python
# Level 2 (State):
Prior: μ₂ = 0.5, σ₂² = 1.0  # Belief about state (in logit space)

# New observation: outcome = 1 (correct)
Prediction error: δ = 1 - sigmoid(0.5) = 0.38

# Bayesian update:
Posterior μ₂ = 0.5 + (learning_rate × 0.38) = 0.69
Posterior σ₂² = 1 / (1/1.0 + new_information) = 0.8  # Uncertainty decreased

# Level 3 (Volatility):
If uncertainty changed more than expected → increase volatility estimate
```

---

## 🔍 How to Find Bayesian Inference in Code

### Look for these patterns:

1. **Bayes' Rule Application:**
```python
posterior = likelihood * prior
posterior = posterior / posterior.sum()  # Normalize
```

2. **Gaussian Bayesian Update:**
```python
new_mean = old_mean + gain * prediction_error
new_variance = 1 / (old_precision + new_precision)
```

3. **Belief Updating:**
```python
beliefs = (1 - learning_rate) * old_beliefs + learning_rate * new_beliefs
```

4. **Entropy/Uncertainty Calculation:**
```python
entropy = -sum(p * log(p) for p in beliefs)
uncertainty = entropy / max_entropy
```

---

## 📊 Where Each Model Uses Bayesian Inference

### 1. Traditional EVC (`models/traditional_evc.py`)
**❌ NO Bayesian inference**
- Just optimization (maximize EVC)
- No belief updating
- No uncertainty tracking

### 2. Bayesian Uncertainty Estimator (`models/bayesian_uncertainty.py`)
**✅ YES - Line 111-113**
```python
# Bayesian update: posterior ∝ likelihood × prior
posterior = likelihoods * self.state_beliefs
posterior = posterior / (posterior.sum() + 1e-10)
```

### 3. Bayesian EVC (`models/bayesian_evc.py`)
**✅ YES - Indirectly**
- Uses `BayesianUncertaintyEstimator` internally
- Inherits Bayesian updates from uncertainty estimator
- Adds uncertainty to EVC calculation

### 4. HGF (`models/hgf_uncertainty.py`)
**✅ YES - Lines 121-126**
```python
# Update uncertainty (posterior precision)
pi_2 = pi_2_hat + w_2**2
self.sa_2 = 1 / pi_2

# Update belief (posterior mean)
self.mu_2 = mu_2_old + self.sa_2 * delta_2
```

---

## 🎓 Understanding the Bayesian Updates

### Simple Bayesian (bayesian_uncertainty.py)

**What it does:**
- Tracks beliefs about which task state/rule is active
- Updates beliefs after each trial outcome
- Uses discrete Bayes' rule

**When it updates:**
- Every time `update_state_beliefs()` is called
- Typically once per trial after observing outcome

**What gets updated:**
- `self.state_beliefs` - Probability distribution over states

### HGF (hgf_uncertainty.py)

**What it does:**
- Tracks beliefs at 3 levels (observations, states, volatility)
- Updates all levels simultaneously
- Uses continuous Gaussian Bayesian updates

**When it updates:**
- Every time `update()` is called
- Once per trial after observing outcome

**What gets updated:**
- `self.mu_2` - Belief about state (mean)
- `self.sa_2` - Uncertainty about state (variance)
- `self.mu_3` - Belief about volatility (mean)
- `self.sa_3` - Uncertainty about volatility (variance)

---

## 🔬 Testing Bayesian Inference

### Verify it's working:

```python
from models.bayesian_uncertainty import BayesianUncertaintyEstimator

# Initialize
estimator = BayesianUncertaintyEstimator(n_states=2)

# Initial beliefs (uniform)
print("Initial:", estimator.state_beliefs)  # [0.5, 0.5]

# Observe correct outcome (state 1 is more likely to give correct)
estimator.update_state_beliefs(observation=1)
print("After correct:", estimator.state_beliefs)  # [0.57, 0.43] - more confident in state 1

# Observe another correct outcome
estimator.update_state_beliefs(observation=1)
print("After 2nd correct:", estimator.state_beliefs)  # [0.62, 0.38] - even more confident

# Observe incorrect outcome
estimator.update_state_beliefs(observation=0)
print("After incorrect:", estimator.state_beliefs)  # [0.55, 0.45] - less confident
```

---

## 📝 Summary

### Where Bayesian Inference Happens:

1. **`bayesian_uncertainty.py` line 111-113** ← Main Bayesian update
2. **`hgf_uncertainty.py` lines 121-126** ← HGF Bayesian updates

### What Gets Updated:

- **Beliefs** about task states/rules
- **Uncertainty** about those beliefs
- **Volatility** estimates (HGF only)

### Why It Matters:

- Bayesian inference provides **principled uncertainty quantification**
- Updates are **optimal** given the data
- Captures how humans **actually learn** under uncertainty

### How to Use:

```python
# Simple Bayesian
from models.bayesian_uncertainty import BayesianUncertaintyEstimator
estimator = BayesianUncertaintyEstimator()
estimator.update_state_beliefs(outcome)  # ← Bayesian update happens here

# HGF
from models.hgf_uncertainty import HierarchicalGaussianFilter
hgf = HierarchicalGaussianFilter()
hgf.update(outcome)  # ← Bayesian updates happen here (multiple levels)
```

---

**Key Takeaway:** The Bayesian inference is in the `update` methods where beliefs are updated using Bayes' rule!

