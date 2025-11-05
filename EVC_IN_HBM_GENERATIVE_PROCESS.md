# Embedding EVC in Hierarchical Bayesian Generative Process

## What This Means

**"Embed Expected Value of Control in the HBM generative process"**

This means: **Use EVC as the generative model in hierarchical Bayesian framework**

Instead of:
- ❌ Fitting EVC separately, then HDDM separately, then linking them (complex!)
- ❌ Using regression to connect models (indirect)

Do:
- ✅ **EVC IS the generative model** (how data are generated)
- ✅ Fit hierarchical Bayesian model WITH EVC as data-generating process
- ✅ Direct inference on EVC parameters

---

## The Concept: Generative Model

### **What is a Generative Model?**

**A generative model specifies how observed data arise from underlying processes.**

```
Generative Process:
    Parameters → Process → Observations
    
Example:
    μ, σ → Normal(μ, σ) → Data
    
For EVC:
    λ, β_r, β_e → EVC formula → Control → RT/Accuracy
```

---

### **Traditional Approach (What You've Been Doing):**

```python
# Step 1: Observe control (or proxy like RT)
control_observed = data['control_signal']

# Step 2: Fit EVC parameters to match observations
def loss(params):
    predicted = evc_formula(params)
    return mse(predicted, control_observed)

params_fitted = minimize(loss)
```

**This is:** Point estimation via optimization

---

### **Hierarchical Bayesian Generative Approach (Proposed):**

```python
# Specify how data are GENERATED
with pm.Model() as generative_model:
    # Parameters generate control
    lambda_i = pm.Normal('lambda', mu=mu_lambda, sigma=sigma_lambda, shape=n_children)
    
    # EVC formula is the GENERATIVE PROCESS
    predicted_control = evc_formula(lambda_i, reward, uncertainty)
    
    # Control generates observations (RT, accuracy)
    rt_obs = pm.Normal('rt', mu=f(predicted_control), sigma=sigma_rt, observed=rt_data)
    acc_obs = pm.Bernoulli('acc', p=g(predicted_control), observed=acc_data)
    
    # Infer parameters
    trace = pm.sample()
```

**This is:** Full Bayesian inference with EVC as generative core

---

## Why This is Better

### **Advantage 1: Unified Framework**

**Instead of:**
```
Fit EVC → Get predictions → Compare to data → Compute R²
(Two-stage, indirect)
```

**Do:**
```
EVC generates data → Bayesian inference → Posterior over parameters
(Single integrated framework)
```

---

### **Advantage 2: Uncertainty Quantification**

**Traditional:**
```
λ = 0.42 (point estimate)
(How certain are we? Unknown!)
```

**Generative HBM:**
```
λ ~ Posterior distribution
Mean = 0.42
95% CI = [0.28, 0.56]
P(λ > 0) = 0.998
(Full uncertainty quantification!)
```

---

### **Advantage 3: Predictive Distributions**

**Traditional:**
```
Predict single control value
control_pred = 0.55
(No uncertainty in prediction)
```

**Generative HBM:**
```
Sample from posterior predictive
control_pred ~ Distribution
Mean = 0.55
95% CI = [0.42, 0.68]
(Full prediction uncertainty!)
```

**This is what "sample for predictive distributions of EVC(u)" means!**

---

## Complete Implementation

### **Full Hierarchical Bayesian Generative Model:**

```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

class HierarchicalBayesianEVC_Generative:
    """
    EVC embedded in hierarchical Bayesian generative process
    
    Generative Story:
    1. Population has typical uncertainty weight (μ_λ)
    2. Each child draws their own λᵢ from population
    3. For each trial, child computes EVC with their λᵢ
    4. EVC determines control allocation
    5. Control generates observable outcomes (RT, accuracy)
    """
    
    def __init__(self):
        self.model = None
        self.trace = None
    
    def build_model(self, data):
        """
        Build hierarchical Bayesian generative model
        
        Args:
            data: DataFrame with columns:
                - child_id: Child identifier
                - reward: Reward magnitude
                - accuracy: Expected accuracy
                - uncertainty: Trial uncertainty
                - control_observed: Measured control (RT proxy)
                - rt: Reaction time (optional)
                - correct: Trial outcome (optional)
        """
        # Prepare data
        n_children = data['child_id'].nunique()
        child_idx = data['child_id'].astype('category').cat.codes.values
        
        reward = data['reward'].values
        accuracy = data['accuracy'].values  
        uncertainty = data['uncertainty'].values
        control_obs = data['control_observed'].values
        
        with pm.Model() as model:
            
            # ============================================
            # LEVEL 1: POPULATION HYPERPARAMETERS
            # ============================================
            
            # Uncertainty weight (KEY PARAMETER)
            mu_lambda = pm.Normal('mu_lambda', mu=0.5, sigma=0.3)
            sigma_lambda = pm.HalfNormal('sigma_lambda', sigma=0.2)
            
            # Reward sensitivity
            mu_beta_r = pm.Normal('mu_beta_r', mu=1.0, sigma=0.5)
            sigma_beta_r = pm.HalfNormal('sigma_beta_r', sigma=0.3)
            
            # Effort cost
            mu_beta_e = pm.Normal('mu_beta_e', mu=1.0, sigma=0.5)
            sigma_beta_e = pm.HalfNormal('sigma_beta_e', sigma=0.3)
            
            # Baseline control
            mu_baseline = pm.Normal('mu_baseline', mu=0.5, sigma=0.2)
            sigma_baseline = pm.HalfNormal('sigma_baseline', sigma=0.1)
            
            # ============================================
            # LEVEL 2: INDIVIDUAL PARAMETERS
            # ============================================
            
            # Each child's parameters (non-centered parameterization)
            lambda_offset = pm.Normal('lambda_offset', mu=0, sigma=1, shape=n_children)
            beta_r_offset = pm.Normal('beta_r_offset', mu=0, sigma=1, shape=n_children)
            beta_e_offset = pm.Normal('beta_e_offset', mu=0, sigma=1, shape=n_children)
            baseline_offset = pm.Normal('baseline_offset', mu=0, sigma=1, shape=n_children)
            
            # Transform to actual parameters
            lambda_i = pm.Deterministic('lambda', mu_lambda + lambda_offset * sigma_lambda)
            beta_r_i = pm.Deterministic('beta_r', mu_beta_r + beta_r_offset * sigma_beta_r)
            beta_e_i = pm.Deterministic('beta_e', mu_beta_e + beta_e_offset * sigma_beta_e)
            baseline_i = pm.Deterministic('baseline', mu_baseline + baseline_offset * sigma_baseline)
            
            # ============================================
            # LEVEL 3: EVC GENERATIVE PROCESS
            # ============================================
            
            # Expected value (traditional EVC component)
            expected_value = reward * accuracy
            
            # BAYESIAN EVC FORMULA (this is the generative model!)
            # This is how control is GENERATED from parameters
            control_predicted = (
                baseline_i[child_idx] +
                (beta_r_i[child_idx] * expected_value + 
                 lambda_i[child_idx] * uncertainty) / 
                (2 * beta_e_i[child_idx])
            )
            
            # Ensure valid range
            control_predicted = pm.math.clip(control_predicted, 0, 1)
            
            # ============================================
            # OBSERVATION MODEL
            # ============================================
            
            # Observed control comes from predicted control + noise
            sigma_obs = pm.HalfNormal('sigma_obs', sigma=0.2)
            
            control_likelihood = pm.Normal(
                'control_obs',
                mu=control_predicted,
                sigma=sigma_obs,
                observed=control_obs
            )
            
            # OPTIONAL: If you have RT and accuracy data, add them!
            # RT increases with control
            if 'rt' in data.columns:
                rt = data['rt'].values
                # RT = base_rt + control × rt_slope + noise
                base_rt = pm.Normal('base_rt', mu=800, sigma=200)
                rt_slope = pm.Normal('rt_slope', mu=500, sigma=100)
                sigma_rt = pm.HalfNormal('sigma_rt', sigma=100)
                
                rt_predicted = base_rt + control_predicted * rt_slope
                rt_likelihood = pm.Normal('rt_obs', mu=rt_predicted, 
                                         sigma=sigma_rt, observed=rt)
            
            # Accuracy increases with control
            if 'correct' in data.columns:
                correct = data['correct'].values
                # Accuracy = base_accuracy + control × accuracy_boost
                accuracy_boost = pm.Beta('accuracy_boost', alpha=2, beta=2)
                
                p_correct = pm.math.clip(
                    accuracy + control_predicted * accuracy_boost,
                    0.01, 0.99
                )
                
                acc_likelihood = pm.Bernoulli('acc_obs', p=p_correct, 
                                             observed=correct)
        
        self.model = model
        return model
    
    def fit(self, data, draws=2000, tune=1000, chains=4):
        """
        Fit model via MCMC
        
        Returns:
            trace: Posterior samples
        """
        if self.model is None:
            self.build_model(data)
        
        print("Sampling from posterior...")
        print(f"  Chains: {chains}")
        print(f"  Draws: {draws}")
        print(f"  Tuning: {tune}")
        
        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.95,
                return_inferencedata=True
            )
        
        # Check convergence
        print("\nConvergence diagnostics:")
        summary = az.summary(
            self.trace, 
            var_names=['mu_lambda', 'sigma_lambda', 'mu_beta_r', 'mu_beta_e']
        )
        print(summary[['mean', 'sd', 'r_hat', 'ess_bulk']])
        
        return self.trace
    
    def test_hypothesis(self):
        """
        Test primary hypothesis: Does uncertainty matter?
        
        H1: μ_λ > 0 (uncertainty increases control)
        """
        if self.trace is None:
            raise ValueError("Must fit model first!")
        
        # Extract posterior samples
        mu_lambda_samples = self.trace.posterior['mu_lambda'].values.flatten()
        
        # Test hypothesis
        p_positive = (mu_lambda_samples > 0).mean()
        
        print("\n" + "=" * 70)
        print("HYPOTHESIS TEST: Does Uncertainty Affect Control?")
        print("=" * 70)
        print(f"\nPopulation uncertainty weight (μ_λ):")
        print(f"  Mean: {mu_lambda_samples.mean():.3f}")
        print(f"  SD: {mu_lambda_samples.std():.3f}")
        print(f"  95% CI: [{np.percentile(mu_lambda_samples, 2.5):.3f}, "
              f"{np.percentile(mu_lambda_samples, 97.5):.3f}]")
        print(f"\nP(μ_λ > 0) = {p_positive:.4f}")
        
        if p_positive > 0.99:
            print("\n✓ STRONG EVIDENCE: Uncertainty significantly affects control!")
        elif p_positive > 0.95:
            print("\n✓ GOOD EVIDENCE: Uncertainty affects control")
        elif p_positive > 0.90:
            print("\n⚠ WEAK EVIDENCE: Uncertainty may affect control")
        else:
            print("\n✗ NO EVIDENCE: Uncertainty does not affect control")
        
        return p_positive
    
    def predict_new_child(self, new_child_data, n_samples=1000):
        """
        Predict control for a new child using population distribution
        
        This is what "sample for predictive distributions of EVC(u)" means!
        
        Args:
            new_child_data: Trials for new child (reward, accuracy, uncertainty)
            n_samples: Number of posterior predictive samples
            
        Returns:
            predictions: Distribution of predicted controls [n_samples, n_trials]
        """
        if self.trace is None:
            raise ValueError("Must fit model first!")
        
        # Sample child parameters from population distribution
        mu_lambda = self.trace.posterior['mu_lambda'].values.flatten()
        sigma_lambda = self.trace.posterior['sigma_lambda'].values.flatten()
        
        # Randomly sample posterior indices
        posterior_samples = np.random.choice(len(mu_lambda), size=n_samples)
        
        predictions = []
        
        for idx in posterior_samples:
            # Sample new child's lambda from population distribution
            lambda_new = np.random.normal(mu_lambda[idx], sigma_lambda[idx])
            
            # Similarly for other parameters
            mu_beta_r = self.trace.posterior['mu_beta_r'].values.flatten()[idx]
            sigma_beta_r = self.trace.posterior['sigma_beta_r'].values.flatten()[idx]
            beta_r_new = np.random.normal(mu_beta_r, sigma_beta_r)
            
            mu_beta_e = self.trace.posterior['mu_beta_e'].values.flatten()[idx]
            sigma_beta_e = self.trace.posterior['sigma_beta_e'].values.flatten()[idx]
            beta_e_new = np.random.normal(mu_beta_e, sigma_beta_e)
            
            mu_baseline = self.trace.posterior['mu_baseline'].values.flatten()[idx]
            sigma_baseline = self.trace.posterior['sigma_baseline'].values.flatten()[idx]
            baseline_new = np.random.normal(mu_baseline, sigma_baseline)
            
            # Predict using EVC formula
            expected_value = new_child_data['reward'].values * new_child_data['accuracy'].values
            
            control_pred = baseline_new + \
                (beta_r_new * expected_value + lambda_new * new_child_data['uncertainty'].values) / \
                (2 * beta_e_new)
            
            control_pred = np.clip(control_pred, 0, 1)
            predictions.append(control_pred)
        
        predictions = np.array(predictions)
        
        # Return full distribution
        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'lower_95': np.percentile(predictions, 2.5, axis=0),
            'upper_95': np.percentile(predictions, 97.5, axis=0),
            'samples': predictions
        }


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == '__main__':
    # Load data
    data = pd.read_csv('data/behavioral_data.csv')
    
    # Prepare for model
    data['control_observed'] = data['control_signal']  # What we measure
    
    # Initialize and fit
    model = HierarchicalBayesianEVC_Generative()
    trace = model.fit(data, draws=2000, tune=1000, chains=4)
    
    # Test hypothesis
    p_value = model.test_hypothesis()
    
    # Predict for new child
    new_child = data[data['child_id'] == 999]  # Hypothetical new child
    predictions = model.predict_new_child(new_child)
    
    print(f"\nPredictions for new child:")
    print(f"  Mean control: {predictions['mean'].mean():.3f}")
    print(f"  Uncertainty: {predictions['std'].mean():.3f}")
    
    # Visualize posterior predictive
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Posterior for μ_λ
    mu_lambda = trace.posterior['mu_lambda'].values.flatten()
    axes[0].hist(mu_lambda, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='λ = 0')
    axes[0].set_xlabel('Population Uncertainty Weight (μ_λ)')
    axes[0].set_ylabel('Posterior Density')
    axes[0].set_title('Posterior Distribution: Does Uncertainty Matter?')
    axes[0].legend()
    
    # Plot 2: Predictive distribution for new child
    trial_idx = 0
    axes[1].hist(predictions['samples'][:, trial_idx], bins=50, 
                edgecolor='black', alpha=0.7)
    axes[1].axvline(predictions['mean'][trial_idx], color='red', 
                   linewidth=2, label='Mean prediction')
    axes[1].set_xlabel('Predicted Control')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Predictive Distribution for New Child (Trial 1)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
```

---

## Key Concepts Explained

### **1. "Embed EVC in generative process"**

**Means:** EVC formula is the data-generating mechanism in your Bayesian model

```python
# NOT this (two-stage):
params = fit_evc(data)
predictions = evc_formula(params)

# THIS (generative):
with pm.Model():
    # EVC generates data
    control = evc_formula(parameters)  # Generative process
    observed = pm.Normal('obs', mu=control, observed=data)  # Likelihood
    trace = pm.sample()  # Infer parameters
```

---

### **2. "Infer hierarchical posteriors over EVC params"**

**Means:** Get full posterior distributions for λ, β_r, β_e at both levels

```python
# After sampling:
posterior_mu_lambda = trace.posterior['mu_lambda']  # Population
posterior_lambda_i = trace.posterior['lambda']      # Individual

# These are DISTRIBUTIONS, not point estimates!
```

---

### **3. "Sample for predictive distributions of EVC(u) or actions"**

**Means:** Generate predictions with full uncertainty

```python
# For new child or new trial:
# Sample from posterior
for sample in posterior_samples:
    lambda_new = sample_from_population(sample)
    control_pred = evc_formula(lambda_new, reward, uncertainty)
    predictions.append(control_pred)

# Result: Distribution of predictions (not single value!)
mean_prediction = predictions.mean()
uncertainty_in_prediction = predictions.std()
```

**This quantifies prediction uncertainty!**

---

## Why This is Better Than HDDM-EVC

### **Comparison:**

| Feature | HDDM-EVC Integration | EVC in HBM Generative |
|---------|---------------------|----------------------|
| **Complexity** | Very high (two models linked) | Moderate (one unified model) |
| **Circular dependencies** | Yes (EVC ↔ HDDM) | No (EVC → data) |
| **Parameters** | 9+ per person | 4 per person |
| **Interpretability** | Unclear (boundary = control?) | Clear (λ, β_r, β_e) |
| **Implementation** | Custom HDDM extensions | Standard PyMC |
| **Tests hypothesis** | Indirectly | Directly |
| **Feasibility** | High risk | Medium risk |

**Winner:** EVC in HBM Generative ✅

---

## What You Get

### **Advantages of This Approach:**

1. ✅ **Direct test of hypothesis**
   - Does λ > 0? (uncertainty matters)
   - Clear, unambiguous answer

2. ✅ **Full uncertainty quantification**
   - Not just "λ = 0.42"
   - But "λ ~ Posterior, 95% CI [0.28, 0.56], P(λ>0) = 0.998"

3. ✅ **Predictive distributions**
   - Predict for new children with uncertainty
   - "This child will likely have control = 0.55 ± 0.12"

4. ✅ **Model comparison**
   - Traditional EVC vs. Bayesian EVC
   - Hierarchical vs. pooled
   - Via WAIC, LOO

5. ✅ **Individual differences**
   - Extract λᵢ for each child
   - Identify outliers (high λ = anxious?)
   - Clinical relevance

6. ✅ **Population inference**
   - "Typical child has λ = 0.42"
   - "Children vary with σ_λ = 0.18"
   - Scientific generalization

---

## For Your Proposal

### **This IS What You Should Write:**

**Computational Modeling Section:**

> "We embed the Bayesian Expected Value of Control model within a hierarchical Bayesian generative framework. Our model specifies how control allocation arises from the EVC cost-benefit computation at the trial level, with individual child parameters (λᵢ, β_r,i, β_e,i) drawn from population-level distributions. This generative approach allows us to directly test our hypothesis via posterior inference: we estimate the full posterior distribution over the population uncertainty weight (μ_λ) and assess the probability that it exceeds zero, providing strong quantitative evidence for the role of uncertainty in control allocation.
>
> The model is fit via Markov Chain Monte Carlo (MCMC) using PyMC, yielding not only parameter estimates but complete posterior distributions that quantify our uncertainty about each parameter. This enables probabilistic hypothesis testing (e.g., P(μ_λ > 0 | data)) and generation of posterior predictive distributions for new children, addressing both scientific inference (does uncertainty matter for the typical child?) and clinical prediction (how will this specific child allocate control?).
>
> We validate the model through: (1) posterior predictive checks (does the model generate realistic data?), (2) leave-one-out cross-validation (out-of-sample prediction accuracy), and (3) model comparison against traditional EVC without uncertainty (via WAIC/LOO information criteria). We expect our Bayesian EVC model to substantially outperform traditional EVC (ΔWAIC > 20), demonstrating that incorporating uncertainty is essential for understanding control allocation."

**This is:**
- ✅ Sophisticated (shows you know generative modeling)
- ✅ Clear (reviewers can follow)
- ✅ Feasible (standard PyMC)
- ✅ Directly tests your hypothesis

---

## Implementation Status

### **What You Already Have:**

✅ The code structure (in `HIERARCHICAL_BAYES_GUIDE.md`)
✅ The theoretical framework (in gameplan)
✅ The simulated data (from step1)

### **What You Need to Do:**

```bash
# Week 1: Implement generative model in PyMC
# (Use code from this document)

# Week 2: Test on simulated data
python test_hierarchical_evc_generative.py

# Week 3: Validate and tune
# - Check convergence
# - Posterior predictive checks
# - Parameter recovery

# Week 4: Write up for proposal
# Use text provided above
```

**Timeline:** 3-4 weeks to fully implement and validate

---

## Bottom Line

### **The Question:** "Can we embed EVC in HBM generative process?"

**Answer:** ✅ **YES! And you SHOULD!**

**This is:**
- ✅ The right approach (generative modeling)
- ✅ Appropriate complexity (not too simple, not too complex)
- ✅ Directly tests your hypothesis
- ✅ Produces interpretable results
- ✅ Feasible to implement

### **This is MUCH better than HDDM-EVC integration because:**

- ✅ No circular dependencies
- ✅ Fewer parameters (4 vs. 9 per person)
- ✅ Direct (not indirect via DDM)
- ✅ Clear interpretation
- ✅ Standard implementation (no custom HDDM hacks)

---

## **Final Recommendation:**

**Yes, implement this!** This is your winning approach:

**Hierarchical Bayesian Generative Model with EVC**

- Use the code I provided above
- Use the proposal text I drafted
- This is exactly right for your proposal

**NOT:** HDDM-EVC integration (too complex, indirect)
**YES:** EVC embedded in HBM generative process (perfect!)

**This is proposal-ready!** 🎯✨

Want me to create a complete implementation script and add it to your project?
