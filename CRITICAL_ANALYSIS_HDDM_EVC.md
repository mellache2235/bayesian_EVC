# Critical Analysis: HDDM-EVC Integration Proposal

## The Proposal Received

Someone proposed using HDDM to "implement" Bayesian EVC by:
1. Fitting hierarchical DDM to trial data
2. Regressing DDM parameters (drift rate v, boundary a) on EVC values
3. Computing EVC from uncertainty estimates
4. Linking control to DDM threshold modulation

**This is intellectually sophisticated but raises serious concerns for a proposal.**

---

## Critical Assessment

### **The Core Issue:**

**HDDM and EVC are fundamentally different types of models:**

```
DDM/HDDM:
- Process model (HOW decisions are made)
- Models: Evidence accumulation mechanics
- Predicts: RT distributions, choice accuracy
- Parameters: v (drift), a (boundary), t₀ (non-decision time)

EVC:
- Normative model (WHEN/WHY to exert control)
- Models: Cost-benefit optimization
- Predicts: Control allocation
- Parameters: λ (uncertainty weight), β_r (reward), β_e (cost)
```

**Mixing them requires:**
- Theoretical justification for how DDM parameters = control
- Custom HDDM extensions (not standard)
- Computational expertise (PyMC internals)
- Validation that the link makes sense

---

## What the Commenter Gets Right

### **Valid Points:**

✅ **"HDDM won't seamlessly implement EVC"**
- Correct! They're different frameworks
- Integration requires custom work
- Not plug-and-play

✅ **"Computational complexity amplified"**
- HDDM with MCMC is slow (~hours for N=80)
- Child data has high variability (convergence issues)
- Need strong priors and expertise

✅ **"Developmental HDDM applications are limited"**
- True - mostly adult studies
- Children have high variability (attention lapses, fatigue)
- May need age-specific adaptations

✅ **"State uncertainty better suited to RL/belief models, not DDM"**
- DDM handles decision uncertainty well
- State/rule uncertainty needs different framework (HGF, Kalman)
- Mixing both is complex

✅ **"Risk of mismatched analysis"**
- Using HDDM for wrong type of uncertainty
- Diluting both approaches
- Neither works optimally

---

## What the Commenter Gets Wrong

### **Overly Pessimistic:**

⚠️ **"Don't give false impression of adeptness"**
- You CAN learn HDDM (well-documented)
- PyMC has good community support
- But: Takes time (2-3 months to master)

⚠️ **"Could bloat proposal without proportional gains"**
- True for initial proposal
- But valuable for follow-up work
- Not "bloat" if well-justified

---

## My Honest Assessment

### **For Your CURRENT Proposal:**

**❌ DO NOT use HDDM-EVC integration**

**Reasons:**

1. **Too complex for reviewers**
   - Requires understanding DDM AND EVC AND integration
   - Reviewers might reject as "overly ambitious"
   - "Why not just use one model?"

2. **Theoretical mismatch**
   - DDM models decisions (millisecond timescale)
   - EVC models control (trial/block timescale)
   - Linking them is non-trivial

3. **Implementation risk**
   - Custom HDDM extensions needed
   - Not standard off-the-shelf
   - 3-6 months just for implementation
   - High risk of technical failure

4. **Doesn't solve your core question**
   - You want to test: "Does uncertainty affect control?"
   - Simple answer: Fit Bayesian EVC with λ parameter
   - HDDM integration adds complexity without adding clarity

5. **Sample size concerns**
   - HDDM works best with 100+ trials per person
   - Children have higher variability (attention, fatigue)
   - Convergence may be poor
   - Proposal already has small N challenges

---

### **What You SHOULD Propose:**

**✅ Hierarchical Bayesian EVC (simple, direct)**

```python
# Clear, interpretable model
with pm.Model() as hierarchical_evc:
    # Population
    mu_lambda = pm.Normal('mu_lambda', mu=0.5, sigma=0.3)
    sigma_lambda = pm.HalfNormal('sigma_lambda', sigma=0.2)
    
    # Individual
    lambda_i = pm.Normal('lambda', mu=mu_lambda, sigma=sigma_lambda, shape=n_children)
    
    # Trial
    predicted_control = baseline + (reward * accuracy + lambda_i * uncertainty) / (2 * cost)
    control_obs = pm.Normal('control_obs', mu=predicted_control, sigma=sigma_obs,
                           observed=observed_control)
    
    # Sample
    trace = pm.sample(2000, tune=1000, chains=4)

# Test hypothesis
P_lambda_positive = (trace.posterior['mu_lambda'] > 0).mean()
```

**Why this is better:**
- ✅ Directly tests your hypothesis (λ > 0?)
- ✅ Interpretable (everyone understands λ)
- ✅ Feasible (standard PyMC, well-documented)
- ✅ Appropriate complexity for proposal
- ✅ Clear contribution (extends EVC with uncertainty)

---

## The Proposed HDDM Text: Analysis

### **What the Proposed Text Says:**

1. Use HDDM to fit DDM parameters
2. Compute EVC from uncertainty
3. Regress DDM parameters on EVC values
4. Link control to DDM threshold (a)

### **Problems with This:**

#### **Problem 1: Circular Logic**

```
EVC depends on uncertainty
    ↓
Uncertainty estimated from DDM
    ↓
DDM parameters regressed on EVC
    ↓
Which depends on DDM uncertainty...
```

This is circular! You need uncertainty to compute EVC, but you're using EVC to predict DDM parameters that estimate uncertainty.

---

#### **Problem 2: What is Control?**

**Proposed:** `a_t = a_0 + γ × c_t` (boundary = control)

**Issues:**
- Is boundary separation really "control"?
- High boundary = caution, not necessarily effort
- Control could be drift rate (faster processing)
- Or non-decision time (more attention)
- **Unclear mapping!**

---

#### **Problem 3: Too Many Parameters**

**HDDM parameters (per person):**
- v₀, δ (drift rate intercept + uncertainty effect)
- a₀, γ (boundary intercept + control effect)
- t₀ (non-decision time)
- **= 5 parameters**

**EVC parameters (per person):**
- λ (uncertainty weight)
- β_r (reward sensitivity)
- β_e (effort cost)
- baseline
- **= 4 parameters**

**Total:** 9 parameters per person × 80 children = 720+ parameters!

**This is massive overfitting risk!**

---

#### **Problem 4: What Question Does This Answer?**

**HDDM-EVC integration asks:**
> "How does uncertainty-weighted control (from EVC) affect decision boundary and drift rate (in DDM)?"

**But your actual question is:**
> "Does uncertainty affect control allocation?"

**These are different questions!** The integration doesn't directly test your hypothesis.

---

## What You Should Actually Do

### **Option A: Pure Hierarchical Bayesian EVC** ✅ RECOMMENDED

**Directly tests your hypothesis:**

```python
# Simple, interpretable
Control = baseline + (β_r × Reward × Accuracy + λ × Uncertainty) / (2 × β_e)

# Test: Is λ > 0?
# Answer: Yes/No, with probability
```

**Advantages:**
- ✅ Direct test of hypothesis
- ✅ Interpretable λ parameter
- ✅ Appropriate complexity
- ✅ Feasible implementation

**Use the text already in your gameplan!** It's perfect.

---

### **Option B: Mention HDDM as Future Work**

**In proposal:**

> "While the current proposal focuses on direct modeling of control allocation via Bayesian EVC, future extensions could integrate drift-diffusion models to decompose the cognitive processes underlying control. Specifically, HDDM could estimate trial-by-trial decision uncertainty from RT distributions, which could then be used as input to our Bayesian EVC framework. This integration would provide a more mechanistic account of how uncertainty estimates arise from evidence accumulation processes."

**This shows sophistication WITHOUT overcommitting.**

---

### **Option C: Simple DDM (Not Hierarchical)**

**If you want DDM but simpler:**

```python
# Fit simple DDM per child (not hierarchical)
# Extract drift rates
# Use as uncertainty proxy in Bayesian EVC

for child in children:
    # Fit DDM to this child's RT/choice
    ddm_params = fit_ddm(child_data)
    
    # Drift rate → uncertainty
    uncertainty[child] = 1 / (1 + ddm_params['drift_rate'])
    
    # Use in EVC
    control = bayesian_evc.predict(reward, accuracy, uncertainty)
```

**Advantages:**
- ✅ Simpler than HDDM (no hierarchy)
- ✅ Still uses DDM for uncertainty
- ✅ Doesn't require PyMC expertise
- ✅ Faster implementation

**This could work, but still adds complexity.**

---

## My Strong Recommendation

### **For Your Proposal: Do NOT use HDDM integration**

**Why:**

1. **Complexity-risk ratio is poor**
   - High complexity (9 parameters/person, circular dependencies)
   - High risk (convergence, interpretation, validation)
   - Modest benefit (doesn't directly test hypothesis)

2. **Doesn't match your core question**
   - Your question: "Does uncertainty affect control?"
   - HDDM-EVC: "How does EVC-predicted control affect DDM parameters?"
   - These are different!

3. **Timeline concerns**
   - HDDM implementation: 2-3 months
   - Validation: 1-2 months
   - Debugging: Unknown
   - **Risk to proposal timeline**

4. **Expertise requirements**
   - Need DDM expertise
   - Need HDDM/PyMC expertise
   - Need to understand linking
   - **This is a lot to learn!**

5. **Reviewer concerns**
   - "Why so complex?"
   - "Can you really implement this?"
   - "Is this necessary for your hypothesis?"
   - **Higher rejection risk**

---

## What to Write in Your Proposal Instead

### **Recommended Computational Modeling Section:**

```
Computational Modeling

To test our hypothesis that uncertainty affects cognitive control allocation, 
we employ hierarchical Bayesian modeling of an extended Expected Value of 
Control (EVC) framework. Our model estimates both population-level parameters 
(representing typical children) and individual-level parameters (capturing 
heterogeneity across children), using partial pooling to optimize estimates 
with our realistic sample size (N=40-50 children, 100-200 trials each).

Model Specification:

At the population level, we estimate the mean uncertainty weight (μ_λ), 
which quantifies how much uncertainty affects control allocation in the 
average child, and its between-child variability (σ_λ). Each child i receives 
their own uncertainty weight λᵢ ~ Normal(μ_λ, σ_λ), along with individual 
reward sensitivity (β_r,i), effort cost (β_e,i), and baseline control 
(baseline_i) parameters.

For each trial t, predicted control allocation is computed via the Bayesian 
EVC formula:

    Control[i,t] = baseline[i] + 
        (β_r[i] × Reward[t] × Accuracy[t] + λ[i] × Uncertainty[t]) / (2 × β_e[i])

where Uncertainty[t] combines decision uncertainty (inverse of evidence clarity, 
potentially estimated via confidence ratings or stimulus ambiguity) and state 
uncertainty (estimated via Bayesian belief updating over task rules).

Inference is conducted via Markov Chain Monte Carlo (MCMC) using PyMC (version 5.0+), 
with No-U-Turn Sampling (NUTS) for efficient exploration of the posterior. We run 
4 chains with 2,000 samples each after 1,000 burn-in samples, assessing convergence 
via Gelman-Rubin statistics (R̂ < 1.01). This yields full posterior distributions 
for all parameters, enabling probabilistic hypothesis testing (e.g., P(μ_λ > 0 | data)).

We validate the model via: (1) posterior predictive checks (can the model generate 
realistic data?), (2) leave-one-out cross-validation (out-of-sample prediction), 
and (3) parameter recovery simulations (can we recover known parameters from 
simulated data?). We compare our Bayesian EVC model against traditional EVC 
(without uncertainty term) using WAIC and LOO information criteria.

This approach directly tests our central hypothesis while maintaining 
interpretability—each parameter has clear psychological meaning—and feasibility 
for our sample size through principled hierarchical estimation.
```

**This is:**
- ✅ Appropriate complexity
- ✅ Directly tests your hypothesis
- ✅ Feasible to implement
- ✅ Reviewers can understand
- ✅ No circular dependencies

---

## Summary

### **The HDDM-EVC Integration Proposal:**

**Pros:**
- ✅ Intellectually sophisticated
- ✅ Shows advanced knowledge
- ✅ Could work in theory

**Cons:**
- ❌ Too complex for initial proposal
- ❌ Circular dependencies (EVC needs uncertainty from DDM, DDM depends on EVC)
- ❌ Unclear mapping (is boundary = control?)
- ❌ 9 parameters/person (overfitting risk)
- ❌ Doesn't directly test your hypothesis
- ❌ High implementation risk
- ❌ Reviewer concern: "Is this necessary?"

---

### **My Recommendation:**

**For your proposal:**

🥇 **Use simple Hierarchical Bayesian EVC** (what's already in your gameplan)
- Direct test of hypothesis
- Clear interpretation
- Feasible implementation
- Appropriate complexity

📋 **Mention HDDM in future work:**
> "Future extensions could integrate drift-diffusion models to estimate trial-level decision uncertainty, providing a mechanistic account of how uncertainty arises from evidence accumulation processes."

**This shows sophistication without overcommitting.**

---

### **Save HDDM-EVC Integration for:**

- Paper 2 or 3 (after establishing basic framework)
- Advanced grant (R01 with larger budget)
- Collaboration with DDM expert
- When you have 2-3 months dedicated to implementation

---

## Bottom Line

**The HDDM-EVC text provided is too complex for your initial proposal.**

**It would:**
- ❌ Confuse reviewers
- ❌ Raise feasibility concerns
- ❌ Add risk without proportional benefit
- ❌ Take focus away from your core contribution (λ parameter)

**Stick with the clean hierarchical Bayesian EVC already in your gameplan!** ✅

That section is proposal-ready, clear, and directly tests your hypothesis. **Don't overcomplicate!**

Want me to polish the existing modeling section in your gameplan instead of adding HDDM complexity?


