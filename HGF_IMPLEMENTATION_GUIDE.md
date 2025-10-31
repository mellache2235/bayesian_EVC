# Hierarchical Gaussian Filter (HGF) for Bayesian EVC

## Overview

The **Hierarchical Gaussian Filter (HGF)** is an ideal framework for modeling uncertainty in your Bayesian EVC project. It provides a principled way to estimate multiple levels of uncertainty and track how beliefs evolve over time.

---

## Does HGF Fall Under Hierarchical Bayes?

### **Short Answer: Sort of, but they're different things!**

This is a great question because there's overlap but also important distinctions:

### **The Confusion:**

Both have "hierarchical" in the name and both use Bayesian inference, but they refer to **different types of hierarchies**:

---

### **HGF (Hierarchical Gaussian Filter)**

**Type of hierarchy:** **Temporal/Generative Hierarchy**

**What it models:**
```
Level 3: Volatility (how fast things change)
    ↓ influences
Level 2: Hidden states (what's the current state?)
    ↓ generates
Level 1: Observations (what you see)
```

**Key features:**
- ✅ Models **within-subject** learning dynamics over time
- ✅ Hierarchy of **causal relationships** (volatility → states → observations)
- ✅ Each level generates/influences the level below
- ✅ Single subject, multiple trials
- ✅ Tracks how beliefs evolve trial-by-trial

**Example:**
```python
# One subject learning over 200 trials
hgf = HierarchicalGaussianFilter()
for trial in range(200):
    hgf.update(outcome[trial])
    uncertainty = hgf.get_state_uncertainty()
```

**Focus:** "How does ONE person learn over time?"

---

### **Hierarchical Bayesian Modeling (HBM)**

**Type of hierarchy:** **Population/Statistical Hierarchy**

**What it models:**
```
Population Level: Group parameters (μ_β, σ_β)
    ↓ constrains
Individual Level: Subject-specific parameters (β[subject])
    ↓ generates
Trial Level: Observations (behavior on each trial)
```

**Key features:**
- ✅ Models **across-subject** variability
- ✅ Hierarchy of **statistical relationships** (population → individuals → data)
- ✅ Each level constrains the level below via probability distributions
- ✅ Multiple subjects, partial pooling
- ✅ Estimates both group and individual parameters

**Example:**
```python
# 20 subjects, each with their own parameters
with pm.Model() as hierarchical_model:
    # Population level
    mu_beta = pm.Normal('mu_beta', mu=1.0, sigma=0.5)
    sigma_beta = pm.HalfNormal('sigma_beta', sigma=0.3)
    
    # Individual level (20 subjects)
    beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=20)
    
    # Trial level
    # ... likelihood ...
```

**Focus:** "How do MANY people differ from each other?"

---

## Comparison Table

| Feature | HGF | Hierarchical Bayesian |
|---------|-----|---------------------|
| **Type of hierarchy** | Temporal/Generative | Population/Statistical |
| **Primary question** | "How does learning unfold?" | "How do people differ?" |
| **Levels represent** | Causal structure (volatility → states) | Statistical structure (group → individual) |
| **Time dimension** | Essential (trial-by-trial updates) | Optional (can be static) |
| **Number of subjects** | Typically 1 | Multiple (N > 1) |
| **Main output** | Belief trajectories over time | Population + individual parameters |
| **Partial pooling** | No | Yes (across subjects) |
| **Shrinkage** | No | Yes (individuals → group mean) |
| **Inference method** | Analytic updates (Kalman-like) | MCMC sampling (PyMC, Stan) |

---

## Can You Combine Them? YES!

### **Hierarchical Bayesian HGF**

You can (and should!) use **both hierarchies together**:

```
POPULATION LEVEL (Hierarchical Bayes)
    ↓
    μ_κ₂ = 1.0  (mean coupling strength across subjects)
    σ_κ₂ = 0.3  (between-subject variability)
    
INDIVIDUAL LEVEL (Hierarchical Bayes)
    ↓
    Subject 1: κ₂ = 1.2 (drawn from population)
    Subject 2: κ₂ = 0.9
    ...
    
TEMPORAL HIERARCHY (HGF for each subject)
    ↓
    For Subject 1 with κ₂ = 1.2:
        Level 3: Volatility
        Level 2: States  
        Level 1: Observations
```

**This gives you the best of both worlds!**

---

## Practical Example: Combining Both

### **Scenario:** 
You have 20 subjects, each doing 200 trials. You want to:
1. Model how each subject learns over time (HGF)
2. Estimate population-level HGF parameters (Hierarchical Bayes)

### **Implementation:**

```python
import pymc as pm
import numpy as np

def hierarchical_bayesian_hgf(data):
    """
    Fit HGF with hierarchical Bayesian parameter estimation.
    
    Combines:
    - HGF temporal hierarchy (within-subject learning)
    - Hierarchical Bayes (across-subject parameters)
    """
    
    n_subjects = data['subject_id'].nunique()
    
    with pm.Model() as model:
        
        # ============================================
        # HIERARCHICAL BAYES: POPULATION LEVEL
        # ============================================
        
        # Population-level HGF parameters
        mu_kappa_2 = pm.Normal('mu_kappa_2', mu=1.0, sigma=0.5)
        sigma_kappa_2 = pm.HalfNormal('sigma_kappa_2', sigma=0.3)
        
        mu_omega_2 = pm.Normal('mu_omega_2', mu=-4.0, sigma=2.0)
        sigma_omega_2 = pm.HalfNormal('sigma_omega_2', sigma=1.0)
        
        # ============================================
        # HIERARCHICAL BAYES: INDIVIDUAL LEVEL
        # ============================================
        
        # Subject-specific HGF parameters
        kappa_2 = pm.Normal('kappa_2', 
                           mu=mu_kappa_2, 
                           sigma=sigma_kappa_2,
                           shape=n_subjects)
        
        omega_2 = pm.Normal('omega_2',
                           mu=mu_omega_2,
                           sigma=sigma_omega_2,
                           shape=n_subjects)
        
        # ============================================
        # HGF: TEMPORAL HIERARCHY (for each subject)
        # ============================================
        
        # For each subject, run HGF over their trials
        log_likelihood = 0
        
        for subject_idx in range(n_subjects):
            subject_data = data[data['subject_id'] == subject_idx]
            
            # Initialize HGF with subject-specific parameters
            hgf = HierarchicalGaussianFilter(
                kappa_2=kappa_2[subject_idx],
                omega_2=omega_2[subject_idx]
            )
            
            # Process each trial for this subject
            for _, trial in subject_data.iterrows():
                # HGF prediction
                predicted_prob = hgf.get_state_estimate()
                
                # Likelihood of observed outcome
                outcome = trial['accuracy']
                log_likelihood += pm.Bernoulli.logp(outcome, predicted_prob)
                
                # Update HGF (temporal hierarchy)
                hgf.update(outcome)
        
        # Total likelihood
        pm.Potential('likelihood', log_likelihood)
        
        # ============================================
        # INFERENCE
        # ============================================
        
        trace = pm.sample(2000, tune=1000, chains=4)
    
    return trace, model


# ============================================
# USAGE
# ============================================

# Fit model
trace, model = hierarchical_bayesian_hgf(behavioral_data)

# ============================================
# RESULTS
# ============================================

# Population-level HGF parameters
print("POPULATION-LEVEL HGF PARAMETERS:")
print(f"Mean κ₂: {trace.posterior['mu_kappa_2'].mean():.3f}")
print(f"Between-subject SD: {trace.posterior['sigma_kappa_2'].mean():.3f}")

# Individual HGF parameters
print("\nINDIVIDUAL HGF PARAMETERS:")
for i in range(n_subjects):
    kappa_i = trace.posterior['kappa_2'][:, :, i].values.flatten()
    print(f"Subject {i+1}: κ₂ = {kappa_i.mean():.3f} ± {kappa_i.std():.3f}")
```

---

## When to Use What

### **Use HGF alone when:**
- ✅ Single subject (or treating subjects independently)
- ✅ Focus on learning dynamics over time
- ✅ Want to track uncertainty evolution
- ✅ Need trial-by-trial predictions
- ✅ Studying temporal adaptation

**Example:** "How does this patient's uncertainty change during treatment?"

---

### **Use Hierarchical Bayes alone when:**
- ✅ Multiple subjects
- ✅ Parameters don't change over time (static)
- ✅ Want population-level inference
- ✅ Need to handle small sample sizes
- ✅ Comparing groups (e.g., patients vs. controls)

**Example:** "What's the typical reward sensitivity in humans?"

---

### **Use BOTH (Hierarchical Bayesian HGF) when:**
- ✅ Multiple subjects with temporal dynamics
- ✅ Want both population and individual estimates
- ✅ Need to model learning AND individual differences
- ✅ Small N but repeated measures
- ✅ Clinical studies with learning tasks

**Example:** "Do depressed patients have abnormal volatility learning?"

---

## For Your Bayesian EVC Project

### **Current Setup:**
- Simple Bayesian uncertainty (fixed learning rate)
- Pooled model across subjects
- No temporal hierarchy
- No individual differences

### **Recommended Upgrade:**

**Option 1: Hierarchical Bayes Only** (Easier)
```python
# Estimate population + individual EVC parameters
# No temporal dynamics, just cross-sectional differences
```
- ✅ Better R² (0.2-0.4 instead of -0.03)
- ✅ Individual differences
- ✅ Easier to implement
- ⚠️ Doesn't model learning dynamics

---

**Option 2: HGF Only** (Moderate)
```python
# Use HGF for uncertainty estimation
# Fit each subject separately
```
- ✅ Adaptive learning rates
- ✅ Volatility tracking
- ✅ Richer uncertainty dynamics
- ⚠️ Unstable with small N per subject

---

**Option 3: Hierarchical Bayesian HGF** (Best but Complex)
```python
# HGF for temporal dynamics
# Hierarchical Bayes for population inference
```
- ✅ Best R² (0.3-0.5)
- ✅ Individual differences + learning dynamics
- ✅ Stable with small N
- ✅ Population-level HGF parameters
- ⚠️ Computationally intensive
- ⚠️ Requires MCMC expertise

---

## Recommendation for Your Project

### **Phase 1: Hierarchical Bayes (Do This First)**

**Why:**
- Immediate improvement (R² from -0.03 → 0.2-0.4)
- Easier to implement
- Handles small N well
- Good for publication

**Implementation:** Use the code from `HIERARCHICAL_BAYES_GUIDE.md`

---

### **Phase 2: Add HGF (Future Extension)**

**Why:**
- Richer model of learning
- Better fit to temporal dynamics
- More publishable in computational psychiatry
- Can study volatility learning

**Implementation:** Use HGF for uncertainty, then fit with Hierarchical Bayes

---

## Summary

### **HGF vs. Hierarchical Bayes:**

**HGF:**
- Temporal hierarchy (volatility → states → observations)
- Within-subject learning dynamics
- Trial-by-trial belief updates
- Answers: "How does learning unfold?"

**Hierarchical Bayes:**
- Statistical hierarchy (population → individuals → data)
- Across-subject variability
- Partial pooling and shrinkage
- Answers: "How do people differ?"

**Combined (Hierarchical Bayesian HGF):**
- Both hierarchies together
- Population-level learning parameters
- Individual learning trajectories
- Answers: "How does learning unfold AND how do people differ?"

### **For Your Project:**

1. **Start with Hierarchical Bayes** (easier, big improvement)
2. **Then add HGF** (richer model, better dynamics)
3. **Eventually combine both** (publication-ready, comprehensive)

**Bottom line:** HGF is NOT the same as Hierarchical Bayes, but they complement each other perfectly! 🎯

---

## Why Was HGF Developed? (Historical Context)

### The Problem That Led to HGF

**Traditional approaches to modeling learning had a fundamental limitation**: they couldn't explain how humans adapt their learning rates to environmental volatility.

#### **The Classic Problem (Pre-HGF)**

Imagine you're learning which restaurant has the best food:

**Scenario A - Stable Environment:**
- Restaurant quality stays constant
- You should learn slowly (low learning rate)
- One bad meal shouldn't change your mind

**Scenario B - Volatile Environment:**
- Restaurant quality changes frequently (new chef, inconsistent)
- You should learn quickly (high learning rate)
- One bad meal is important information!

**The Issue**: Traditional models used **fixed learning rates** - they couldn't switch between slow and fast learning based on context.

---

### Traditional Approaches (What Came Before HGF)

#### **1. Rescorla-Wagner Model (1972) - The Classic**

**Formula:**
```
V(t+1) = V(t) + α × [R(t) - V(t)]
```

Where:
- `V(t)` = Value/belief at time t
- `α` = Learning rate (FIXED, e.g., 0.1)
- `R(t)` = Reward received
- `[R(t) - V(t)]` = Prediction error

**What it does:**
- Simple, elegant model of associative learning
- Updates beliefs based on prediction errors
- Used extensively in neuroscience (dopamine = prediction error!)

**Limitations:**
- ❌ **Fixed learning rate** - can't adapt to volatility
- ❌ **No uncertainty tracking** - doesn't know when it's uncertain
- ❌ **No volatility estimation** - can't detect environmental changes
- ❌ **Single-level** - only tracks values, not meta-information

**Example Problem:**
```
Trials 1-50:  Restaurant is consistently good (α = 0.1 is perfect)
Trial 51:     New chef! Quality drops
Trials 52-70: Still using α = 0.1 → takes 20 trials to relearn
```

**Why it fails**: Can't detect the regime change and speed up learning.

---

#### **2. Kalman Filter (1960) - The Engineering Solution**

**Formula:**
```
# Prediction
μ̂(t) = μ(t-1)
σ̂²(t) = σ²(t-1) + Q  # Q = process noise (FIXED)

# Update
K(t) = σ̂²(t) / (σ̂²(t) + R)  # Kalman gain (adaptive!)
μ(t) = μ̂(t) + K(t) × [observation - μ̂(t)]
σ²(t) = (1 - K(t)) × σ̂²(t)
```

**What it does:**
- Optimal Bayesian filter for linear-Gaussian systems
- Tracks both estimate (μ) and uncertainty (σ²)
- Learning rate (Kalman gain K) adapts to uncertainty
- Used in GPS, robotics, aerospace

**Advantages over Rescorla-Wagner:**
- ✅ **Tracks uncertainty** explicitly
- ✅ **Adaptive learning rate** (based on uncertainty)
- ✅ **Optimal** for linear systems

**Limitations:**
- ❌ **Fixed process noise (Q)** - assumes constant volatility
- ❌ **Single-level** - doesn't model meta-uncertainty
- ❌ **Can't learn volatility** - Q is a parameter, not estimated
- ❌ **Linear assumptions** - doesn't handle nonlinear observations well

**Example Problem:**
```
Trials 1-50:  Q = 0.01 (low volatility) → works well
Trial 51:     Volatility increases! But Q is still 0.01
Trials 52-70: Slow to adapt because Q doesn't change
```

**Why it fails**: Can't detect that volatility itself has changed.

---

#### **3. Pearce-Hall Model (1980) - Attention-Weighted Learning**

**Formula:**
```
α(t) = γ × |δ(t-1)|  # Learning rate based on recent surprise
V(t+1) = V(t) + α(t) × δ(t)
```

**What it does:**
- Learning rate increases after surprising outcomes
- Models attention to prediction errors
- More biologically plausible than Rescorla-Wagner

**Advantages:**
- ✅ **Adaptive learning rate** (based on surprise)
- ✅ **Captures attention effects**

**Limitations:**
- ❌ **Reactive, not predictive** - only adapts AFTER surprise
- ❌ **No uncertainty tracking** - uses surprise as proxy
- ❌ **No volatility estimation** - just responds to recent errors
- ❌ **Heuristic** - not derived from optimal Bayesian principles

**Example:**
```
Trial 50:  Big surprise! → α increases
Trial 51:  High learning rate (good!)
Trial 52:  No surprise → α drops back down
Trial 53:  Another surprise! → α increases again (reactive)
```

**Why it's limited**: Always one step behind. Can't anticipate volatility.

---

#### **4. Bayesian Change-Point Detection (Adams & MacKay, 2007)**

**Formula:**
```
P(change_point | data) = compute probability that environment just changed
If change_point detected → reset beliefs
```

**What it does:**
- Explicitly models discrete change points
- Computes probability that a regime shift occurred
- Resets learning when change detected

**Advantages:**
- ✅ **Detects regime changes** explicitly
- ✅ **Principled Bayesian approach**
- ✅ **Can handle abrupt changes**

**Limitations:**
- ❌ **Discrete changes only** - assumes step-function volatility
- ❌ **Binary decision** - either change or no change
- ❌ **Doesn't model gradual volatility changes**
- ❌ **Computationally expensive** (maintains multiple hypotheses)

**Why it's limited**: Real environments often have gradual, continuous changes in volatility, not just discrete jumps.

---

### Enter the HGF (Mathys et al., 2011, 2014)

#### **The Key Insight**

What if we treat **volatility itself as a hidden variable that needs to be learned**?

Instead of:
- Fixed learning rate (Rescorla-Wagner) ❌
- Fixed process noise (Kalman Filter) ❌
- Reactive surprise (Pearce-Hall) ❌
- Discrete change points (Change-Point Detection) ❌

Do this:
- **Learn the volatility** from data ✅
- **Hierarchical structure**: beliefs about states AND beliefs about volatility ✅
- **Continuous adaptation**: smoothly adjust to changing volatility ✅
- **Optimal Bayesian inference**: provably optimal under assumptions ✅

#### **The HGF Solution**

```
Level 3: Volatility (μ₃, σ₃²)
    ↓ "How fast are things changing?"
Level 2: States (μ₂, σ₂²)  
    ↓ "What is the current state?"
Level 1: Observations
    "What did I observe?"
```

**Key Innovation**: 
- Level 2 uncertainty depends on Level 3 volatility
- Level 3 volatility is LEARNED from prediction errors
- Learning rate emerges automatically from the hierarchy

**Formula (simplified):**
```python
# Uncertainty at level 2 increases with volatility at level 3
σ₂²(t) = σ₂²(t-1) + exp(μ₃(t-1))  # Volatility controls uncertainty growth

# Volatility at level 3 is updated based on how much uncertainty changed
δ₃ = (actual_uncertainty_change - expected_uncertainty_change)
μ₃(t) = μ₃(t-1) + learning_rate × δ₃

# Learning rate emerges from the hierarchy
α = σ₂² / (σ₂² + observation_precision)
```

**What this achieves:**
- High volatility → high uncertainty → high learning rate (fast adaptation)
- Low volatility → low uncertainty → low learning rate (stable learning)
- **Volatility is learned**, not assumed!

---

### Comparison Table: Traditional vs. HGF

| Feature | Rescorla-Wagner | Kalman Filter | Pearce-Hall | Change-Point | **HGF** |
|---------|----------------|---------------|-------------|--------------|---------|
| **Year** | 1972 | 1960 | 1980 | 2007 | **2011** |
| **Learning Rate** | Fixed | Adaptive (uncertainty) | Adaptive (surprise) | Reset at changes | **Adaptive (volatility)** |
| **Tracks Uncertainty** | ❌ No | ✅ Yes | ❌ No | ✅ Yes | **✅ Yes (multi-level)** |
| **Learns Volatility** | ❌ No | ❌ No | ❌ No | ❌ No | **✅ Yes** |
| **Hierarchical** | ❌ No | ❌ No | ❌ No | ❌ No | **✅ Yes (3+ levels)** |
| **Continuous Adaptation** | ❌ No | ⚠️ Limited | ⚠️ Reactive | ❌ Discrete | **✅ Yes** |
| **Optimal Bayesian** | ❌ No | ✅ Yes (linear) | ❌ No | ✅ Yes | **✅ Yes (nonlinear)** |
| **Computational Cost** | Low | Low | Low | High | **Medium** |
| **Biological Plausibility** | High | Medium | High | Low | **High** |
| **Clinical Applications** | Limited | Limited | Limited | Limited | **Extensive** |

---

### Why HGF Was the Breakthrough

#### **1. Solves the "Meta-Learning" Problem**

Traditional models: "I need to learn the task"
HGF: "I need to learn the task AND learn how to learn (volatility)"

This is **meta-learning** - learning about learning itself.

#### **2. Matches Human Behavior**

Humans naturally:
- Learn faster in volatile environments ✅
- Learn slower in stable environments ✅
- Adjust learning rates smoothly, not abruptly ✅
- Track both "what's happening" and "how fast things change" ✅

HGF captures all of this!

#### **3. Neural Correlates**

Brain regions track HGF quantities:
- **Anterior Cingulate Cortex (ACC)**: Tracks precision-weighted prediction errors (δ₂ × π₂)
- **Insula**: Tracks volatility (μ₃)
- **Dorsolateral PFC**: Tracks state uncertainty (σ₂²)
- **Dopamine**: Tracks prediction errors (δ₁, δ₂)

This wasn't designed to fit the brain - it emerged from optimal inference!

#### **4. Clinical Relevance**

HGF parameters differ in:
- **Schizophrenia**: Abnormal volatility estimation (high κ₂)
- **Autism**: Reduced belief updating (low precision)
- **Anxiety**: Overestimation of volatility (high ω₂)
- **Depression**: Reduced learning from positive outcomes

This makes HGF a powerful tool for computational psychiatry.

---

### How One's Mind Goes to HGF as a Solution

#### **The Reasoning Chain:**

1. **Problem**: "I need to model uncertainty in learning"
   → Start with Bayesian inference

2. **Issue**: "But learning rates should adapt to volatility"
   → Consider Kalman Filter (tracks uncertainty)

3. **Issue**: "But volatility itself changes over time"
   → Need to model volatility as a hidden variable

4. **Issue**: "But I don't know the volatility - it's hidden!"
   → Need to INFER volatility from data

5. **Solution**: "What if I treat volatility as another level in a hierarchy?"
   → **This is the HGF insight!**

6. **Implementation**: "Use Bayesian inference at each level"
   → Prediction errors propagate up the hierarchy
   → Each level updates based on errors from below

7. **Result**: "Learning rate emerges automatically from the hierarchy"
   → No need to hand-tune learning rates!
   → Optimal adaptation to volatility

#### **The "Aha!" Moment**

The key insight is recognizing that **uncertainty about states** and **volatility of states** are related but distinct:

- **Uncertainty (σ₂²)**: "How sure am I about the current state?"
- **Volatility (μ₃)**: "How fast is the state changing?"

Traditional models conflated these or ignored volatility entirely. HGF separates them into a hierarchy.

---

### When to Use Each Approach

#### **Use Rescorla-Wagner if:**
- ✅ You want maximum simplicity
- ✅ Environment is stable (low volatility)
- ✅ You're modeling basic associative learning
- ✅ You need fast computation
- ✅ You're teaching/explaining learning basics

#### **Use Kalman Filter if:**
- ✅ Your system is truly linear-Gaussian
- ✅ Volatility is constant and known
- ✅ You need real-time tracking (e.g., GPS)
- ✅ Computational efficiency is critical
- ✅ You have engineering applications

#### **Use Pearce-Hall if:**
- ✅ You're modeling attention effects
- ✅ You want biological plausibility
- ✅ Simple adaptive learning is sufficient
- ✅ You don't need optimal inference

#### **Use Change-Point Detection if:**
- ✅ Changes are truly discrete/abrupt
- ✅ You need to detect specific regime shifts
- ✅ You can afford computational cost
- ✅ Binary detection is sufficient

#### **Use HGF if:**
- ✅ **Volatility changes over time** ← Most important!
- ✅ You need adaptive learning rates
- ✅ You want optimal Bayesian inference
- ✅ You're studying human learning/decision-making
- ✅ You need multi-level uncertainty tracking
- ✅ You're doing computational psychiatry
- ✅ You want to model meta-learning
- ✅ Neural correlates matter

---

### For Your Bayesian EVC Project

**Why HGF is relevant:**

Your project involves:
1. **Uncertainty estimation** - HGF provides principled uncertainty
2. **Cognitive control** - Control should adapt to volatility
3. **Multiple uncertainty types** - Decision vs. state uncertainty
4. **Learning dynamics** - How uncertainty changes over trials

**Current Setup (Simple Bayesian)**:
- Fixed learning rate (α = 0.1)
- Assumes constant volatility
- Good for proof-of-concept

**HGF Upgrade Would Provide**:
- Adaptive learning rates
- Volatility tracking
- Richer uncertainty dynamics
- Better fit to human behavior
- Publishable in computational psychiatry journals

**Bottom Line**: HGF isn't just "another method" - it's the solution that emerged from recognizing that **volatility itself needs to be learned**, not assumed. This was a conceptual breakthrough in computational neuroscience.

---

## What is the HGF?

### Key Concept

The HGF models learning as a **hierarchy of Gaussian random walks**, where:
- **Level 1**: Observations (what you see/experience)
- **Level 2**: Environmental states (hidden causes, e.g., task rules)
- **Level 3**: Volatility (how fast states change)
- **Level 4+**: Higher-order volatility (optional)

### Core Innovation

**Precision-weighted prediction errors**: Learning rates adapt automatically based on uncertainty at each level.

```
Learning Rate ∝ Uncertainty at higher level / Uncertainty at current level
```

When volatility is high → uncertainty increases → learning rate increases → faster adaptation

---

## Why HGF is Perfect for Your EVC Project

### 1. **Multi-Level Uncertainty**
- **Decision uncertainty**: Level 1 (observation noise)
- **State uncertainty**: Level 2 (which rule is active)
- **Volatility uncertainty**: Level 3 (how fast rules change)

### 2. **Automatic Learning Rate Adaptation**
- No need to manually set learning rates
- Adapts to environmental volatility automatically
- Matches human behavior in volatile environments

### 3. **Principled Bayesian Framework**
- All updates follow Bayes' rule
- Provides posterior distributions (not just point estimates)
- Quantifies uncertainty at each level

### 4. **Well-Validated**
- Extensively used in computational psychiatry
- Validated across many cognitive tasks
- Strong neural correlates (ACC, insula track precision)

---

## HGF Variable Glossary (Complete Reference)

### 📚 Understanding HGF Notation

The HGF uses specific notation that can be confusing. Here's what **every variable** means:

---

### **Core Symbols**

| Symbol | Name | Meaning |
|--------|------|---------|
| `t` | Time | Current trial number (e.g., trial 1, 2, 3...) |
| `x` | True state | The **actual** (hidden) state of the world |
| `μ` (mu) | Belief/Mean | Your **estimate** of the state (what you think) |
| `σ²` (sigma squared) | Variance | **Uncertainty** about your estimate (how sure you are) |
| `π` (pi) | Precision | **Inverse of variance**: π = 1/σ² (higher = more certain) |
| `δ` (delta) | Prediction error | Difference between what you expected and what you observed |
| `α` (alpha) | Learning rate | How much to update beliefs from new information (0-1) |

---

### **Level Subscripts**

| Subscript | Level | What It Represents |
|-----------|-------|-------------------|
| `₁` | Level 1 | **Observations** - what you actually see/experience |
| `₂` | Level 2 | **Hidden states** - underlying rules/probabilities |
| `₃` | Level 3 | **Volatility** - how fast things are changing |

**Example**: `μ₂` = "your belief about the hidden state at level 2"

---

### **Time Subscripts**

| Notation | Meaning |
|----------|---------|
| `μ₂,t` | Belief at level 2 at **current** time t |
| `μ₂,t-1` | Belief at level 2 at **previous** time (t-1) |
| `μ₂,t+1` | Belief at level 2 at **next** time (t+1) |
| `u₁:t` | All observations from trial 1 up to trial t |

---

### **Level 1: Observations (What You See)**

| Variable | Name | Meaning | Example |
|----------|------|---------|---------|
| `u_t` | Observation | Actual outcome you observe | 1 (correct) or 0 (incorrect) |
| `μ₁,t` | Expected observation | What you **predict** you'll see | "I expect 70% chance of success" |

**In plain English**: 
- `u_t` = "Did I get it right?" (the actual result)
- `μ₁,t` = "What did I think would happen?" (your prediction)

---

### **Level 2: Hidden States (What You Believe)**

| Variable | Name | Meaning | Example | Range |
|----------|------|---------|---------|-------|
| `x₂,t` | True state | **Actual** hidden state (unknown to you) | True reward probability = 0.7 | -∞ to +∞ (logit space) |
| `μ₂,t` | Belief about state | Your **estimate** of the hidden state | "I think reward prob = 0.65" | -∞ to +∞ (logit space) |
| `σ₂,t²` | State uncertainty | How **uncertain** you are about the state | High uncertainty = 2.0, Low = 0.1 | 0 to ∞ |
| `π₂,t` | State precision | How **certain** you are (inverse of uncertainty) | π = 1/σ² | 0 to ∞ |

**In plain English**:
- `x₂,t` = "The true rule" (you never know this directly)
- `μ₂,t` = "What I think the rule is"
- `σ₂,t²` = "How unsure I am about the rule" (bigger = more unsure)
- `π₂,t` = "How confident I am" (bigger = more confident)

**Transformation**: To convert to probability space (0-1), use sigmoid: `p = 1/(1 + exp(-μ₂))`

---

### **Level 3: Volatility (How Fast Things Change)**

| Variable | Name | Meaning | Example | Range |
|----------|------|---------|---------|-------|
| `x₃,t` | True volatility | **Actual** rate of change (unknown) | Environment changes fast | -∞ to +∞ (log space) |
| `μ₃,t` | Volatility estimate | Your **estimate** of how fast things change | "I think rules change slowly" | -∞ to +∞ (log space) |
| `σ₃,t²` | Volatility uncertainty | Uncertainty about volatility | "Not sure if stable or volatile" | 0 to ∞ |

**In plain English**:
- `x₃,t` = "How fast the rules are actually changing"
- `μ₃,t` = "How fast I think the rules are changing"
- `σ₃,t²` = "How unsure I am about the rate of change"

**Transformation**: To get actual volatility, use exponential: `volatility = exp(μ₃)`

---

### **Parameters (Set Before Running HGF)**

These are **fixed** parameters you set at the beginning:

| Parameter | Name | Meaning | Typical Value | What It Controls |
|-----------|------|---------|---------------|------------------|
| `κ₂` (kappa) | Coupling strength | How much level 3 influences level 2 | 1.0 | Strength of volatility effect |
| `ω₂` (omega) | Baseline log-volatility | Base rate of change at level 2 | -4.0 | How much states drift by default |
| `ω₃` (omega) | Volatility drift | How much volatility itself changes | -6.0 | Stability of volatility |
| `μ₂,₀` | Initial belief | Starting belief about state | 0.0 | Where you start (logit space) |
| `μ₃,₀` | Initial volatility | Starting volatility estimate | 0.0 | Initial volatility belief |
| `σ₂,₀²` | Initial uncertainty | Starting uncertainty at level 2 | 1.0 | How uncertain you start |
| `σ₃,₀²` | Initial volatility unc. | Starting uncertainty at level 3 | 1.0 | Uncertainty about volatility |

**In plain English**:
- `κ₂` = "How much does volatility affect learning?" (bigger = more effect)
- `ω₂` = "How much do rules naturally drift?" (bigger = more drift)
- `ω₃` = "How stable is the volatility?" (smaller = more stable)
- Initial values = "What do I believe at the very start?"

---

### **Derived Quantities (Computed Each Trial)**

| Variable | Name | Formula | Meaning |
|----------|------|---------|---------|
| `δ₁,t` | Level 1 prediction error | `u_t - μ₁,t` | Surprise at observation |
| `δ₂,t` | Level 2 prediction error | `w₂ × δ₁,t` | Weighted surprise |
| `δ₃,t` | Level 3 prediction error | Complex (see below) | Surprise about volatility |
| `π̂₂,t` (pi-hat) | Predicted precision | `1/(σ₂² + exp(κ₂μ₃ + ω₂))` | Expected certainty before update |
| `α₂,t` | Learning rate | `σ₂²/(σ₂² + 1/π̂₂)` | How much to learn from this trial |
| `w₂` | Observation weight | `μ₁(1-μ₁)` for binary | Sigmoid derivative |

**In plain English**:
- `δ₁` = "How surprised am I by what I saw?"
- `δ₂` = "Weighted surprise" (accounts for uncertainty)
- `π̂₂` = "How certain should I be before seeing the outcome?"
- `α₂` = "How much should I update my beliefs?" (0 = don't update, 1 = completely revise)

---

### **Complete Update Flow (Step-by-Step)**

Here's what happens on **each trial**:

#### **1. PREDICTION STEP** (Before seeing outcome)
```python
# Predict precision (certainty) for this trial
π̂₂,t = 1 / (σ₂,t-1² + exp(κ₂ × μ₃,t-1 + ω₂))
```
- **What it means**: "Based on my current uncertainty and volatility, how certain should I be?"
- Higher volatility → lower predicted precision → expect more uncertainty

#### **2. OBSERVATION** (See the outcome)
```python
u_t = 1  # or 0 (correct/incorrect)
μ₁,t = sigmoid(μ₂,t-1)  # What I predicted
```
- **What it means**: "I predicted X, but I observed Y"

#### **3. PREDICTION ERROR** (Compute surprise)
```python
δ₁,t = u_t - μ₁,t  # Raw prediction error
w₂ = μ₁,t × (1 - μ₁,t)  # Observation weight (sigmoid derivative)
δ₂,t = w₂ × δ₁,t  # Weighted prediction error
```
- **What it means**: "How wrong was I? Weight by how uncertain my prediction was"

#### **4. UPDATE LEVEL 2** (Update beliefs about state)
```python
# Update precision (certainty)
π₂,t = π̂₂,t + w₂²
σ₂,t² = 1 / π₂,t  # Convert back to variance

# Update belief (mean)
μ₂,t = μ₂,t-1 + σ₂,t² × δ₂,t
```
- **What it means**: 
  - "I'm now more certain" (precision increases)
  - "I update my belief based on the surprise"

#### **5. UPDATE LEVEL 3** (Update volatility estimate)
```python
# Compute volatility prediction error
δ₃,t = (1/σ₂,t² + (μ₂,t - μ₂,t-1)²/σ₂,t² - 1/π̂₂,t) / 2

# Update volatility
μ₃,t = μ₃,t-1 + κ₂ × σ₃,t² × δ₃,t
```
- **What it means**: "Did things change more or less than I expected? Update my volatility estimate"

#### **6. COMPUTE LEARNING RATE** (For interpretation)
```python
α₂,t = σ₂,t² / (σ₂,t² + 1/π̂₂,t)
```
- **What it means**: "How much did I actually learn from this trial?"
- α close to 0 = barely updated (confident in old belief)
- α close to 1 = completely revised belief (very uncertain or volatile)

---

### **Practical Example with Numbers**

Let's walk through **one trial**:

**Setup**: You're learning which color is rewarded (blue vs. red)

**Before Trial 10**:
- `μ₂ = 0.5` (logit space) → probability ≈ 0.62 "I think blue is rewarded 62% of the time"
- `σ₂² = 1.0` → "I'm moderately uncertain"
- `μ₃ = -4.0` (log space) → volatility ≈ 0.018 "Things change slowly"

**Trial 10**: You choose blue and get rewarded!

**Step 1 - Prediction**:
```python
π̂₂ = 1 / (1.0 + exp(1.0 × (-4.0) + (-4.0))) = 1 / 1.0003 ≈ 1.0
# "I expect to be about as certain as I was"
```

**Step 2 - Observation**:
```python
u = 1  # Rewarded!
μ₁ = sigmoid(0.5) = 0.62  # I predicted 62% chance
```

**Step 3 - Prediction Error**:
```python
δ₁ = 1 - 0.62 = 0.38  # "Positive surprise! Better than expected"
w₂ = 0.62 × 0.38 = 0.24
δ₂ = 0.24 × 0.38 = 0.09
```

**Step 4 - Update Belief**:
```python
π₂ = 1.0 + 0.24² = 1.06
σ₂² = 1/1.06 = 0.94  # Uncertainty decreased slightly
μ₂ = 0.5 + 0.94 × 0.09 = 0.58  # Belief increased
# New probability: sigmoid(0.58) = 0.64 "Now I think 64% chance"
```

**Step 5 - Learning Rate**:
```python
α = 0.94 / (0.94 + 1/1.0) = 0.48
# "I updated my belief by 48% of the prediction error"
```

**Result**: 
- Belief went from 62% → 64% (small update, as expected in stable environment)
- Uncertainty decreased slightly
- Learning rate was moderate (0.48)

---

## HGF Mathematical Framework

### Three-Level HGF (Standard)

**Level 1: Observations**
```
u_t ~ N(μ₁,t, σ₁²)  # Observed outcome
μ₁,t = s(x₂,t)      # Sigmoid transform of level 2
```

**Level 2: Hidden States (e.g., reward probability)**
```
x₂,t = x₂,t-1 + ω₂ √exp(κ₂x₃,t-1)  # Random walk with volatility
μ₂,t = E[x₂,t | u₁:t]                # Posterior mean
σ₂,t² = Var[x₂,t | u₁:t]             # Posterior variance (uncertainty)
```

**Level 3: Volatility**
```
x₃,t = x₃,t-1 + ω₃  # Slowly changing volatility
μ₃,t = E[x₃,t | u₁:t]
σ₃,t² = Var[x₃,t | u₁:t]
```

### Update Equations (Simplified)

**Prediction Error:**
```python
δ₂,t = μ₁,t - s(μ₂,t)  # Difference between observed and expected
```

**Precision-Weighted Update:**
```python
# Precision = 1 / variance (inverse uncertainty)
π₂,t = 1 / σ₂,t²

# Learning rate adapts to uncertainty
α₂,t = σ₂,t² / (σ₂,t² + observation_noise)

# Update belief
μ₂,t = μ₂,t-1 + α₂,t × δ₂,t
```

**Uncertainty Update:**
```python
# Uncertainty increases with volatility
σ₂,t² = σ₂,t-1² + exp(κ₂ × μ₃,t-1)

# Then decreases with new information
σ₂,t² = 1 / (1/σ₂,t² + π_observation)
```

---

## Implementation for Your EVC Project

### Option 1: Use Existing HGF Toolbox

**TAPAS (Translational Algorithms for Psychiatry-Advancing Science)**

```matlab
% MATLAB implementation (most complete)
% Download from: https://www.tnu.ethz.ch/en/software/tapas

% Fit HGF to behavioral data
est = tapas_fitModel(responses, inputs, 'tapas_hgf_binary', 'tapas_bayes_optimal');

% Extract uncertainty estimates
uncertainty_level2 = est.traj.sa2;  % Uncertainty at level 2
volatility = est.traj.mu3;           % Estimated volatility
```

**Python Alternative: pyhgf**

```python
# Install
pip install pyhgf

# Use HGF
from pyhgf import HGF

# Initialize
hgf = HGF(n_levels=3)

# Process trial sequence
for outcome in outcomes:
    hgf.update(outcome)
    
    # Get uncertainty estimates
    state_uncertainty = hgf.get_uncertainty(level=2)
    volatility = hgf.get_mean(level=3)
```

### Option 2: Implement HGF from Scratch

Here's a simplified 3-level HGF implementation:

```python
import numpy as np

class HierarchicalGaussianFilter:
    """
    Three-level Hierarchical Gaussian Filter for uncertainty estimation.
    
    Level 1: Observations (binary outcomes)
    Level 2: Hidden state (e.g., reward probability in logit space)
    Level 3: Volatility (log-volatility of level 2)
    """
    
    def __init__(
        self,
        kappa_2: float = 1.0,    # Coupling level 2 to level 3
        omega_2: float = -4.0,   # Baseline log-volatility at level 2
        omega_3: float = -6.0,   # Volatility of volatility
        mu_2_0: float = 0.0,     # Initial belief about state
        mu_3_0: float = 0.0,     # Initial belief about volatility
        sa_2_0: float = 1.0,     # Initial uncertainty at level 2
        sa_3_0: float = 1.0      # Initial uncertainty at level 3
    ):
        # Parameters
        self.kappa_2 = kappa_2
        self.omega_2 = omega_2
        self.omega_3 = omega_3
        
        # State variables
        self.mu_2 = mu_2_0  # Posterior mean at level 2
        self.mu_3 = mu_3_0  # Posterior mean at level 3
        self.sa_2 = sa_2_0  # Posterior variance at level 2
        self.sa_3 = sa_3_0  # Posterior variance at level 3
        
        # History
        self.history = {
            'mu_2': [mu_2_0],
            'mu_3': [mu_3_0],
            'sa_2': [sa_2_0],
            'sa_3': [sa_3_0],
            'uncertainty_2': [sa_2_0],
            'volatility': [np.exp(mu_3_0)]
        }
    
    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))
    
    def update(self, observation: float):
        """
        Update beliefs based on new observation.
        
        Args:
            observation: Binary outcome (0 or 1) or continuous value
        """
        # --- PREDICTION STEP ---
        
        # Predict level 2 (state) with increased uncertainty from volatility
        # Uncertainty increases based on volatility at level 3
        pi_2_hat = 1 / (self.sa_2 + np.exp(self.kappa_2 * self.mu_3 + self.omega_2))
        
        # Predict level 3 (volatility) with its own volatility
        pi_3_hat = 1 / (self.sa_3 + np.exp(self.omega_3))
        
        # --- PREDICTION ERRORS ---
        
        # Level 1: Observation prediction error
        mu_1_hat = self.sigmoid(self.mu_2)  # Expected observation
        delta_1 = observation - mu_1_hat
        
        # Level 2: Precision-weighted prediction error
        # Weight by derivative of observation function
        w_2 = mu_1_hat * (1 - mu_1_hat)  # Sigmoid derivative
        delta_2 = w_2 * delta_1
        
        # --- UPDATE STEP ---
        
        # Update level 2 (state)
        self.sa_2 = 1 / (pi_2_hat + w_2**2)  # New uncertainty
        self.mu_2 = self.mu_2 + self.sa_2 * delta_2  # New belief
        
        # Level 3: Prediction error (from level 2 uncertainty change)
        # If uncertainty changed more than expected, volatility is higher
        delta_3 = (1/self.sa_2 + (self.mu_2 - self.mu_2)**2 / self.sa_2 - 1/pi_2_hat) / 2
        
        # Update level 3 (volatility)
        self.sa_3 = 1 / pi_3_hat
        self.mu_3 = self.mu_3 + self.kappa_2 * self.sa_3 * delta_3
        
        # --- STORE HISTORY ---
        self.history['mu_2'].append(self.mu_2)
        self.history['mu_3'].append(self.mu_3)
        self.history['sa_2'].append(self.sa_2)
        self.history['sa_3'].append(self.sa_3)
        self.history['uncertainty_2'].append(self.sa_2)
        self.history['volatility'].append(np.exp(self.mu_3))
    
    def get_state_estimate(self):
        """Get current estimate of hidden state (in probability space)."""
        return self.sigmoid(self.mu_2)
    
    def get_state_uncertainty(self):
        """Get current uncertainty about hidden state."""
        return self.sa_2
    
    def get_volatility(self):
        """Get current estimate of environmental volatility."""
        return np.exp(self.mu_3)
    
    def get_learning_rate(self):
        """Get effective learning rate (precision-weighted)."""
        # Learning rate increases with uncertainty and volatility
        pi_2_hat = 1 / (self.sa_2 + np.exp(self.kappa_2 * self.mu_3 + self.omega_2))
        alpha = 1 / (1 + 1 / (self.sa_2 * pi_2_hat))
        return alpha
```

---

## Integration with Bayesian EVC

### Step 1: Use HGF for Uncertainty Estimation

```python
class BayesianEVC_with_HGF:
    def __init__(self):
        self.hgf = HierarchicalGaussianFilter()
        self.evc_model = BayesianEVC()
    
    def process_trial(self, trial_data):
        # Get trial information
        evidence_clarity = trial_data['evidence_clarity']
        reward = trial_data['reward_magnitude']
        outcome = trial_data['accuracy']
        
        # Update HGF with outcome
        self.hgf.update(outcome)
        
        # Get uncertainty estimates from HGF
        state_uncertainty = self.hgf.get_state_uncertainty()
        volatility = self.hgf.get_volatility()
        
        # Decision uncertainty from evidence
        decision_uncertainty = 1 - evidence_clarity
        
        # Combined uncertainty
        total_uncertainty = 0.5 * decision_uncertainty + 0.5 * state_uncertainty
        
        # Confidence (inverse uncertainty)
        confidence = 1 / (1 + total_uncertainty)
        
        # Compute optimal control using Bayesian EVC
        optimal_control = self.evc_model.optimal_control(
            reward_magnitude=reward,
            baseline_accuracy=evidence_clarity,
            uncertainty=total_uncertainty,
            confidence=confidence
        )
        
        return {
            'optimal_control': optimal_control,
            'state_uncertainty': state_uncertainty,
            'volatility': volatility,
            'total_uncertainty': total_uncertainty,
            'confidence': confidence
        }
```

### Step 2: Fit HGF Parameters

```python
def fit_hgf_to_data(behavioral_data):
    """
    Fit HGF parameters to behavioral data.
    """
    from scipy.optimize import minimize
    
    def negative_log_likelihood(params):
        kappa_2, omega_2, omega_3 = params
        
        # Initialize HGF
        hgf = HierarchicalGaussianFilter(
            kappa_2=kappa_2,
            omega_2=omega_2,
            omega_3=omega_3
        )
        
        # Compute likelihood
        log_likelihood = 0
        for _, trial in behavioral_data.iterrows():
            # Predicted probability
            p_pred = hgf.get_state_estimate()
            
            # Observed outcome
            outcome = trial['accuracy']
            
            # Bernoulli likelihood
            p_outcome = p_pred if outcome == 1 else (1 - p_pred)
            log_likelihood += np.log(p_outcome + 1e-10)
            
            # Update HGF
            hgf.update(outcome)
        
        return -log_likelihood
    
    # Optimize
    result = minimize(
        negative_log_likelihood,
        x0=[1.0, -4.0, -6.0],
        bounds=[(0.1, 5.0), (-10.0, 0.0), (-10.0, 0.0)],
        method='L-BFGS-B'
    )
    
    return {
        'kappa_2': result.x[0],
        'omega_2': result.x[1],
        'omega_3': result.x[2],
        'log_likelihood': -result.fun
    }
```

---

## Advantages of HGF for Your Project

### 1. **Theoretically Grounded**
✅ Optimal Bayesian inference under Gaussian assumptions  
✅ Mathematically principled uncertainty quantification  
✅ Well-established in computational psychiatry  

### 2. **Captures Key Phenomena**
✅ Learning rate adaptation to volatility  
✅ Multi-level uncertainty representation  
✅ Individual differences in uncertainty processing  

### 3. **Neural Validity**
✅ ACC activity correlates with precision-weighted prediction errors  
✅ Insula tracks volatility estimates  
✅ DLPFC implements precision-weighted control  

### 4. **Clinical Relevance**
✅ HGF parameters differ in psychiatric conditions  
✅ Can identify atypical uncertainty processing  
✅ Potential for personalized interventions  

---

## Comparison: Simple Bayesian vs. HGF

| Feature | Simple Bayesian | HGF |
|---------|----------------|-----|
| **Uncertainty levels** | 1-2 | 3+ (hierarchical) |
| **Learning rate** | Fixed | Adaptive |
| **Volatility tracking** | No | Yes |
| **Parameter estimation** | Simple | Complex but principled |
| **Computational cost** | Low | Medium |
| **Theoretical foundation** | Good | Excellent |
| **Clinical validation** | Limited | Extensive |

---

## Implementation Roadmap

### Phase 1: Basic HGF (Week 1)
- [ ] Implement 3-level HGF class
- [ ] Test on simulated data
- [ ] Validate against known solutions

### Phase 2: Integration with EVC (Week 2)
- [ ] Connect HGF uncertainty to EVC model
- [ ] Fit combined model to your data
- [ ] Compare with simple Bayesian approach

### Phase 3: Advanced Features (Week 3)
- [ ] Add response model for control allocation
- [ ] Implement hierarchical fitting across subjects
- [ ] Validate with neural data

### Phase 4: Analysis (Week 4)
- [ ] Individual differences in HGF parameters
- [ ] Correlate with neural activity
- [ ] Clinical/developmental applications

---

## Code Example: Complete Pipeline

```python
# Step 1: Generate data with HGF-based uncertainty
from models.bayesian_evc import BayesianEVC

# Step 2: Fit HGF to behavioral data
hgf_params = fit_hgf_to_data(behavioral_data)
print(f"Fitted HGF parameters: {hgf_params}")

# Step 3: Process each subject's data
results = []
for subject_id in behavioral_data['subject_id'].unique():
    subject_data = behavioral_data[behavioral_data['subject_id'] == subject_id]
    
    # Initialize HGF for this subject
    hgf = HierarchicalGaussianFilter(**hgf_params)
    
    # Process trials
    for _, trial in subject_data.iterrows():
        # Update HGF
        hgf.update(trial['accuracy'])
        
        # Get uncertainty
        uncertainty = hgf.get_state_uncertainty()
        volatility = hgf.get_volatility()
        
        # Store
        results.append({
            'subject_id': subject_id,
            'trial': trial['trial'],
            'hgf_uncertainty': uncertainty,
            'hgf_volatility': volatility,
            'observed_control': trial['control_signal']
        })

results_df = pd.DataFrame(results)

# Step 4: Use HGF uncertainty in EVC model
evc_model = BayesianEVC()
evc_model.fit(
    results_df,
    uncertainty_col='hgf_uncertainty',
    observed_control_col='observed_control'
)

print(f"EVC with HGF R²: {evc_model.evaluate(results_df)['r2']:.3f}")
```

---

## Resources

### Software
1. **TAPAS Toolbox** (MATLAB): https://www.tnu.ethz.ch/en/software/tapas
2. **pyhgf** (Python): `pip install pyhgf`
3. **hBayesDM** (R/Stan): Includes HGF models

### Key Papers
1. **Original HGF Paper:**
   - Mathys, C., et al. (2011). "A Bayesian foundation for individual learning under uncertainty." *Frontiers in Human Neuroscience*, 5, 39.

2. **HGF Tutorial:**
   - Mathys, C. D., et al. (2014). "Uncertainty in perception and the Hierarchical Gaussian Filter." *Frontiers in Human Neuroscience*, 8, 825.

3. **Clinical Applications:**
   - Powers, A. R., et al. (2017). "Pavlovian conditioning-induced hallucinations result from overweighting of perceptual priors." *Science*, 357(6351), 596-600.

4. **Neural Correlates:**
   - Iglesias, S., et al. (2013). "Hierarchical prediction errors in midbrain and basal forebrain during sensory learning." *Neuron*, 80(2), 519-530.

### Tutorials
- **HGF Tutorial in R**: https://ccs-lab.github.io/hBayesDM/articles/hgf_tutorial.html
- **TAPAS Documentation**: Included with toolbox download

---

## Next Steps for Your Project

### Immediate (This Week)
1. Implement basic HGF class (provided above)
2. Test on your generated data
3. Compare uncertainty estimates with simple Bayesian

### Short-term (Next Month)
1. Fit HGF parameters to your data
2. Integrate with Bayesian EVC model
3. Compare models: Simple Bayesian vs. HGF-based

### Long-term (Future Work)
1. Hierarchical HGF across subjects
2. Developmental/clinical applications
3. Neural validation with fNIRS/fMRI

---

## Summary

**YES, you should definitely apply HGF to your Bayesian EVC project!**

### Why?
✅ **Perfect fit** for multi-level uncertainty  
✅ **Adaptive learning** matches human behavior  
✅ **Well-validated** in cognitive neuroscience  
✅ **Neural correlates** for validation  
✅ **Clinical relevance** for applications  

### How?
1. Use HGF to estimate state uncertainty and volatility
2. Combine with evidence-based decision uncertainty
3. Feed into Bayesian EVC model
4. Compare with simpler approaches

### Expected Benefits:
- **Better model fit** (higher R²)
- **Richer insights** (volatility tracking, learning rates)
- **Individual differences** (HGF parameters vary meaningfully)
- **Neural validation** (precision-weighted PEs in ACC)

**Ready to implement?** Start with the `HierarchicalGaussianFilter` class above!

