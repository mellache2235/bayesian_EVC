# How Cognitive Models Handle Uncertainty: Before and After Our Proposal

## Overview

This document compares how different cognitive modeling frameworks handle uncertainty, contrasting traditional approaches with our Bayesian EVC extension.

---

## Summary Table: Uncertainty Across Models

| Model | Uncertainty Type | How It's Represented | How It Affects Behavior | Limitations |
|-------|-----------------|---------------------|------------------------|-------------|
| **DDM** | Decision uncertainty | Distance from boundary | RT, confidence | ❌ No state/environmental uncertainty |
| **Rescorla-Wagner** | None explicitly | Implicit in prediction error | Learning rate (fixed) | ❌ No uncertainty tracking |
| **Kalman Filter** | State uncertainty | Variance (σ²) | Learning rate (adaptive) | ❌ No volatility learning |
| **HGF** | Multi-level | σ² at each level | Learning rate, precision | ✅ Most complete, but complex |
| **Bayesian Inference** | Posterior uncertainty | Posterior variance/entropy | Choice probability | ⚠️ Often doesn't affect control |
| **RL Models** | Value uncertainty | Exploration bonus | Exploration vs exploitation | ⚠️ Separate from control |
| **Traditional EVC** | None | N/A | N/A | ❌ Ignores uncertainty entirely |
| **Our Bayesian EVC** | Multi-type | Decision + state uncertainty | Control allocation | ✅ Directly affects control |

---

## 1. Drift Diffusion Model (DDM)

### **What It Models:**
Decision-making as evidence accumulation toward a threshold.

### **How Uncertainty Appears:**

#### **A. Decision Uncertainty (Implicit)**

```
Evidence accumulation:
x(t) = x(t-1) + v·dt + σ·√dt·ε

Where:
- v = drift rate (evidence strength)
- σ = diffusion noise (uncertainty)
- x = accumulated evidence
```

**Uncertainty representation:**
- **Distance from boundary**: How close to decision threshold
- **Drift rate**: Lower v → more uncertain → slower RT
- **Diffusion noise**: Higher σ → more variable → less confident

---

#### **B. Confidence Computation**

```python
# Traditional DDM confidence
confidence = f(balance_of_evidence, RT)

# Higher confidence when:
# 1. Strong evidence (high drift rate)
# 2. Fast RT (quick accumulation)
# 3. Far from boundary (clear winner)
```

**Example:**
```
Trial 1: v = 0.8 (strong evidence) → RT = 500ms → Confidence = 0.9
Trial 2: v = 0.2 (weak evidence) → RT = 1200ms → Confidence = 0.4
```

---

### **How Uncertainty Affects Behavior in DDM:**

#### **1. Reaction Time**
```
High uncertainty (low v) → Slow RT
Low uncertainty (high v) → Fast RT
```

#### **2. Accuracy**
```
High uncertainty (low v) → More errors
Low uncertainty (high v) → Fewer errors
```

#### **3. Confidence**
```
High uncertainty → Low confidence
Low uncertainty → High confidence
```

---

### **What DDM Does NOT Model:**

❌ **State uncertainty**: "What's the true reward probability?"
❌ **Environmental volatility**: "How fast are things changing?"
❌ **Control allocation**: "How much effort should I exert?"
❌ **Learning**: "How should I update my beliefs?"

**DDM focuses on single-trial decisions, not learning or control.**

---

### **Our Extension:**

```python
# Traditional DDM
confidence = f(drift_rate, RT)
# Uncertainty is OUTPUT, doesn't affect control

# Our Bayesian EVC
control = f(reward, effort, uncertainty)
# Uncertainty is INPUT, directly affects control allocation
```

**Key difference:** DDM estimates uncertainty from behavior; we use uncertainty to predict control.

---

## 2. Rescorla-Wagner Model

### **What It Models:**
Associative learning through prediction errors.

### **How Uncertainty Appears:**

#### **Formula:**
```
V(t+1) = V(t) + α × [R(t) - V(t)]

Where:
- V(t) = Value estimate at time t
- α = Learning rate (FIXED, e.g., 0.1)
- R(t) = Reward received
- [R(t) - V(t)] = Prediction error
```

---

### **Uncertainty Representation:**

**Implicit only:**
- Large prediction errors → "I was uncertain/wrong"
- Small prediction errors → "I was certain/correct"

**But:**
- ❌ No explicit uncertainty variable
- ❌ Learning rate doesn't adapt to uncertainty
- ❌ Can't distinguish "uncertain" from "volatile"

---

### **Example:**

```
Trial 1-10: Reward = 1.0 every time
├─ V increases: 0 → 0.1 → 0.19 → ... → 0.65
├─ Prediction errors shrink: 1.0 → 0.9 → 0.81 → ... → 0.35
└─ Uncertainty decreasing (implicit)

Trial 11: Reward = 0 (surprise!)
├─ Large prediction error: -0.65
├─ V updates: 0.65 → 0.585
└─ But α stays 0.1 (doesn't increase despite surprise!)
```

---

### **What Rescorla-Wagner Does NOT Model:**

❌ **Explicit uncertainty tracking**
❌ **Adaptive learning rates**
❌ **Volatility estimation**
❌ **Control allocation**

**It's a learning model, not a control model.**

---

### **Our Extension:**

```python
# Rescorla-Wagner
V(t+1) = V(t) + α × PE
# Fixed α, no uncertainty

# Our Bayesian EVC
control = f(reward, effort, uncertainty)
# Uncertainty from Bayesian estimator affects control
```

---

## 3. Kalman Filter

### **What It Models:**
Optimal Bayesian filtering for linear-Gaussian systems.

### **How Uncertainty Appears:**

#### **Explicit Uncertainty Tracking:**

```
State estimate: μ(t)
State uncertainty: σ²(t)

Prediction step:
μ̂(t) = μ(t-1)
σ̂²(t) = σ²(t-1) + Q  # Q = process noise (FIXED)

Update step:
K(t) = σ̂²(t) / (σ̂²(t) + R)  # Kalman gain
μ(t) = μ̂(t) + K(t) × [observation - μ̂(t)]
σ²(t) = (1 - K(t)) × σ̂²(t)
```

---

### **How Uncertainty Affects Behavior:**

#### **1. Learning Rate (Kalman Gain)**

```
High uncertainty (σ² large) → High K → Fast learning
Low uncertainty (σ² small) → Low K → Slow learning
```

**Example:**
```
Trial 1: σ² = 10 → K = 0.9 → Learn a lot from observation
Trial 50: σ² = 0.5 → K = 0.3 → Learn little from observation
```

**This is adaptive!** Unlike Rescorla-Wagner's fixed α.

---

#### **2. Confidence**

```
confidence = 1 / σ²  (precision)

High uncertainty → Low confidence
Low uncertainty → High confidence
```

---

### **What Kalman Filter Models Well:**

✅ **State uncertainty**: Explicitly tracked
✅ **Adaptive learning**: K adapts to uncertainty
✅ **Optimal inference**: Provably optimal for linear-Gaussian

---

### **What Kalman Filter Does NOT Model:**

❌ **Volatility learning**: Q is fixed, not estimated
❌ **Nonlinear observations**: Assumes linear relationships
❌ **Control allocation**: No mechanism for effort/control

**Example problem:**
```
Trials 1-50: Q = 0.01 (stable environment) → Works great
Trial 51: Environment becomes volatile! But Q is still 0.01
Trials 52-70: Slow to adapt because Q doesn't change
```

---

### **Our Extension:**

```python
# Kalman Filter
learning_rate = σ² / (σ² + R)  # Adaptive to uncertainty
# But no control allocation

# Our Bayesian EVC
control = f(reward, effort, uncertainty)
# Uses uncertainty (from Kalman or other estimator) to allocate control
```

---

## 4. Hierarchical Gaussian Filter (HGF)

### **What It Models:**
Multi-level Bayesian inference with volatility learning.

### **How Uncertainty Appears:**

#### **Multi-Level Uncertainty:**

```
Level 3: Volatility
├─ μ₃(t) = Volatility estimate
├─ σ₃²(t) = Uncertainty about volatility
└─ Affects: How fast level 2 changes

Level 2: Hidden states
├─ μ₂(t) = State estimate
├─ σ₂²(t) = State uncertainty ← KEY!
└─ Affected by: Volatility from level 3

Level 1: Observations
├─ u(t) = Observed outcome
└─ Affected by: State from level 2
```

---

### **How Uncertainty Affects Behavior:**

#### **1. Learning Rate (Precision-Weighted)**

```
α(t) = f(σ₂², volatility)

High volatility → High uncertainty → High learning rate
Low volatility → Low uncertainty → Low learning rate
```

**This is fully adaptive!** Learning rate emerges from the hierarchy.

---

#### **2. Precision-Weighted Prediction Errors**

```
Update = precision × prediction_error

High uncertainty (low precision) → Small update
Low uncertainty (high precision) → Large update
```

---

#### **3. Volatility Estimation**

```
If prediction errors are consistently large:
→ Increase volatility estimate (μ₃)
→ Increase uncertainty (σ₂²)
→ Increase learning rate

If prediction errors are consistently small:
→ Decrease volatility estimate
→ Decrease uncertainty
→ Decrease learning rate
```

**This is meta-learning!** Learning about learning itself.

---

### **What HGF Models Well:**

✅ **Multi-level uncertainty**: State + volatility
✅ **Adaptive learning**: Fully adaptive to volatility
✅ **Volatility learning**: Estimates how fast things change
✅ **Optimal Bayesian inference**: Principled framework

---

### **What HGF Does NOT Model:**

❌ **Control allocation**: No mechanism for effort/control
❌ **Reward-based decisions**: Focuses on belief updating
❌ **Cost-benefit tradeoffs**: No cost function

**HGF is about learning, not control.**

---

### **Our Extension:**

```python
# HGF
uncertainty = hgf.get_state_uncertainty()  # σ₂²
volatility = hgf.get_volatility()  # exp(μ₃)
# Rich uncertainty estimates, but no control

# Our Bayesian EVC
control = f(reward, effort, uncertainty)
# Uses HGF uncertainty to allocate control
```

**We can use HGF as the uncertainty estimator for Bayesian EVC!**

---

## 5. Bayesian Inference Models

### **What They Model:**
Optimal decision-making under uncertainty.

### **How Uncertainty Appears:**

#### **Posterior Distribution:**

```
Prior: P(state)
Likelihood: P(observation | state)
Posterior: P(state | observation) ∝ P(observation | state) × P(state)

Uncertainty = Variance or Entropy of posterior
```

---

### **Example: Bayesian Categorization**

```python
# Two categories: A and B
# Observation: x = 5

# Posterior probabilities
P(A | x=5) = 0.7
P(B | x=5) = 0.3

# Uncertainty (entropy)
H = -0.7×log(0.7) - 0.3×log(0.3) = 0.61 bits
```

---

### **How Uncertainty Affects Behavior:**

#### **1. Choice Probability**

```
High uncertainty → More variable choices
Low uncertainty → Consistent choices
```

**Example:**
```
High certainty: P(A)=0.95, P(B)=0.05 → Always choose A
High uncertainty: P(A)=0.55, P(B)=0.45 → Variable choices
```

---

#### **2. Confidence**

```
confidence = max(P(state | observation))

or

confidence = 1 - Entropy(posterior)
```

---

### **What Bayesian Models Do Well:**

✅ **Optimal inference**: Provably optimal given assumptions
✅ **Uncertainty quantification**: Full posterior distribution
✅ **Flexible**: Can model any generative process

---

### **What Bayesian Models Typically Do NOT Model:**

❌ **Control allocation**: Uncertainty doesn't affect effort
❌ **Cost-benefit tradeoffs**: No cost function
❌ **Individual differences**: Often assume optimal

**Traditional Bayesian models focus on inference, not control.**

---

### **Our Extension:**

```python
# Traditional Bayesian
posterior = bayesian_update(prior, likelihood, observation)
uncertainty = entropy(posterior)
choice = argmax(posterior)
# Uncertainty affects choice, not control

# Our Bayesian EVC
uncertainty = entropy(posterior)
control = f(reward, effort, uncertainty)
# Uncertainty directly affects control allocation
```

---

## 6. Reinforcement Learning Models

### **What They Model:**
Learning to maximize reward through trial and error.

### **How Uncertainty Appears:**

#### **A. Value Uncertainty**

```
Q(s,a) = Expected reward for action a in state s
σ²(s,a) = Uncertainty about Q(s,a)

# Often tracked with:
- Bayesian RL: Posterior distribution over Q
- Count-based: σ² ∝ 1/N(s,a)
- UCB: Upper confidence bound
```

---

#### **B. Exploration Bonus**

```
Action selection:
a* = argmax[Q(s,a) + β × σ(s,a)]

Where:
- Q(s,a) = Expected value (exploitation)
- β × σ(s,a) = Exploration bonus (uncertainty)
- β = Exploration parameter
```

**Example:**
```
Action A: Q = 10, σ = 0.1 → Total = 10.1 (well-known)
Action B: Q = 8, σ = 5.0 → Total = 13.0 (uncertain, worth exploring!)
```

---

### **How Uncertainty Affects Behavior:**

#### **1. Exploration vs. Exploitation**

```
High uncertainty → Explore (try uncertain options)
Low uncertainty → Exploit (choose best-known option)
```

---

#### **2. Learning Rate (in some models)**

```
α(s,a) = f(N(s,a))  # Function of visit count

Few visits → High uncertainty → High learning rate
Many visits → Low uncertainty → Low learning rate
```

---

### **What RL Models Do Well:**

✅ **Value uncertainty**: Explicitly tracked
✅ **Exploration**: Uncertainty drives exploration
✅ **Reward maximization**: Optimal in the limit

---

### **What RL Models Typically Do NOT Model:**

❌ **Control/effort allocation**: Uncertainty affects exploration, not control
❌ **Metacognition**: No confidence or monitoring
❌ **Cost of control**: Effort is often ignored

**RL uses uncertainty for exploration, not control allocation.**

---

### **Our Extension:**

```python
# RL with uncertainty
action = argmax(Q + β × σ)  # Exploration bonus
# Uncertainty affects exploration

# Our Bayesian EVC
control = f(reward, effort, uncertainty)
# Uncertainty affects control/effort, not just exploration
```

---

## 7. Traditional EVC (Shenhav et al., 2013)

### **What It Models:**
Control allocation as cost-benefit optimization.

### **Formula:**

```
EVC = Expected_Benefit - Expected_Cost

Expected_Benefit = Reward × P(success | control)
Expected_Cost = Effort_cost(control)

Control* = argmax[EVC(control)]
```

---

### **How Uncertainty Appears:**

**It doesn't!** ❌

```
EVC = Reward × Accuracy - Cost(Control)

Where:
- Reward = Fixed value
- Accuracy = Fixed probability
- Cost = Function of control level

No uncertainty term!
```

---

### **Example:**

```
Task 1: Reward = $10, Accuracy = 0.8, Cost = 2×control²
├─ EVC = 10×0.8 - 2×control²
├─ Optimal control = 2.0
└─ No consideration of uncertainty!

Task 2: Same reward and accuracy, but you're very uncertain
├─ Traditional EVC: Same control = 2.0
└─ But you should probably exert more control when uncertain!
```

---

### **What Traditional EVC Models Well:**

✅ **Reward sensitivity**: Higher reward → more control
✅ **Effort costs**: Control is metabolically expensive
✅ **Task difficulty**: Harder tasks need more control
✅ **Neural correlates**: ACC tracks EVC

---

### **What Traditional EVC Does NOT Model:**

❌ **Uncertainty**: Completely ignored
❌ **Learning**: No belief updating
❌ **Volatility**: No environmental changes
❌ **Information value**: No information gain

**This is the gap we're filling!**

---

## 8. Our Bayesian EVC Extension

### **What We Add:**

```
Bayesian_EVC = Expected_Benefit + Uncertainty_Benefit - Expected_Cost

Expected_Benefit = Reward × Accuracy
Uncertainty_Benefit = λ × Uncertainty  ← NEW!
Expected_Cost = Effort_cost(Control)

Control* = argmax[Bayesian_EVC(control)]
```

---

### **How Uncertainty Appears:**

#### **A. Decision Uncertainty**
```
From evidence clarity:
decision_uncertainty = 1 - evidence_clarity
```

#### **B. State Uncertainty**
```
From Bayesian estimator (or HGF):
state_uncertainty = σ²(state)
```

#### **C. Combined Uncertainty**
```
total_uncertainty = w₁ × decision_uncertainty + w₂ × state_uncertainty
```

---

### **How Uncertainty Affects Behavior:**

#### **1. Control Allocation**

```
High uncertainty → High control
Low uncertainty → Low control
```

**Example:**
```
Certain trial: uncertainty = 0.2 → control = 0.4
Uncertain trial: uncertainty = 0.8 → control = 0.7
```

---

#### **2. Individual Differences**

```
High λ (uncertainty weight):
├─ Very sensitive to uncertainty
├─ Allocate much more control when uncertain
└─ May be anxious, perfectionistic

Low λ:
├─ Less sensitive to uncertainty
├─ Similar control regardless of uncertainty
└─ May be impulsive, risk-seeking
```

---

#### **3. Learning and Adaptation**

```
Early trials: High uncertainty → High control → Learn quickly
Late trials: Low uncertainty → Low control → Efficient performance
```

---

### **What Our Bayesian EVC Models:**

✅ **Uncertainty-sensitive control**: Control adapts to uncertainty
✅ **Information value**: Uncertainty reduction is valuable
✅ **Individual differences**: λ parameter captures variability
✅ **Adaptive behavior**: Explains exploration, learning, monitoring
✅ **Clinical relevance**: Abnormal λ in psychiatric conditions

---

## Comparison: How Each Model Handles Uncertainty

### **Scenario: Learning a Volatile Task**

```
Trials 1-50: Rule A is correct (stable)
Trial 51: Rule switches to B (volatile!)
Trials 52-100: Rule B is correct (stable again)
```

---

### **1. DDM**

```
Trials 1-50: High drift rate → Fast RT → High confidence
Trial 51: Low drift rate (conflict) → Slow RT → Low confidence
Trials 52-100: Drift rate recovers → Fast RT → High confidence
```

**Captures:** Trial-by-trial decision uncertainty
**Misses:** No learning, no adaptation

---

### **2. Rescorla-Wagner**

```
Trials 1-50: V(A) increases to 0.9
Trial 51: Large prediction error! But α stays 0.1
Trials 52-100: V(B) slowly increases (α = 0.1)
```

**Captures:** Learning from prediction errors
**Misses:** Learning rate doesn't adapt to volatility

---

### **3. Kalman Filter**

```
Trials 1-50: σ² decreases to 0.1 (confident)
Trial 51: Large prediction error → σ² increases to 0.3
Trials 52-100: σ² decreases again
```

**Captures:** Adaptive learning rate
**Misses:** Doesn't learn that environment is volatile

---

### **4. HGF**

```
Trials 1-50: 
├─ σ₂² decreases (state uncertainty)
├─ μ₃ decreases (low volatility)
└─ Learning rate decreases

Trial 51: Large prediction error!
├─ σ₂² increases (state uncertainty)
├─ μ₃ increases (detected volatility!)
└─ Learning rate increases

Trials 52-100:
├─ σ₂² decreases (learning new rule)
├─ μ₃ stays elevated (environment is volatile)
└─ Learning rate stays higher than trials 1-50
```

**Captures:** Everything! Adaptive learning + volatility detection
**Misses:** No control allocation

---

### **5. Traditional EVC**

```
Trials 1-100: Same control throughout
```

**Captures:** Nothing about uncertainty or learning!
**Misses:** Everything temporal

---

### **6. Our Bayesian EVC**

```
Trials 1-50:
├─ State uncertainty decreases (learning)
├─ Control decreases (efficient)
└─ control = 0.5 → 0.3

Trial 51:
├─ State uncertainty increases (surprise!)
├─ Control increases (need to relearn)
└─ control = 0.7

Trials 52-100:
├─ State uncertainty decreases (learning new rule)
├─ Control decreases (efficient again)
└─ control = 0.7 → 0.4
```

**Captures:** Uncertainty-driven control adaptation
**Combines:** Learning (from Bayesian/HGF) + Control (from EVC)

---

## Summary: Before vs. After Our Proposal

### **Before (Existing Models):**

| Model | Strength | Weakness |
|-------|----------|----------|
| **DDM** | Decision uncertainty | No learning, no control |
| **Rescorla-Wagner** | Simple learning | No uncertainty tracking |
| **Kalman Filter** | Adaptive learning | No volatility learning |
| **HGF** | Multi-level uncertainty | No control allocation |
| **Bayesian** | Optimal inference | No control allocation |
| **RL** | Exploration | Uncertainty → exploration, not control |
| **Traditional EVC** | Control allocation | No uncertainty |

**Gap:** No model combines uncertainty with control allocation!

---

### **After (Our Bayesian EVC):**

```
Bayesian EVC = Traditional EVC + Uncertainty

Combines:
✅ Control allocation (from EVC)
✅ Uncertainty tracking (from Bayesian/HGF)
✅ Cost-benefit optimization (from EVC)
✅ Individual differences (λ parameter)
✅ Clinical relevance (abnormal uncertainty processing)
```

---

## Key Innovations

### **1. Uncertainty as Control Input**

**Before:**
```
DDM: behavior → uncertainty (output)
HGF: observations → uncertainty (output)
```

**After:**
```
Bayesian EVC: uncertainty → control (input)
```

---

### **2. Unified Framework**

**Before:**
```
Learning models: Handle uncertainty, ignore control
Control models: Handle control, ignore uncertainty
```

**After:**
```
Bayesian EVC: Handles both uncertainty AND control
```

---

### **3. Individual Differences**

**Before:**
```
Most models: Assume optimal or fixed parameters
```

**After:**
```
Bayesian EVC: λ parameter captures individual differences in uncertainty sensitivity
```

---

### **4. Clinical Applications**

**Before:**
```
Limited: Hard to relate uncertainty processing to control deficits
```

**After:**
```
Direct: Abnormal λ → abnormal control under uncertainty
```

---

## Practical Implications

### **For Researchers:**

**Before:**
- Use DDM for decisions
- Use RL for learning
- Use EVC for control
- **Models are separate!**

**After:**
- Use Bayesian EVC for all three!
- **Unified framework**

---

### **For Clinicians:**

**Before:**
- Hard to measure uncertainty processing
- Unclear how uncertainty relates to symptoms

**After:**
- Measure λ (uncertainty weight)
- Direct link: High λ → anxiety, Low λ → impulsivity

---

## Conclusion

### **The Landscape Before:**

```
Uncertainty Models          Control Models
    ↓                           ↓
  DDM, HGF, RL          Traditional EVC
    ↓                           ↓
Estimate uncertainty    Allocate control
    ↓                           ↓
  No control            No uncertainty
    ↓                           ↓
    ╰─────────GAP────────────╯
```

---

### **Our Contribution:**

```
    Uncertainty Models
         ↓
    Bayesian/HGF
         ↓
    Uncertainty Estimate
         ↓
    ┌────────────┐
    │ Bayesian   │
    │    EVC     │ ← NEW!
    └────────────┘
         ↓
    Control Allocation
```

**We bridge the gap between uncertainty estimation and control allocation!**

---

## Key References

### **DDM:**
- Ratcliff, R., & McKoon, G. (2008). "The diffusion decision model." *Neural Computation*, 20(4), 873-922.

### **Rescorla-Wagner:**
- Rescorla, R. A., & Wagner, A. R. (1972). "A theory of Pavlovian conditioning." *Classical Conditioning II*, 64-99.

### **Kalman Filter:**
- Kalman, R. E. (1960). "A new approach to linear filtering and prediction problems." *Journal of Basic Engineering*, 82(1), 35-45.

### **HGF:**
- Mathys, C., et al. (2011). "A Bayesian foundation for individual learning under uncertainty." *Frontiers in Human Neuroscience*, 5, 39.

### **Bayesian Models:**
- Doya, K., et al. (2007). *Bayesian Brain: Probabilistic Approaches to Neural Coding*. MIT Press.

### **RL:**
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

### **Traditional EVC:**
- Shenhav, A., et al. (2013). "The expected value of control." *Neuron*, 79(2), 217-240.

