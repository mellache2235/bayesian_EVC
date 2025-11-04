# Hierarchical Drift Diffusion Model (HDDM) vs. Bayesian EVC

## The Question

**"Should we use Hierarchical Drift Diffusion Model (HDDM) instead of Bayesian EVC?"**

## Short Answer

**Both are valuable, but they answer DIFFERENT questions!**

- **HDDM**: Best for understanding **decision processes** (how choices are made)
- **Bayesian EVC**: Best for understanding **control allocation** (when/why effort is exerted)

**You might want BOTH!**

---

## What is HDDM?

### **Drift Diffusion Model (DDM) Basics:**

**Core idea:** Decisions are made by accumulating noisy evidence until a threshold is reached.

```
Evidence accumulation:
    dx/dt = v + σ × ε(t)
    
    v = drift rate (evidence strength)
    σ = diffusion noise
    a = boundary separation (caution)
    t₀ = non-decision time
```

**Visual:**
```
Upper boundary (a) ─────────────────
    ↑
    │  ╱╲╱╲  ← Noisy evidence accumulation
    │ ╱    ╲╱
    │╱         (drift rate v pushes upward)
Start ───────────────────────────────
    │
    │
Lower boundary (-a) ────────────────
    
When evidence hits boundary → Make decision!
Time to boundary → Reaction time
```

---

### **Hierarchical DDM (HDDM):**

**Extension:** DDM with hierarchical Bayesian structure

```
POPULATION LEVEL:
    μ_v = 0.5  (mean drift rate across people)
    σ_v = 0.2  (between-person variability)
    
INDIVIDUAL LEVEL:
    Person 1: v₁ ~ Normal(μ_v, σ_v) → v₁ = 0.6
    Person 2: v₂ ~ Normal(μ_v, σ_v) → v₂ = 0.4
    
TRIAL LEVEL:
    RT and Choice generated from individual's DDM parameters
```

**Advantages over regular DDM:**
- ✅ Handles small N per person (partial pooling)
- ✅ Population + individual estimates
- ✅ Stable parameter estimates
- ✅ Can compare groups (e.g., children vs. adults)

---

## HDDM vs. Bayesian EVC: Key Differences

### **What Each Model Predicts:**

| Model | Predicts | Input Variables | Output | Focus |
|-------|----------|-----------------|--------|-------|
| **HDDM** | RT distribution & choice accuracy | Stimulus features | Drift rate, boundary, non-decision time | **How decisions are made** |
| **Bayesian EVC** | Control allocation | Reward, uncertainty, difficulty | Control level (effort) | **When/why effort is exerted** |

---

### **Different Questions:**

**HDDM asks:**
- "How fast does this child accumulate evidence?"
- "How cautious are they?" (boundary separation)
- "What's their processing speed?" (non-decision time)

**Bayesian EVC asks:**
- "How much cognitive control does this child allocate?"
- "Do they exert more effort when uncertain?"
- "How does control depend on rewards and costs?"

---

## Detailed Comparison

### **1. What They Measure**

#### **HDDM Parameters:**

```python
v = drift rate (evidence quality / processing speed)
    - High v: Strong evidence, fast decisions
    - Low v: Weak evidence, slow decisions
    
a = boundary separation (caution)
    - High a: Cautious, slow but accurate
    - Low a: Impulsive, fast but error-prone
    
t₀ = non-decision time (motor/encoding)
    - Pure processing time unrelated to decision
```

**What this tells you:**
- Information processing efficiency
- Speed-accuracy tradeoff strategy
- Cognitive processing speed

---

#### **Bayesian EVC Parameters:**

```python
λ = uncertainty weight
    - High λ: Sensitive to uncertainty, allocate more control when uncertain
    - Low λ: Insensitive to uncertainty
    
β_r = reward weight
    - High β_r: Motivated by rewards
    - Low β_r: Less reward-sensitive
    
β_e = effort cost weight
    - High β_e: Effort is costly, minimize control
    - Low β_e: Effort is cheap, willing to exert control
```

**What this tells you:**
- Control allocation strategy
- Value of uncertainty reduction
- Cost-benefit optimization

---

### **2. Uncertainty Representation**

#### **HDDM:**

**Uncertainty is OUTPUT (estimated from behavior):**

```python
# Fit HDDM to data
hddm_model.fit(rt, choice, accuracy)

# Extract trial-by-trial uncertainty
for trial in trials:
    drift_rate_trial = hddm_model.get_drift_rate(trial)
    
    # Low drift rate → high uncertainty
    uncertainty = 1 / drift_rate_trial
    
    # Or from posterior
    confidence = P(correct | evidence)
    uncertainty = 1 - confidence
```

**Uncertainty comes FROM the model fit.**

---

#### **Bayesian EVC:**

**Uncertainty is INPUT (affects predictions):**

```python
# Uncertainty is given or estimated separately
uncertainty = estimate_uncertainty(trial)

# Use uncertainty to predict control
control = bayesian_evc.predict(reward, accuracy, uncertainty)
```

**Uncertainty goes INTO the model.**

---

### **3. What They're Good For**

#### **HDDM:**

✅ **Understanding decision mechanisms**
- How does evidence accumulation work?
- What's the cognitive process during decisions?

✅ **Decomposing RT**
- Processing speed (t₀)
- Evidence quality (v)
- Caution (a)

✅ **Clinical diagnosis**
- ADHD: Low boundary (impulsive)
- Anxiety: High boundary (over-cautious)
- Depression: Low drift rate (slowed processing)

✅ **Individual differences in processing**
- Who's faster/slower?
- Who's more cautious?

---

#### **Bayesian EVC:**

✅ **Understanding control allocation**
- When do people exert effort?
- How does uncertainty affect control?

✅ **Predicting effort/resources**
- Which tasks will require more control?
- How to optimize task difficulty?

✅ **Individual differences in motivation**
- Who values uncertainty reduction?
- Who's more effort-averse?

✅ **Clinical applications**
- Math anxiety: High λ (over-allocate control under uncertainty)
- Low motivation: Low β_r (insensitive to rewards)

---

## Can You Combine Them? YES!

### **Integrated Approach:**

```
HDDM
  ↓ (estimates)
Drift rate, Boundary, Uncertainty
  ↓ (feed into)
Bayesian EVC
  ↓ (predicts)
Control allocation
```

---

### **Implementation:**

```python
class HDDM_BayesianEVC_Integration:
    """
    Combined model:
    1. HDDM estimates uncertainty from RT/choice
    2. Bayesian EVC uses uncertainty to predict control
    """
    
    def __init__(self):
        import hddm
        
        # HDDM for uncertainty estimation
        self.hddm = hddm.HDDM(...)
        
        # Bayesian EVC for control prediction
        from models.bayesian_evc import BayesianEVC
        self.evc = BayesianEVC()
    
    def fit(self, data):
        """
        Two-stage fitting:
        1. Fit HDDM to get trial-by-trial drift rates
        2. Use drift rates to compute uncertainty
        3. Fit Bayesian EVC using HDDM uncertainty
        """
        
        # Stage 1: Fit HDDM
        print("Stage 1: Fitting HDDM to RT and choice data...")
        self.hddm.fit(data)
        
        # Extract drift rates per trial
        drift_rates = self.hddm.get_drift_rates()
        
        # Compute uncertainty from drift rate
        # Low drift = high uncertainty
        data['hddm_uncertainty'] = 1 / (1 + drift_rates)
        
        # Stage 2: Fit Bayesian EVC
        print("Stage 2: Fitting Bayesian EVC with HDDM uncertainty...")
        evc_results = self.evc.fit(
            data,
            observed_control_col='control_proxy',  # e.g., RT or neural
            uncertainty_col='hddm_uncertainty'  # From HDDM!
        )
        
        return evc_results
```

---

## For Your Arithmetic Task Application

### **Which Model to Use?**

#### **Use HDDM if you want to understand:**

✅ **Processing speed differences**
- Why do some children solve faster?
- Does processing speed improve with age?

✅ **Speed-accuracy tradeoffs**
- Are children setting appropriate caution levels?
- Do they adjust boundaries based on difficulty?

✅ **Cognitive efficiency**
- Information processing rate
- Neural efficiency

**Example research question:**
> "How does drift rate (processing efficiency) change with age and math ability in children?"

---

#### **Use Bayesian EVC if you want to understand:**

✅ **Effort allocation**
- How do children decide how much effort to exert?
- Does uncertainty make them try harder?

✅ **Motivation and value**
- Are children motivated by points/praise?
- Do they value reducing uncertainty?

✅ **Control strategies**
- When do children give up vs. persist?
- Individual differences in persistence

**Example research question:**
> "Do children allocate more cognitive control when facing uncertain math problems?"

---

#### **Use BOTH (Recommended!) if you want:**

✅ **Complete picture**
- HDDM: How they process information
- Bayesian EVC: How they allocate control

✅ **Richer insights**
- Does high drift rate reduce need for control?
- Do children with low drift compensate with high control?

**Example research question:**
> "How do processing efficiency (HDDM drift rate) and control allocation (Bayesian EVC) jointly predict math performance?"

---

## Practical Example: Arithmetic Task

### **Scenario:**

Child solving: **234 + 567 = ?**

---

### **What HDDM Models:**

```
Problem presented → Start accumulating evidence
    ↓ (mental calculation)
Evidence accumulates: 801... 799... 802... 800... 801...
    ↓ (noisy process)
Hits boundary at 801 → Respond "801"
    ↓
RT = time to boundary
Accuracy = whether crossed correct boundary

HDDM extracts:
- v = 0.3 (slow accumulation - hard problem!)
- a = 1.2 (high caution - wants to be sure)
- t₀ = 400ms (encoding + motor time)
```

**Interpretation:**
- Low v: Child finds problem difficult
- High a: Child is being careful (good strategy!)

---

### **What Bayesian EVC Models:**

```
Problem presented → Assess task demands
    ↓
Difficulty = 4 (hard)
Reward = 40 points
Uncertainty = 0.7 (not confident)
    ↓
Compute EVC = (40 × 0.5 + λ × 0.7) / (2 × 1.0)
    ↓
If λ = 0.4:
Control = 0.3 + (20 + 0.28) / 2 = 0.44
    ↓
Allocate 44% of max control
    ↓
Results in: More WM engagement, slower RT, higher accuracy
```

**Interpretation:**
- High uncertainty → allocate more control
- λ = 0.4 means child values uncertainty reduction

---

### **Combined Interpretation:**

```
HDDM says: "Child has slow processing (v = 0.3) but compensates with caution (a = 1.2)"
Bayesian EVC says: "Child allocates high control (0.44) due to high uncertainty (λ = 0.4)"

Together: Child knows problem is hard, so they:
1. Set high decision boundary (HDDM: a = 1.2)
2. Allocate high cognitive control (EVC: control = 0.44)
3. This is an adaptive strategy!
```

---

## Implementation: HDDM for Your Project

### **Using HDDM Python Package:**

```python
import hddm
import pandas as pd
import matplotlib.pyplot as plt

# Load arithmetic task data
data = pd.read_csv('data/arithmetic/arithmetic_task_data.csv')

# Prepare for HDDM (needs specific format)
hddm_data = data[['child_id', 'rt', 'correct']].copy()
hddm_data = hddm_data.rename(columns={
    'child_id': 'subj_idx',
    'correct': 'response'  # 1 for correct, 0 for error
})

# Convert RT to seconds
hddm_data['rt'] = hddm_data['rt'] / 1000.0

# Add difficulty as regressor
hddm_data['difficulty'] = data['difficulty'].values

# ============================================
# FIT HIERARCHICAL DDM
# ============================================

# Model 1: Basic HDDM (no regressors)
print("Fitting basic HDDM...")
m_basic = hddm.HDDM(hddm_data)
m_basic.sample(2000, burn=1000)

# Model 2: HDDM with difficulty affecting drift rate
print("Fitting HDDM with difficulty regressor...")
m_difficulty = hddm.HDDM(hddm_data, depends_on={'v': 'difficulty'})
m_difficulty.sample(2000, burn=1000)

# Compare models
print(f"\nDIC comparison:")
print(f"Basic: {m_basic.dic}")
print(f"With difficulty: {m_difficulty.dic}")
# Lower DIC = better model

# ============================================
# EXTRACT PARAMETERS
# ============================================

# Population-level parameters
params = m_difficulty.get_group_means()
print(f"\nPopulation parameters:")
print(f"  Drift rate intercept: {params['v_Intercept']:.3f}")
print(f"  Drift rate difficulty effect: {params['v_difficulty']:.3f}")
print(f"  Boundary: {params['a']:.3f}")
print(f"  Non-decision time: {params['t']:.3f}")

# Individual parameters
individual_params = m_difficulty.get_individual_parameters()
print(f"\nIndividual drift rates:")
for child_id in individual_params.index[:5]:
    v = individual_params.loc[child_id, 'v']
    a = individual_params.loc[child_id, 'a']
    print(f"  Child {child_id}: v = {v:.3f}, a = {a:.3f}")

# ============================================
# EXTRACT TRIAL-BY-TRIAL UNCERTAINTY
# ============================================

# Method 1: From drift rate
data['drift_rate'] = individual_params.loc[data['child_id'].values, 'v'].values
data['hddm_uncertainty'] = 1 / (1 + data['drift_rate'])

# Method 2: From posterior predictive
# Uncertainty = entropy of predicted choice distribution
def compute_choice_uncertainty(v, a):
    """Compute uncertainty from DDM parameters"""
    # Probability of correct response
    p_correct = 1 / (1 + np.exp(-2 * v * a))
    # Entropy
    if p_correct < 0.01:
        p_correct = 0.01
    elif p_correct > 0.99:
        p_correct = 0.99
    entropy = -p_correct * np.log(p_correct) - (1-p_correct) * np.log(1-p_correct)
    return entropy

data['hddm_uncertainty_entropy'] = data.apply(
    lambda row: compute_choice_uncertainty(row['drift_rate'], 
                                          individual_params.loc[row['child_id'], 'a']),
    axis=1
)

# ============================================
# NOW USE IN BAYESIAN EVC!
# ============================================

from models.bayesian_evc import BayesianEVC

evc_model = BayesianEVC()
evc_results = evc_model.fit(
    data,
    observed_control_col='control_signal',
    reward_col='reward',
    uncertainty_col='hddm_uncertainty'  # Using HDDM uncertainty!
)

print(f"\nBayesian EVC with HDDM uncertainty:")
print(f"  Uncertainty weight (λ): {evc_results['uncertainty_weight']:.4f}")
print(f"  R²: {evc_results['r2']:.4f}")
```

---

## When to Use Which

### **Use HDDM When:**

✅ You have RT and choice data (binary or multi-choice)
✅ You want to understand decision processes
✅ You need to decompose RT into components
✅ You want to measure processing efficiency
✅ Clinical question about impulsivity/caution
✅ Small N per person (10-30 subjects, 100+ trials each)

**Example questions:**
- "Do children with math anxiety have higher decision boundaries?"
- "Does processing speed (drift rate) improve with age?"
- "Are ADHD children impulsive (lower boundaries)?"

---

### **Use Bayesian EVC When:**

✅ You want to predict effort/control allocation
✅ You care about when/why people try hard
✅ You want to test uncertainty-control relationship
✅ You have control proxies (RT, pupil, neural activity)
✅ Clinical question about motivation/effort

**Example questions:**
- "Do children allocate more effort on uncertain problems?"
- "Does uncertainty sensitivity (λ) predict math anxiety?"
- "How do children learn to allocate control efficiently?"

---

### **Use BOTH When:**

✅ You want complete understanding (decision process + control)
✅ You have rich data (RT, accuracy, control proxies)
✅ You want to test if processing efficiency affects control needs

**Example questions:**
- "Do children with low drift rates compensate with high control?"
- "How do decision parameters (HDDM) and control parameters (EVC) jointly predict performance?"

---

## Concrete Recommendation for Your Lab

### **Arithmetic Task Study Design:**

#### **Phase 1: HDDM Analysis**

**Purpose:** Understand basic cognitive processing

```python
# Fit HDDM to arithmetic data
# Extract:
- Drift rates per difficulty level
- Individual differences in processing speed
- Age effects on drift rate
```

**Publications:**
- "Developmental changes in arithmetic processing efficiency" (HDDM focus)

---

#### **Phase 2: Bayesian EVC Analysis**

**Purpose:** Understand control allocation

```python
# Fit Bayesian EVC
# Extract:
- Uncertainty weight (λ) per child
- Control allocation strategies
- Relationship to math anxiety
```

**Publications:**
- "Uncertainty and control in mathematical cognition" (EVC focus)

---

#### **Phase 3: Integration**

**Purpose:** Complete picture

```python
# Use HDDM uncertainty in Bayesian EVC
# Test: Does processing efficiency (v) moderate control allocation?

# Model:
control = f(reward, hddm_uncertainty, drift_rate)

# Question: Do children with low drift rate allocate MORE control to compensate?
```

**Publications:**
- "Integrating decision and control models of mathematical cognition" (top-tier journal!)

---

## Practical Workflow

### **Option 1: HDDM Only (Simpler)**

```bash
# Install HDDM
pip install hddm

# Fit model
python fit_hddm_arithmetic.py

# Analyze
# - Extract drift rates
# - Test age/ability effects
# - Publish!
```

**Timeline:** 2-3 months
**Complexity:** Moderate (HDDM well-documented)

---

### **Option 2: Bayesian EVC Only (Current Path)**

```bash
# You're already doing this!
python3 step5_compare_all_models.py

# Analyze
# - Extract λ parameters
# - Test uncertainty effects
# - Publish!
```

**Timeline:** 1-2 months (nearly done!)
**Complexity:** Low (you've already implemented it)

---

### **Option 3: Both (Best Science)**

```bash
# Phase 1: HDDM
python fit_hddm_arithmetic.py
# → Paper 1

# Phase 2: Bayesian EVC
python fit_bayesian_evc_arithmetic.py
# → Paper 2

# Phase 3: Integration
python fit_integrated_hddm_evc.py
# → Paper 3 (synthesis)
```

**Timeline:** 6-9 months (3 papers!)
**Complexity:** High (but high impact!)

---

## My Recommendation

### **For Your Current Project:**

**Stick with Bayesian EVC, then add HDDM later** ✅

**Reasons:**
1. ✅ You're 90% done with Bayesian EVC
2. ✅ Bayesian EVC directly tests your hypothesis (uncertainty → control)
3. ✅ Can finish and publish in 1-2 months
4. ✅ HDDM can be follow-up study

**Timeline:**
```
Now - Month 2: Finish Bayesian EVC → Paper 1
Month 3-4: Add HDDM → Paper 2
Month 5-6: Integrate both → Paper 3
```

---

### **For Your Lab's Future Work:**

**Use HDDM as standard tool** ✅

**Reasons:**
1. ✅ Perfect for arithmetic tasks (RT + accuracy)
2. ✅ Decompose processing components
3. ✅ Clinical applications (ADHD, anxiety)
4. ✅ Well-validated framework

**Integrate with Bayesian EVC:**
- HDDM provides uncertainty estimates
- EVC uses uncertainty to predict control
- Both together = complete understanding

---

## Comparison Summary Table

| Feature | HDDM | Bayesian EVC | HDDM + EVC |
|---------|------|--------------|------------|
| **Predicts** | RT distribution | Control allocation | Both |
| **Uncertainty** | Output | Input | Output → Input |
| **Temporal** | Can add (via regression) | Via HGF | Both |
| **Individual differences** | Hierarchical Bayes | Hierarchical Bayes | Both |
| **Control allocation** | No | Yes | Yes |
| **Decision process** | Yes | No | Yes |
| **Complexity** | Medium | Low-Medium | High |
| **Interpretability** | High | High | High |
| **Data requirements** | RT + choice | RT + control proxy | RT + choice + control |
| **Software** | HDDM package | Your code | Both |

---

## Bottom Line

### **Should you use HDDM instead of Bayesian EVC?**

**NO - Use both!**

But do them **sequentially**, not simultaneously:

1. **Now:** Finish Bayesian EVC (1-2 months)
   - You're almost done!
   - Clear contribution (λ parameter)
   - Publishable

2. **Next:** Add HDDM (2-3 months)
   - Use HDDM for uncertainty estimation
   - Feed into Bayesian EVC
   - Enhanced paper

3. **Future:** Full integration (3-4 months)
   - HDDM + Bayesian EVC + HGF
   - Comprehensive model
   - Top-tier journal

---

## Resources for HDDM

### **If you want to explore:**

**Software:**
```bash
pip install hddm
```

**Documentation:**
- https://hddm.readthedocs.io/

**Papers:**
1. Wiecki, T. V., et al. (2013). "HDDM: Hierarchical Bayesian estimation of the Drift-Diffusion Model in Python." *Frontiers in Neuroinformatics*, 7, 14.

**Tutorial:**
- HDDM tutorial: https://hddm.readthedocs.io/en/latest/tutorial.html

---

**My advice: Finish Bayesian EVC first, then explore HDDM as enhancement!** 🎯

Want me to create a simple HDDM integration script for future use?

