# Understanding the Control Variable in Bayesian EVC

## What is "Control"?

### **Control is the Observable Dependent Variable**

**Control** (`control_signal` in the data) represents **cognitive control allocation** - how much mental effort/resources someone invests in a task.

```
Control is NOT latent - it's OBSERVABLE behavior!
```

---

## What Control Represents

### **⚠️ CRITICAL DISTINCTION: Real Experiments vs. Our Simulation**

---

### **In REAL Experiments:**

**Control is LATENT** - we can't directly observe it!

We **infer** control from behavioral/neural **proxies**:

1. **Reaction Time (RT)**
   - Slower RT → infer more control (more deliberation)
   - Faster RT → infer less control (more automatic)
   - **Formula**: `control ∝ RT` or `control = f(RT)`

2. **Accuracy**
   - Higher accuracy → infer more control invested
   - Lower accuracy → infer less control
   - **Formula**: `control ∝ accuracy`

3. **Neural Activity**
   - DLPFC activation → control allocation
   - ACC activity → control monitoring
   - Pupil dilation → cognitive effort
   - **Formula**: `control ∝ BOLD signal in DLPFC`

4. **Composite Measures**
   - Combine multiple proxies: `control = w₁×RT + w₂×accuracy + w₃×neural`
   - Use factor analysis or PCA to extract control

**The Challenge:** Control is **unobservable** - we only see its effects!

```
Real World:
┌─────────────────────────────────────────────────────┐
│  Control (LATENT - unobservable)                    │
│         ↓                                           │
│  Manifests as:                                      │
│    - Reaction Time                                  │
│    - Accuracy                                       │
│    - Neural Activity                                │
│    - Pupil Dilation                                 │
│         ↓                                           │
│  We INFER control from these proxies                │
└─────────────────────────────────────────────────────┘
```

---

### **In OUR Simulation:**

**Control is DIRECTLY GENERATED** - we know the "ground truth"!

This is a **simplification** for testing the EVC framework:

```python
# Step 1: GENERATE control directly (we know the true value)
control_signal = baseline + reward_benefit + uncertainty_benefit + noise

# Step 2: Control CAUSES RT and accuracy (not the other way around!)
accuracy_prob = evidence_clarity × (0.5 + 0.5 × control_signal)
rt = base_rt + uncertainty_rt - control_signal × 100  # more control → faster
```

**Where:**
- `baseline` = individual's baseline control level (0.3-0.7)
- `reward_benefit` = (reward/10) × evidence_clarity × 0.3
- `uncertainty_benefit` = uncertainty_tolerance × total_uncertainty × 0.3
- `noise` = random variation (~N(0, 0.1))

**Key Point:** In simulation, control is **known** and **causes** RT/accuracy, not inferred from them!

```
Our Simulation:
┌─────────────────────────────────────────────────────┐
│  Control (GENERATED - we know it!)                  │
│         ↓                                           │
│  Causes:                                            │
│    - Reaction Time (faster with more control)      │
│    - Accuracy (higher with more control)           │
│         ↓                                           │
│  We TEST if EVC can predict the control we know    │
└─────────────────────────────────────────────────────┘
```

---

### **Why This Matters:**

#### **In Real Experiments:**
```
RT, Accuracy, Neural → INFER Control → Test EVC predictions
```
- Control must be **extracted** from proxies
- Measurement error in control
- Need to validate control measure

#### **In Our Simulation:**
```
GENERATE Control → Causes RT/Accuracy → Test EVC predictions
```
- Control is **known ground truth**
- No measurement error
- Can directly test EVC framework

**Advantage of Simulation:** We can test if EVC models work **when we know the true control**!

---

### **The Causal Structure:**

#### **Real World (What We Think Happens):**
```
Task Demands → Person Decides Control → Behavior (RT, Accuracy, Neural)
     ↑                    ↑
  Reward            Uncertainty
  Accuracy          
```

#### **Our Simulation (What We Generate):**
```
Reward + Accuracy + Uncertainty → Control Signal → RT, Accuracy
                                       ↓
                              (We know this value!)
```

#### **What EVC Models Do:**
```
Reward + Accuracy + Uncertainty → EVC Formula → Predicted Control
                                                        ↓
                                        Compare to Observed Control
```

**The Test:** Can EVC formula recover the control signal we generated?

---

## The Modeling Framework

### **What We're Doing:**

```
┌─────────────────────────────────────────────────────────────┐
│                    OBSERVED DATA                             │
│  - Reward magnitude (trial-level)                           │
│  - Evidence clarity (trial-level)                           │
│  - Uncertainty (trial-level)                                │
│  - Control signal (trial-level) ← DEPENDENT VARIABLE        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    EVC MODELS                                │
│  Traditional: Control = f(reward, accuracy)                 │
│  Bayesian:    Control = f(reward, accuracy, uncertainty)    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODEL FITTING                             │
│  Find parameters (β_r, β_e, β_u) that best predict          │
│  observed control from reward, accuracy, uncertainty         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION                                │
│  - R²: How well does model predict control?                 │
│  - Correlation: Do predictions track observations?          │
│  - RMSE: How far off are predictions?                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Control is NOT Latent - Here's Why

### **Latent Variables** (Hidden, Unobservable):
- ❌ Uncertainty (must be inferred from behavior)
- ❌ Beliefs (internal mental states)
- ❌ Volatility (rate of environmental change)
- ❌ Value representations (internal valuations)

### **Observable Variables** (Can Be Measured):
- ✅ **Control** (RT, accuracy, neural activity)
- ✅ Reward (given by experimenter)
- ✅ Accuracy (correct/incorrect responses)
- ✅ Reaction time (measured directly)

**Control is observable** - we can measure it through behavior and brain activity!

---

## The EVC Formula and Control

### **Traditional EVC:**

```
EVC = β_r × Reward × Accuracy - β_e × Control^α

Optimal Control: c* = argmax_c EVC(c)
```

**What this means:**
- EVC is the **value** of allocating control
- Control is chosen to **maximize** EVC
- Model predicts: "Given reward and accuracy, how much control should I allocate?"

**Predicted Control:**
```python
control_predicted = baseline + (β_r × reward × accuracy) / (2 × β_e)
```

### **Bayesian EVC (Our Innovation):**

```
EVC = β_r × Reward × Accuracy - β_e × Control^α + β_u × Uncertainty × Control

Optimal Control: c* = argmax_c EVC(c)
```

**What's new:**
- Added **uncertainty benefit** term
- Control reduces uncertainty (valuable!)
- Model predicts: "Given reward, accuracy, AND uncertainty, how much control?"

**Predicted Control:**
```python
control_predicted = baseline + (β_r × reward × accuracy + β_u × uncertainty) / (2 × β_e)
```

**Key insight:** Uncertainty weight `β_u` tells us **how much people value reducing uncertainty**!

---

## What We're Testing

### **Research Question:**

> "Does accounting for uncertainty improve prediction of cognitive control allocation?"

### **Hypothesis:**

Traditional EVC:
```
Control ~ Reward × Accuracy
```

Bayesian EVC:
```
Control ~ Reward × Accuracy + Uncertainty
```

**Prediction:** Bayesian EVC should have:
- Higher R² (better prediction)
- Lower RMSE (more accurate)
- Positive β_u (uncertainty matters!)

---

## The Correlation and Error Tests

### **What We're Correlating:**

```
Observed Control  vs.  Predicted Control
      ↓                       ↓
  (from data)          (from EVC model)
```

**NOT:**
```
Control vs. Uncertainty  ← This is just a bivariate correlation
```

**BUT:**
```
Observed Control vs. Model-Predicted Control
```

Where model-predicted control comes from:
- Traditional: `f(reward, accuracy, β_r, β_e)`
- Bayesian: `f(reward, accuracy, uncertainty, β_r, β_e, β_u)`

### **Metrics We Compute:**

1. **R² (Coefficient of Determination)**
   ```
   R² = 1 - (SS_residual / SS_total)
   ```
   - Measures: "What proportion of variance in control is explained by the model?"
   - Range: -∞ to 1.0
   - Interpretation:
     - R² = 1.0: Perfect prediction
     - R² = 0.0: Model = predicting mean
     - R² < 0.0: Model worse than mean

2. **Correlation (Pearson r)**
   ```
   r = corr(observed_control, predicted_control)
   ```
   - Measures: "Do predictions track observations?"
   - Range: -1 to +1
   - Interpretation:
     - r = 1.0: Perfect linear relationship
     - r = 0.0: No relationship
     - r can be positive even if R² is negative!

3. **RMSE (Root Mean Squared Error)**
   ```
   RMSE = sqrt(mean((observed - predicted)²))
   ```
   - Measures: "How far off are predictions on average?"
   - Range: 0 to ∞
   - Interpretation:
     - RMSE = 0: Perfect prediction
     - Lower is better

---

## Parameter Convergence

### **What Are We Fitting?**

**Traditional EVC Parameters:**
- `baseline` (β₀): Baseline control level
- `reward_weight` (β_r): Sensitivity to reward
- `effort_cost_weight` (β_e): Cost of effort
- `effort_exponent` (α): Effort cost function shape

**Bayesian EVC Parameters:**
- All of the above PLUS:
- `uncertainty_weight` (β_u): **Value of uncertainty reduction** ← KEY!

### **How Fitting Works:**

```python
# Start with initial guesses
params_initial = [0.5, 1.0, 1.0, 0.5, 2.0]

# Optimization loop
for iteration in range(max_iterations):
    # 1. Compute predicted control with current parameters
    predicted = model.predict_control(data, params)
    
    # 2. Compute error
    error = mean((observed - predicted)²)
    
    # 3. Update parameters to reduce error
    params = update_params(params, gradient(error))
    
    # 4. Check convergence
    if change_in_params < threshold:
        break  # Converged!
```

### **Convergence Means:**

✅ **Parameters have stabilized** - they're not changing much between iterations
✅ **Found local optimum** - can't improve fit by small parameter changes
✅ **Model is "trained"** - ready to evaluate on test data

### **What to Look For in Convergence Plots:**

**Good Convergence:**
```
Parameter Value
    │     ╱─────────  ← Flat line (converged)
    │    ╱
    │   ╱
    │  ╱
    └──────────────────> Iteration
```

**Poor Convergence:**
```
Parameter Value
    │   ╱╲╱╲╱╲╱╲  ← Oscillating (not converged)
    │  ╱  ╲  ╱
    │ ╱    ╲╱
    └──────────────────> Iteration
```

**Still Changing:**
```
Parameter Value
    │              ╱  ← Still increasing (need more iterations)
    │            ╱
    │          ╱
    │        ╱
    └──────────────────> Iteration
```

---

## Running the Convergence Visualization

### **Command:**

```bash
python3 visualize_parameter_convergence.py
```

### **What It Does:**

1. Loads behavioral data
2. Fits Traditional EVC while tracking parameters at each iteration
3. Fits Bayesian EVC while tracking parameters at each iteration
4. Plots parameter evolution for both models
5. Creates comparison plot
6. Saves to `results/convergence/`

### **Output:**

Three plots:
1. `traditional_convergence.png` - Traditional EVC parameter trajectories
2. `bayesian_convergence.png` - Bayesian EVC parameter trajectories
3. `convergence_comparison.png` - Side-by-side comparison

### **Interpretation:**

**Check for:**
- ✅ Do parameters flatten out? (convergence)
- ✅ How many iterations to converge? (efficiency)
- ✅ Are final values reasonable? (validity)
- ✅ Is uncertainty_weight > 0? (uncertainty matters!)

---

## Key Takeaways

### **1. Control is Observable**
- NOT a latent variable
- Can be measured through behavior and neural activity
- It's what we're trying to predict

### **2. EVC Models Predict Control**
- Traditional: From reward and accuracy only
- Bayesian: From reward, accuracy, AND uncertainty
- Parameters are fitted to maximize prediction accuracy

### **3. We Test Model Predictions**
- Compare predicted vs. observed control
- Use R², correlation, RMSE
- Bayesian should outperform Traditional

### **4. Parameter Convergence Shows Optimization**
- Flat lines = converged
- Oscillations = problems
- Uncertainty weight (β_u) is key parameter to watch

### **5. The Research Question**
- Does uncertainty matter for control allocation?
- If β_u > 0 and Bayesian R² > Traditional R², then YES!

---

## Conceptual Summary

```
┌──────────────────────────────────────────────────────────┐
│  REAL WORLD                                              │
│  - People allocate cognitive control                     │
│  - Control depends on rewards, difficulty, uncertainty   │
│  - We measure control via RT, accuracy, brain activity   │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  OUR SIMULATION                                          │
│  - Generate data mimicking real control allocation      │
│  - Control ~ reward + accuracy + uncertainty + noise    │
│  - This gives us "observed control" to predict          │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  EVC MODELS                                              │
│  - Traditional: Predicts control from reward + accuracy │
│  - Bayesian: Adds uncertainty to prediction             │
│  - Fit parameters to match observed control             │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│  EVALUATION                                              │
│  - Does Bayesian predict better than Traditional?       │
│  - Is uncertainty_weight significantly > 0?             │
│  - Do parameters converge to stable values?             │
└──────────────────────────────────────────────────────────┘
```

**Bottom line:** We're testing whether explicitly modeling uncertainty improves our ability to predict cognitive control allocation!

---

## Moving from Simulation to Real Data

### **How to Apply This to Real Experiments:**

#### **Step 1: Measure Control Proxies**

Collect behavioral and neural data:
```python
real_data = {
    'subject_id': [...],
    'trial': [...],
    'reward': [...],           # Experimenter-controlled
    'accuracy': [...],         # Observed (correct/incorrect)
    'rt': [...],              # Observed (milliseconds)
    'dlpfc_activity': [...],  # fMRI BOLD signal
    'pupil_dilation': [...]   # Eye tracking
}
```

#### **Step 2: Extract Control Measure**

**Option A: Single Proxy**
```python
# Use RT as proxy (normalize)
control_observed = (rt - rt.mean()) / rt.std()

# Or use neural activity
control_observed = dlpfc_activity  # Already normalized
```

**Option B: Composite Measure (Better!)**
```python
# Combine multiple proxies
from sklearn.decomposition import PCA

# Stack proxies
X = np.column_stack([
    normalize(rt),
    normalize(accuracy),
    normalize(dlpfc_activity),
    normalize(pupil_dilation)
])

# Extract first principal component
pca = PCA(n_components=1)
control_observed = pca.fit_transform(X).flatten()
```

**Option C: Factor Analysis**
```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=1)
control_observed = fa.fit_transform(X).flatten()
```

#### **Step 3: Estimate Uncertainty**

**Option A: Use HGF (Recommended)**
```python
from models.hgf_uncertainty import HGFSequentialEstimator

estimator = HGFSequentialEstimator()
data_with_uncertainty = estimator.process_subject_data(
    real_data,
    subject_col='subject_id',
    outcome_col='accuracy'
)

# Now have: state_uncertainty, volatility, etc.
```

**Option B: Use Simple Bayesian**
```python
from models.bayesian_uncertainty import SequentialBayesianEstimator

estimator = SequentialBayesianEstimator()
data_with_uncertainty = estimator.process_subject_data(
    real_data,
    subject_col='subject_id',
    evidence_col='rt',  # Use RT as evidence proxy
    outcome_col='accuracy'
)
```

**Option C: Use Confidence Ratings**
```python
# If you collected confidence ratings
uncertainty = 1 - confidence_ratings
```

#### **Step 4: Fit EVC Models**

```python
from models.traditional_evc import TraditionalEVC
from models.bayesian_evc import BayesianEVC

# Fit Traditional EVC
trad_model = TraditionalEVC()
trad_results = trad_model.fit(
    data_with_uncertainty,
    observed_control_col='control_observed',  # Your extracted control
    reward_col='reward',
    accuracy_col='accuracy'
)

# Fit Bayesian EVC
bayes_model = BayesianEVC()
bayes_results = bayes_model.fit(
    data_with_uncertainty,
    observed_control_col='control_observed',
    reward_col='reward',
    accuracy_col='accuracy',
    uncertainty_col='state_uncertainty',  # From HGF or Bayesian estimator
    confidence_col='confidence'
)

# Compare
print(f"Traditional R²: {trad_results['r2']:.3f}")
print(f"Bayesian R²: {bayes_results['r2']:.3f}")
print(f"Uncertainty weight: {bayes_results['uncertainty_weight']:.3f}")
```

#### **Step 5: Validate**

**Check if results make sense:**
```python
# 1. Is uncertainty weight positive and significant?
if bayes_results['uncertainty_weight'] > 0:
    print("✓ Uncertainty matters for control!")

# 2. Does Bayesian outperform Traditional?
if bayes_results['r2'] > trad_results['r2']:
    print("✓ Adding uncertainty improves prediction!")

# 3. Do parameters converge?
# Run visualize_parameter_convergence.py

# 4. Cross-validate
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
r2_scores = []
for train_idx, test_idx in kf.split(subjects):
    # Fit on train, test on test
    # ...
    r2_scores.append(test_r2)

print(f"Cross-validated R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
```

---

### **Key Differences: Simulation vs. Real Data**

| Aspect | Simulation | Real Experiments |
|--------|-----------|------------------|
| **Control** | Directly generated (known) | Latent (must be inferred) |
| **Uncertainty** | Set or computed | Must be estimated (HGF, Bayesian) |
| **Evidence Clarity** | Set in design | Infer from RT/confidence |
| **Ground Truth** | We know true values | We never know true values |
| **Validation** | Can check if we recover truth | Can only check predictive accuracy |
| **Advantage** | Clean test of theory | Tests real human behavior |
| **Limitation** | May not match reality | Measurement error, confounds |

---

### **Recommended Workflow for Real Data**

```
1. Collect Data
   ├── Behavioral (RT, accuracy, confidence)
   ├── Neural (fMRI, EEG, pupil)
   └── Task variables (reward, difficulty)

2. Preprocess
   ├── Remove outliers
   ├── Normalize measures
   └── Check data quality

3. Extract Control
   ├── Option A: Single proxy (RT or neural)
   ├── Option B: PCA/Factor analysis
   └── Option C: Model-based (DDM drift rate)

4. Estimate Uncertainty
   ├── Option A: HGF (best for volatility)
   ├── Option B: Simple Bayesian
   └── Option C: Confidence ratings

5. Fit EVC Models
   ├── Traditional EVC (baseline)
   ├── Bayesian EVC (with uncertainty)
   └── Compare performance

6. Validate
   ├── Cross-validation
   ├── Parameter recovery
   ├── Out-of-sample prediction
   └── Neural validation (fMRI)

7. Report
   ├── Model comparison (R², BIC, AIC)
   ├── Parameter estimates (β_u significance)
   ├── Individual differences
   └── Clinical implications
```

---

### **Example: Real fMRI Study**

**Hypothetical study design:**

```python
# Data structure
real_study = {
    'subject_id': 30 subjects × 200 trials,
    'reward': [1, 2, 5, 10],  # Experimenter-controlled
    'accuracy': [0, 1],        # Observed
    'rt': [300-1500 ms],       # Observed
    'dlpfc_bold': [...],       # fMRI signal (control proxy)
    'acc_bold': [...],         # fMRI signal (uncertainty signal)
    'confidence': [1-5],       # Self-report
}

# Extract control from DLPFC
control_observed = normalize(real_study['dlpfc_bold'])

# Estimate uncertainty from ACC and confidence
uncertainty_observed = (
    0.5 × normalize(real_study['acc_bold']) +
    0.5 × (1 - normalize(real_study['confidence']))
)

# Fit models
bayes_model.fit(
    real_study,
    observed_control_col='control_observed',
    uncertainty_col='uncertainty_observed'
)

# Result: β_u = 0.45 (p < 0.001)
# Interpretation: Uncertainty significantly predicts control allocation!
```

**This would be publishable in Nature Neuroscience!** 🎯📄

