# Understanding the Control Variable in Bayesian EVC

## What is "Control"?

### **Control is the Observable Dependent Variable**

**Control** (`control_signal` in the data) represents **cognitive control allocation** - how much mental effort/resources someone invests in a task.

```
Control is NOT latent - it's OBSERVABLE behavior!
```

---

## What Control Represents

### **In Real Experiments, Control Can Be Measured As:**

1. **Reaction Time (RT)**
   - Slower RT = more control (more deliberation)
   - Faster RT = less control (more automatic)

2. **Accuracy**
   - Higher accuracy = more control invested
   - Lower accuracy = less control

3. **Neural Activity**
   - DLPFC activation = control allocation
   - ACC activity = control monitoring
   - Pupil dilation = cognitive effort

4. **Behavioral Measures**
   - Error rates
   - Response variability
   - Switch costs

### **In Our Simulation:**

Control is generated based on:
```python
control_signal = baseline + reward_benefit + uncertainty_benefit + noise
```

Where:
- `baseline` = individual's baseline control level (0.3-0.7)
- `reward_benefit` = (reward/10) × evidence_clarity × 0.3
- `uncertainty_benefit` = uncertainty_tolerance × total_uncertainty × 0.3
- `noise` = random variation

**This simulates realistic control allocation behavior!**

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

