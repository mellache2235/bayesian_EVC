# Working with Existing Datasets

## The Challenge

Your Bayesian EVC models require:
1. **Control** (often not directly measured)
2. **Uncertainty** (rarely explicitly measured)
3. Reward magnitude (usually available)
4. Accuracy (usually available)

**Question:** Can you test your theory without collecting new data?

**Answer:** YES! With some creativity and assumptions.

---

## Solution 1: Use Proxies for Missing Variables

### **A. Control Proxies**

#### **Option 1A: Reaction Time as Control Proxy**

**Assumption:** Slower RT = more control (deliberation)

```python
# Load existing dataset
data = pd.read_csv('existing_study.csv')

# Use RT as control proxy
data['control_proxy'] = normalize(data['rt'])

# Or inverse (faster = more control for some tasks)
data['control_proxy'] = normalize(-data['rt'])  # Negative so higher = more control

# Fit models
trad_model.fit(data, observed_control_col='control_proxy', ...)
```

**When this works:**
- Tasks where deliberation helps (e.g., difficult decisions)
- Stroop, flanker, go/no-go tasks
- Cognitive control paradigms

**When this fails:**
- Speed-accuracy tradeoff tasks
- Tasks where control speeds up responses (e.g., prepared responses)

---

#### **Option 1B: Accuracy as Control Proxy**

**Assumption:** Higher accuracy = more control invested

```python
# Use accuracy as control proxy
data['control_proxy'] = data['accuracy']

# Or trial-by-trial accuracy (0/1)
# Smooth it with running average
data['control_proxy'] = data['accuracy'].rolling(window=10, center=True).mean()
```

**When this works:**
- Tasks with clear accuracy measure
- Difficulty varies across trials
- Control directly improves performance

**When this fails:**
- Ceiling effects (everyone gets 100%)
- Floor effects (too hard)
- Accuracy depends more on stimulus than control

---

#### **Option 1C: Composite Control Measure (Best!)**

**Combine multiple proxies:**

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare proxies
scaler = StandardScaler()
proxies = scaler.fit_transform(data[[
    'rt',                    # Reaction time
    'accuracy',              # Trial accuracy
    'error_rate',            # Error rate (if available)
    'response_variability'   # RT variability (if available)
]])

# Extract first principal component
pca = PCA(n_components=1)
data['control_proxy'] = pca.fit_transform(proxies).flatten()

print(f"Control proxy explains {pca.explained_variance_ratio_[0]:.1%} of variance")
```

**Advantages:**
- More robust than single proxy
- Captures multiple aspects of control
- Less sensitive to measurement noise

---

#### **Option 1D: Neural Activity as Control (If Available)**

**If dataset has fMRI/EEG:**

```python
# Use DLPFC activity as control proxy
data['control_proxy'] = normalize(data['dlpfc_bold'])

# Or ACC activity
data['control_proxy'] = normalize(data['acc_bold'])

# Or composite neural measure
data['control_proxy'] = (
    0.5 * normalize(data['dlpfc_bold']) +
    0.5 * normalize(data['acc_bold'])
)
```

**Gold standard if available!**

---

### **B. Uncertainty Proxies**

#### **Option 2A: Infer from Confidence Ratings**

**If dataset has confidence ratings:**

```python
# Uncertainty = inverse of confidence
data['uncertainty_proxy'] = 1 - normalize(data['confidence'])

# Or use confidence directly (scale to 0-1)
data['uncertainty_proxy'] = 1 - (data['confidence'] / data['confidence'].max())
```

**When this works:**
- Participants gave confidence ratings
- Confidence is well-calibrated
- Metacognitive awareness is good

---

#### **Option 2B: Infer from RT Variability**

**Assumption:** More variable RT = more uncertain

```python
# Compute RT variability per subject/condition
data['rt_std'] = data.groupby(['subject_id', 'condition'])['rt'].transform('std')
data['uncertainty_proxy'] = normalize(data['rt_std'])

# Or trial-by-trial variability (rolling window)
data['uncertainty_proxy'] = data.groupby('subject_id')['rt'].transform(
    lambda x: x.rolling(window=10, center=True).std()
)
```

---

#### **Option 2C: Use HGF to Infer Uncertainty (Recommended!)**

**Infer uncertainty from trial outcomes:**

```python
from models.hgf_uncertainty import HGFSequentialEstimator

# Initialize HGF
hgf = HGFSequentialEstimator(
    kappa_2=1.0,
    omega_2=-4.0,
    omega_3=-6.0
)

# Process data to infer uncertainty
data_with_uncertainty = hgf.process_subject_data(
    data,
    subject_col='subject_id',
    outcome_col='accuracy'  # Use trial outcomes
)

# Now have:
# - state_uncertainty: Uncertainty about task state
# - volatility: Estimated environmental volatility
# - learning_rate: Trial-by-trial learning rate
```

**Advantages:**
- Principled Bayesian inference
- No need for explicit uncertainty measure
- Captures trial-by-trial dynamics
- Estimates volatility automatically

**This is the best option if you don't have confidence ratings!**

---

#### **Option 2D: Task-Based Uncertainty**

**Use task design to define uncertainty:**

```python
# If task has blocks with different volatility
data['uncertainty_proxy'] = data['block'].map({
    'stable': 0.2,      # Low uncertainty block
    'volatile': 0.8     # High uncertainty block
})

# Or based on trial history
# High uncertainty after rule changes
data['trials_since_change'] = ...  # Compute from task structure
data['uncertainty_proxy'] = np.exp(-data['trials_since_change'] / 10)

# Or based on stimulus ambiguity
data['uncertainty_proxy'] = 1 - data['stimulus_clarity']
```

**When this works:**
- Task explicitly manipulates uncertainty
- Clear block structure
- Known rule changes

---

## Solution 2: Partial Model Testing

### **Test What You Can with Available Data**

#### **Scenario A: Have Control, Missing Uncertainty**

```python
# Test Traditional EVC only
trad_model = TraditionalEVC()
results = trad_model.fit(data, observed_control_col='control_proxy', ...)

print(f"Traditional EVC R²: {results['r2']:.3f}")
print(f"Reward weight: {results['reward_weight']:.3f}")

# Interpretation: Does reward and accuracy predict control?
```

**What you learn:**
- Whether basic EVC framework works
- Parameter estimates for reward/effort sensitivity
- Baseline for comparison

---

#### **Scenario B: Have Uncertainty, Missing Control**

```python
# Can't fit EVC directly, but can test correlations

# Does uncertainty correlate with behavioral measures?
corr_rt = data['uncertainty_proxy'].corr(data['rt'])
corr_acc = data['uncertainty_proxy'].corr(data['accuracy'])

print(f"Uncertainty-RT correlation: {corr_rt:.3f}")
print(f"Uncertainty-Accuracy correlation: {corr_acc:.3f}")

# Regression: Does uncertainty predict behavior?
from sklearn.linear_model import LinearRegression

X = data[['reward', 'uncertainty_proxy']].values
y = data['rt'].values

model = LinearRegression()
model.fit(X, y)

print(f"Uncertainty coefficient: {model.coef_[1]:.3f}")
```

**What you learn:**
- Whether uncertainty matters for behavior
- Preliminary evidence for Bayesian EVC
- Motivation for full study

---

## Solution 3: Leverage Existing Datasets Creatively

### **A. Meta-Analysis Approach**

**Combine multiple existing datasets:**

```python
# Dataset 1: Has RT and accuracy
dataset1 = load_dataset1()
dataset1['control_proxy'] = normalize(dataset1['rt'])

# Dataset 2: Has neural activity
dataset2 = load_dataset2()
dataset2['control_proxy'] = normalize(dataset2['dlpfc_bold'])

# Dataset 3: Has confidence ratings
dataset3 = load_dataset3()
dataset3['uncertainty_proxy'] = 1 - dataset3['confidence']

# Fit models on each
results = []
for data in [dataset1, dataset2, dataset3]:
    if 'control_proxy' in data.columns:
        result = fit_model(data)
        results.append(result)

# Meta-analyze
mean_r2 = np.mean([r['r2'] for r in results])
print(f"Meta-analytic R²: {mean_r2:.3f}")
```

---

### **B. Re-analyze Published Data**

**Many papers share data openly:**

```python
# Example: Stroop task dataset from OpenNeuro
# Has: RT, accuracy, fMRI (DLPFC, ACC)

# Extract control from DLPFC
control = normalize(fmri_data['dlpfc_bold'])

# Infer uncertainty from ACC
uncertainty = normalize(fmri_data['acc_bold'])

# Define reward (Stroop: correct = reward)
reward = fmri_data['accuracy']

# Fit Bayesian EVC
bayes_model.fit(
    fmri_data,
    observed_control_col='dlpfc_bold',
    uncertainty_col='acc_bold',
    reward_col='accuracy'
)
```

**Good sources:**
- OpenNeuro (fMRI datasets)
- OSF (Open Science Framework)
- Journal supplementary materials
- Author data sharing

---

### **C. Simulation-Guided Analysis**

**Use simulation to validate proxy approach:**

```python
# Step 1: Generate data with known control and uncertainty
sim_data = generate_simulation()

# Step 2: Test if RT proxy recovers true control
control_true = sim_data['control_signal']
control_proxy = normalize(sim_data['rt'])

correlation = np.corrcoef(control_true, control_proxy)[0, 1]
print(f"RT proxy correlation with true control: {correlation:.3f}")

# Step 3: If correlation is good (>0.5), use RT proxy on real data
if correlation > 0.5:
    print("✓ RT is valid control proxy for this task!")
    # Apply to real data
    real_data['control_proxy'] = normalize(real_data['rt'])
```

---

## Solution 4: Minimal Data Collection

### **Add-On Study to Existing Paradigm**

**Collect ONLY the missing measures:**

#### **Option A: Online Confidence Ratings**

```python
# Existing task: Stroop (has RT, accuracy)
# Add: Confidence rating after each trial

# Minimal addition:
for trial in trials:
    # Existing
    response = show_stroop_stimulus()
    rt = measure_rt()
    accuracy = check_accuracy(response)
    
    # NEW: Add confidence rating (1 second)
    confidence = ask_confidence()  # "How confident? 1-5"
    
    # Now have uncertainty proxy!
    uncertainty = 1 - confidence
```

**Cost:** +1 second per trial, minimal burden

---

#### **Option B: Pupillometry Add-On**

```python
# Existing task: Any cognitive task
# Add: Eye tracker (pupil dilation)

# Pupil dilation = cognitive effort/control
control_proxy = normalize(pupil_dilation)

# Pupil variability = uncertainty
uncertainty_proxy = normalize(pupil_variability)
```

**Cost:** Eye tracker (~$5K), no task changes needed

---

#### **Option C: Brief Questionnaire**

```python
# After each block, ask:
# "How uncertain did you feel about the rules in this block?"
# Scale: 1 (very certain) to 7 (very uncertain)

block_uncertainty = questionnaire_response

# Apply to all trials in block
data.loc[data['block'] == block_id, 'uncertainty_proxy'] = block_uncertainty
```

**Cost:** ~30 seconds per block

---

## Recommended Strategy

### **Tier 1: No New Data Collection**

```
1. Find existing dataset with RT and accuracy
2. Use RT as control proxy
3. Infer uncertainty using HGF from trial outcomes
4. Fit Traditional and Bayesian EVC
5. Compare models
6. Report as "proof of concept" with caveats
```

**Publishable in:** Computational journals, conference papers

---

### **Tier 2: Minimal Data Collection**

```
1. Use existing task paradigm
2. Add confidence ratings (1 sec/trial)
3. Use confidence for uncertainty
4. Use RT or neural activity for control
5. Fit both models
6. Full comparison
```

**Publishable in:** Cognitive neuroscience journals

---

### **Tier 3: Purpose-Built Study (Gold Standard)**

```
1. Design task to manipulate uncertainty
2. Collect fMRI (DLPFC = control, ACC = uncertainty)
3. Collect confidence ratings
4. Measure RT, accuracy, pupil
5. Fit models with multiple control/uncertainty measures
6. Full validation
```

**Publishable in:** Nature Neuroscience, PNAS, etc.

---

## Example: Working with Existing Stroop Dataset

### **Typical Stroop Dataset Has:**
- ✅ RT (reaction time)
- ✅ Accuracy (correct/incorrect)
- ✅ Condition (congruent/incongruent)
- ✅ Subject ID
- ❌ Control (not measured)
- ❌ Uncertainty (not measured)

### **What You Can Do:**

```python
import pandas as pd
from models.traditional_evc import TraditionalEVC
from models.bayesian_evc import BayesianEVC
from models.hgf_uncertainty import HGFSequentialEstimator

# Load Stroop data
data = pd.read_csv('stroop_data.csv')

# 1. Create control proxy from RT
# (Slower RT = more control in Stroop)
data['control_proxy'] = normalize(data['rt'])

# 2. Infer uncertainty using HGF
hgf = HGFSequentialEstimator()
data = hgf.process_subject_data(
    data,
    subject_col='subject_id',
    outcome_col='accuracy'
)

# 3. Define reward (correct = rewarded)
data['reward'] = data['accuracy']

# 4. Fit Traditional EVC
trad_model = TraditionalEVC()
trad_results = trad_model.fit(
    data,
    observed_control_col='control_proxy',
    reward_col='reward',
    accuracy_col='accuracy'
)

# 5. Fit Bayesian EVC
bayes_model = BayesianEVC()
bayes_results = bayes_model.fit(
    data,
    observed_control_col='control_proxy',
    reward_col='reward',
    accuracy_col='accuracy',
    uncertainty_col='state_uncertainty',  # From HGF
    confidence_col='confidence'  # From HGF (1 - uncertainty)
)

# 6. Compare
print(f"Traditional R²: {trad_results['r2']:.3f}")
print(f"Bayesian R²: {bayes_results['r2']:.3f}")
print(f"Uncertainty weight: {bayes_results['uncertainty_weight']:.3f}")

if bayes_results['r2'] > trad_results['r2']:
    print("✓ Uncertainty improves control prediction!")
```

**Result:** Publishable analysis using existing data!

---

## Bottom Line

### **Can you test your theory without new data collection?**

**YES!** With caveats:

✅ **Use proxies** for missing variables (RT, confidence, HGF)
✅ **Re-analyze existing datasets** (OpenNeuro, OSF)
✅ **Meta-analyze** multiple datasets
✅ **Start with proof-of-concept**, then collect ideal data

### **When you NEED new data collection:**

❌ No existing datasets in your domain
❌ Proxies are too noisy/invalid
❌ Want gold-standard validation
❌ Need to manipulate uncertainty experimentally

### **Recommended Path:**

```
Phase 1: Proof of concept with existing data + proxies
         ↓ (Publish in computational journal)
Phase 2: Minimal data collection (add confidence ratings)
         ↓ (Publish in cognitive journal)
Phase 3: Full study with fMRI and designed task
         ↓ (Publish in Nature Neuroscience)
```

**You can make progress at EVERY phase!** 🎯


