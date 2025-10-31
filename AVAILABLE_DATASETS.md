# Available Public Datasets for Testing Bayesian EVC

## Overview

This document lists publicly available datasets that can be used to test the Bayesian EVC framework. Datasets are organized by what measures they contain and how applicable they are to your research.

---

## 🌟 Highly Recommended Datasets

### **1. OpenNeuro - Cognitive Control fMRI Datasets**

**Website:** https://openneuro.org/

**Search Terms:** "Stroop", "flanker", "cognitive control", "DLPFC", "ACC"

#### **Example Datasets:**

**a) Multi-Source Interference Task (MSIT)**
- **OpenNeuro ID:** ds000164
- **URL:** https://openneuro.org/datasets/ds000164
- **What it has:**
  - ✅ fMRI data (DLPFC, ACC activity)
  - ✅ Reaction times
  - ✅ Accuracy
  - ✅ Task conditions (congruent/incongruent)
- **What's missing:**
  - ❌ Explicit uncertainty measures
  - ❌ Confidence ratings
- **How to use:**
  - Extract control from DLPFC activity
  - Infer uncertainty using HGF from trial outcomes
  - Fit Bayesian EVC

**b) Stroop Task Datasets**
- **Search:** "Stroop" on OpenNeuro
- **Multiple datasets available**
- **Typical measures:**
  - ✅ RT, accuracy
  - ✅ fMRI (some datasets)
  - ✅ Conflict conditions
- **Use for:** Testing control allocation under conflict

---

### **2. Human Connectome Project (HCP)**

**Website:** https://www.humanconnectome.org/

**What it has:**
- ✅ Massive fMRI dataset (1000+ subjects)
- ✅ Multiple cognitive tasks
- ✅ High-quality neural data
- ✅ Behavioral measures (RT, accuracy)

**Relevant Tasks:**
- **Gambling Task** - reward processing, decision-making
- **Working Memory Task** - cognitive control
- **Relational Processing** - cognitive flexibility

**How to access:**
1. Register at https://db.humanconnectome.org/
2. Download behavioral + fMRI data
3. Free for research use

**How to use:**
- Extract control from DLPFC activity
- Use task difficulty as uncertainty proxy
- Fit models to predict neural/behavioral measures

---

### **3. Confidence Database (Fleming Lab)**

**Website:** http://www.metacoglab.org/

**Paper:** "Confidence Database" (Fleming et al.)

**What it has:**
- ✅ **Confidence ratings** (KEY!)
- ✅ Perceptual decision-making data
- ✅ RT, accuracy
- ✅ Multiple studies compiled

**Perfect for:**
- Using confidence as uncertainty measure
- Testing Bayesian EVC with explicit uncertainty

**How to access:**
- Contact Fleming lab or check publications
- Often shared on OSF or lab website

---

### **4. Open Science Framework (OSF)**

**Website:** https://osf.io/

**Search Strategy:**

#### **Search 1: "cognitive control reaction time"**
- Filter: Public datasets
- Look for: Stroop, flanker, task-switching studies

#### **Search 2: "decision making confidence"**
- Filter: Has data
- Look for: Confidence ratings, uncertainty measures

#### **Search 3: "reward learning volatility"**
- Filter: Neuroscience
- Look for: Probabilistic learning tasks

**Example Projects:**
- Search for "Shenhav" (EVC theory author) - may have shared data
- Search for "Botvinick" (cognitive control) - lab datasets
- Search for "Daw" (decision-making) - computational modeling datasets

---

### **5. Neurosynth / NeuroVault**

**Neurosynth:** https://neurosynth.org/
**NeuroVault:** https://neurovault.org/

**What it has:**
- ✅ Meta-analytic brain maps
- ✅ Statistical maps from published studies
- ✅ Coordinates for DLPFC, ACC activation

**Use for:**
- Validating neural correlates
- Meta-analysis of control-related activation
- ROI definition for your own studies

---

## 📊 Specific Dataset Recommendations

### **Dataset 1: Probabilistic Reversal Learning**

**Type:** Behavioral + fMRI
**Where to find:** OpenNeuro, search "reversal learning"

**What it has:**
- ✅ Trial-by-trial choices
- ✅ Outcomes (reward/no reward)
- ✅ Volatility manipulation (rule reversals)
- ✅ fMRI (some datasets)

**Why it's good:**
- Clear uncertainty manipulation
- Can use HGF to infer uncertainty
- Reward is explicit

**Example analysis:**
```python
# Load reversal learning data
data = load_reversal_learning_data()

# Infer uncertainty using HGF
hgf = HGFSequentialEstimator()
data = hgf.process_subject_data(data, outcome_col='reward_received')

# Extract control from RT or neural activity
data['control_proxy'] = normalize(data['rt'])

# Fit Bayesian EVC
bayes_model.fit(
    data,
    observed_control_col='control_proxy',
    reward_col='reward_magnitude',
    uncertainty_col='state_uncertainty'
)
```

---

### **Dataset 2: Two-Armed Bandit Tasks**

**Type:** Behavioral
**Where to find:** OSF, many available

**What it has:**
- ✅ Choice data (left/right)
- ✅ Reward outcomes
- ✅ Often has volatility blocks
- ✅ Sometimes confidence ratings

**Why it's good:**
- Simple, clean design
- Clear reward structure
- Can manipulate uncertainty via volatility

**Search terms:**
- "two-armed bandit"
- "multi-armed bandit"
- "explore exploit"

---

### **Dataset 3: Perceptual Decision-Making**

**Type:** Behavioral + sometimes neural
**Where to find:** OSF, lab websites

**What it has:**
- ✅ RT, accuracy
- ✅ **Confidence ratings** (often!)
- ✅ Stimulus difficulty levels
- ✅ Sometimes pupillometry

**Why it's good:**
- Confidence = uncertainty measure
- Difficulty = control demand
- Clean experimental design

**Key labs:**
- Fleming Lab (metacognition)
- Pouget Lab (probabilistic inference)
- Shadlen Lab (decision-making)

---

### **Dataset 4: Stroop Task with fMRI**

**Type:** fMRI + behavioral
**Where to find:** OpenNeuro

**What it has:**
- ✅ RT, accuracy
- ✅ DLPFC, ACC activity
- ✅ Conflict manipulation (congruent/incongruent)

**Why it's good:**
- Classic control task
- Neural measures of control (DLPFC)
- Neural measures of conflict (ACC)

**How to use:**
```python
# Extract control from DLPFC
control = normalize(fmri_data['dlpfc_bold'])

# Extract uncertainty from ACC
uncertainty = normalize(fmri_data['acc_bold'])

# Reward = accuracy (correct = rewarded)
reward = fmri_data['accuracy']

# Fit Bayesian EVC
bayes_model.fit(
    fmri_data,
    observed_control_col='dlpfc_bold',
    uncertainty_col='acc_bold',
    reward_col='accuracy'
)
```

---

## 🔍 How to Search for Datasets

### **Step-by-Step Search Strategy:**

#### **1. OpenNeuro Search**

Go to: https://openneuro.org/

**Search queries:**
```
"Stroop"
"flanker"
"cognitive control"
"decision making"
"reward learning"
"DLPFC"
"anterior cingulate"
```

**Filter by:**
- Has fMRI data
- Has behavioral data
- Number of subjects (>20 recommended)

---

#### **2. OSF Search**

Go to: https://osf.io/search/

**Search queries:**
```
"cognitive control" + "reaction time" + "data"
"decision making" + "confidence" + "dataset"
"reward learning" + "volatility"
"Stroop task" + "data"
```

**Filter by:**
- Public projects
- Has files/data
- Recent (last 5 years)

---

#### **3. Google Dataset Search**

Go to: https://datasetsearch.research.google.com/

**Search queries:**
```
"cognitive control fMRI"
"decision making confidence ratings"
"Stroop task dataset"
"reward learning behavioral data"
```

---

#### **4. Paper Supplementary Materials**

**Strategy:**
1. Find relevant papers on Google Scholar
2. Check supplementary materials
3. Look for data sharing statements
4. Contact authors if data not public

**Key papers to check:**
- Shenhav et al. (2013) - Original EVC paper
- Shenhav et al. (2017) - EVC review
- Holroyd & Yeung (2012) - ACC and control
- Daw et al. (2006) - Model-based learning

---

## 📥 How to Download and Use

### **General Workflow:**

#### **1. Download Data**

```bash
# For OpenNeuro datasets
pip install openneuro-py
openneuro-py download --dataset=ds000164

# For OSF datasets
# Use web interface or osf-cli
```

#### **2. Load and Inspect**

```python
import pandas as pd
import numpy as np

# Load behavioral data
data = pd.read_csv('downloaded_data/behavioral.csv')

# Inspect columns
print(data.columns)
print(data.head())

# Check what you have
print(f"Has RT: {'rt' in data.columns or 'reaction_time' in data.columns}")
print(f"Has accuracy: {'accuracy' in data.columns or 'correct' in data.columns}")
print(f"Has confidence: {'confidence' in data.columns}")
```

#### **3. Prepare for EVC Analysis**

```python
from models.hgf_uncertainty import HGFSequentialEstimator
from models.bayesian_evc import BayesianEVC

# Standardize column names
data = data.rename(columns={
    'reaction_time': 'rt',
    'correct': 'accuracy',
    'subject': 'subject_id'
})

# Create control proxy
data['control_proxy'] = normalize(data['rt'])

# Infer uncertainty if not available
if 'confidence' not in data.columns:
    hgf = HGFSequentialEstimator()
    data = hgf.process_subject_data(
        data,
        subject_col='subject_id',
        outcome_col='accuracy'
    )
    data['uncertainty_proxy'] = data['state_uncertainty']
else:
    data['uncertainty_proxy'] = 1 - normalize(data['confidence'])

# Fit Bayesian EVC
model = BayesianEVC()
results = model.fit(
    data,
    observed_control_col='control_proxy',
    reward_col='reward',  # or 'accuracy' if no explicit reward
    accuracy_col='accuracy',
    uncertainty_col='uncertainty_proxy',
    confidence_col='confidence'
)

print(f"Bayesian EVC R²: {results['r2']:.3f}")
print(f"Uncertainty weight: {results['uncertainty_weight']:.3f}")
```

---

## 🎯 Quick Start: Recommended First Dataset

### **Start Here: OpenNeuro MSIT Dataset**

**Why this one:**
- ✅ Free and public
- ✅ Well-documented
- ✅ Has neural data (DLPFC, ACC)
- ✅ Has behavioral data (RT, accuracy)
- ✅ Cognitive control task
- ✅ Multiple subjects

**Download:**
```bash
# Install openneuro-py
pip install openneuro-py

# Download dataset
openneuro-py download --dataset=ds000164
```

**Analyze:**
```python
# See WORKING_WITH_EXISTING_DATA.md for complete analysis code
```

---

## 📚 Additional Resources

### **Tutorials and Guides:**

1. **OpenNeuro Tutorial**
   - https://openneuro.org/docs

2. **OSF Data Sharing Guide**
   - https://help.osf.io/

3. **HCP Data Access**
   - https://www.humanconnectome.org/study/hcp-young-adult/document/quick-reference-guide

### **Relevant Papers with Shared Data:**

1. **Shenhav et al. (2013)** - "The Expected Value of Control"
   - Check supplementary materials

2. **Fleming & Dolan (2012)** - "The neural basis of metacognitive ability"
   - Confidence database

3. **Daw et al. (2011)** - "Model-based influences on humans' choices"
   - Two-step task data (often shared)

---

## 🚀 Action Plan

### **Week 1: Explore**
- [ ] Browse OpenNeuro for cognitive control datasets
- [ ] Search OSF for behavioral datasets
- [ ] Identify 2-3 candidate datasets

### **Week 2: Download & Inspect**
- [ ] Download selected datasets
- [ ] Inspect data structure
- [ ] Check what measures are available

### **Week 3: Prepare**
- [ ] Create control proxies
- [ ] Infer uncertainty (HGF or confidence)
- [ ] Standardize data format

### **Week 4: Analyze**
- [ ] Fit Traditional EVC
- [ ] Fit Bayesian EVC
- [ ] Compare models
- [ ] Generate plots

### **Week 5: Document**
- [ ] Write up results
- [ ] Create visualizations
- [ ] Prepare for publication

---

## 💡 Tips for Success

### **1. Start Simple**
- Begin with behavioral-only dataset
- Use RT as control proxy
- Use HGF for uncertainty

### **2. Document Everything**
- Keep track of preprocessing steps
- Note any assumptions made
- Save all analysis scripts

### **3. Validate Proxies**
- Check if RT correlates with task difficulty
- Verify HGF uncertainty makes sense
- Plot relationships before modeling

### **4. Be Transparent**
- Acknowledge proxy limitations
- Report sensitivity analyses
- Discuss alternative interpretations

---

## 📧 Getting Help

### **If you can't find suitable data:**

1. **Post on forums:**
   - Neurostars: https://neurostars.org/
   - Reddit: r/neuroscience, r/cognitivescience

2. **Contact authors:**
   - Email corresponding authors of relevant papers
   - Ask if they'd share data

3. **Collaborate:**
   - Find labs collecting relevant data
   - Offer to analyze their data

4. **Collect minimal data:**
   - See WORKING_WITH_EXISTING_DATA.md for minimal collection strategies

---

## Summary

**Best starting points:**
1. 🥇 **OpenNeuro** - fMRI + behavioral, free, well-documented
2. 🥈 **OSF** - Behavioral datasets, often has confidence
3. 🥉 **HCP** - Large-scale, high-quality, requires registration

**What you need minimum:**
- RT and accuracy (can proxy control)
- Trial outcomes (can infer uncertainty with HGF)
- Subject IDs (for proper analysis)

**You can start testing your theory TODAY!** 🎯

