# HGF Implementation Summary

## ✅ What Was Created

I've implemented a complete **Hierarchical Gaussian Filter (HGF)** for your Bayesian EVC project!

---

## 📦 New Files

### 1. `HGF_IMPLEMENTATION_GUIDE.md`
**Complete guide** covering:
- What HGF is and why it's perfect for your project
- Mathematical framework
- Comparison with simple Bayesian approaches
- Integration with EVC
- Literature review and key papers
- Implementation roadmap

### 2. `models/hgf_uncertainty.py`
**Working implementation** with:
- `HierarchicalGaussianFilter` class
- `HGFSequentialEstimator` for processing trials
- `fit_hgf_parameters()` for parameter estimation
- Complete demo showing it works!

---

## 🎯 Why Use HGF?

### Key Advantages

✅ **Multi-Level Uncertainty**
- Level 1: Observation noise
- Level 2: State uncertainty (which rule?)
- Level 3: Volatility (how fast things change)

✅ **Adaptive Learning**
- Learning rate automatically adjusts to volatility
- Fast learning when environment is volatile
- Slow learning when environment is stable

✅ **Theoretically Principled**
- Optimal Bayesian inference under Gaussian assumptions
- All updates follow Bayes' rule
- Provides full posterior distributions

✅ **Clinically Validated**
- Extensively used in computational psychiatry
- HGF parameters differ in psychiatric conditions
- Strong neural correlates (ACC, insula)

---

## 🚀 How to Use

### Quick Start

```python
from models.hgf_uncertainty import HierarchicalGaussianFilter

# Initialize HGF
hgf = HierarchicalGaussianFilter(
    kappa_2=1.0,    # Coupling strength
    omega_2=-4.0,   # Log-volatility
    omega_3=-6.0    # Volatility of volatility
)

# Process trials
for outcome in outcomes:
    hgf.update(outcome)
    
    # Get uncertainty estimates
    uncertainty = hgf.get_state_uncertainty()
    volatility = hgf.get_volatility()
    confidence = hgf.get_confidence()
    learning_rate = hgf.get_learning_rate()
```

### With Your Data

```python
from models.hgf_uncertainty import HGFSequentialEstimator
import pandas as pd

# Load your data
data = pd.read_csv('data/behavioral_data.csv')

# Initialize estimator
estimator = HGFSequentialEstimator(
    kappa_2=1.0,
    omega_2=-4.0,
    omega_3=-6.0
)

# Process all subjects
results = estimator.process_subject_data(
    data,
    subject_col='subject_id',
    outcome_col='accuracy',
    evidence_col='evidence_clarity'
)

# Now you have HGF estimates for each trial!
print(results[['hgf_state_uncertainty', 'hgf_volatility', 'hgf_confidence']].head())
```

### Integrate with Bayesian EVC

```python
from models.bayesian_evc import BayesianEVC
from models.hgf_uncertainty import HGFSequentialEstimator

# Step 1: Get HGF uncertainty estimates
estimator = HGFSequentialEstimator()
data_with_hgf = estimator.process_subject_data(data)

# Step 2: Use HGF uncertainty in EVC model
evc_model = BayesianEVC()
evc_results = evc_model.fit(
    data_with_hgf,
    uncertainty_col='hgf_combined_uncertainty',  # Use HGF uncertainty!
    confidence_col='hgf_combined_confidence'
)

print(f"Bayesian EVC with HGF R²: {evc_results['r2']:.3f}")
```

---

## 📊 What HGF Provides

### Trial-by-Trial Estimates

For each trial, you get:

| Variable | Description | Use in EVC |
|----------|-------------|------------|
| `hgf_state_estimate` | Estimated probability | Baseline accuracy |
| `hgf_state_uncertainty` | Uncertainty about state | State uncertainty |
| `hgf_volatility` | Environmental volatility | Context for control |
| `hgf_learning_rate` | Adaptive learning rate | How fast to update |
| `hgf_confidence` | Confidence (1 - uncertainty) | Weight rewards |
| `hgf_combined_uncertainty` | Decision + state uncertainty | Total uncertainty for EVC |

---

## 🔬 Tested and Working

The implementation has been tested with:

✅ **Stable environments** - Low volatility, steady learning  
✅ **Volatile environments** - High volatility, fast adaptation  
✅ **Block transitions** - Detects changes in reward probability  
✅ **Learning rate adaptation** - Increases with volatility  

Demo output shows:
- State estimates track true probabilities
- Uncertainty increases in volatile blocks
- Volatility estimate adapts correctly
- Learning rates adjust appropriately

---

## 📈 Expected Benefits for Your Project

### 1. Better Model Fit
- HGF captures adaptive learning
- Accounts for volatility effects
- Should improve R² over simple Bayesian

### 2. Richer Insights
- Track how learning rates change
- Identify volatile vs. stable periods
- Understand individual differences in volatility sensitivity

### 3. Neural Validation
- HGF precision-weighted prediction errors correlate with ACC activity
- Volatility estimates correlate with insula activity
- Can validate your model with fNIRS/fMRI data

### 4. Clinical Applications
- HGF parameters differ in psychiatric conditions
- Can identify atypical uncertainty processing
- Potential for personalized interventions

---

## 🎓 Comparison: Simple Bayesian vs. HGF

| Feature | Simple Bayesian | HGF |
|---------|----------------|-----|
| **Uncertainty levels** | 1-2 | 3 (hierarchical) |
| **Learning rate** | Fixed | Adaptive |
| **Volatility tracking** | ❌ No | ✅ Yes |
| **Computational cost** | Low | Medium |
| **Theoretical foundation** | Good | Excellent |
| **Clinical validation** | Limited | Extensive |
| **Neural correlates** | Some | Strong |
| **Individual differences** | Basic | Rich |

**Recommendation:** Start with simple Bayesian, then upgrade to HGF for richer analysis.

---

## 📚 Key References

### Must-Read Papers

1. **Mathys et al. (2011)** - Original HGF paper
   - "A Bayesian foundation for individual learning under uncertainty"
   - *Frontiers in Human Neuroscience*

2. **Mathys et al. (2014)** - HGF tutorial
   - "Uncertainty in perception and the Hierarchical Gaussian Filter"
   - *Frontiers in Human Neuroscience*

3. **Powers et al. (2017)** - Clinical application
   - "Pavlovian conditioning-induced hallucinations result from overweighting of perceptual priors"
   - *Science*

4. **Iglesias et al. (2013)** - Neural correlates
   - "Hierarchical prediction errors in midbrain and basal forebrain during sensory learning"
   - *Neuron*

### Software Resources

- **TAPAS Toolbox** (MATLAB): https://www.tnu.ethz.ch/en/software/tapas
- **pyhgf** (Python): `pip install pyhgf`
- **hBayesDM** (R/Stan): Includes HGF models

---

## 🛠️ Next Steps

### Immediate (Today)
1. ✅ Review `HGF_IMPLEMENTATION_GUIDE.md`
2. ✅ Test `models/hgf_uncertainty.py` demo
3. ✅ Understand the three levels

### This Week
1. Run HGF on your generated data
2. Compare HGF vs. simple Bayesian uncertainty
3. Visualize learning rates and volatility over time

### Next Week
1. Integrate HGF with Bayesian EVC model
2. Fit HGF parameters to your data
3. Compare model performance

### Future
1. Hierarchical HGF across subjects
2. Individual differences in HGF parameters
3. Neural validation with fNIRS/fMRI
4. Clinical/developmental applications

---

## 💡 Pro Tips

### 1. Parameter Tuning
- `kappa_2` (0.1-5.0): Higher = more influence from volatility
- `omega_2` (-10 to 0): Higher = more baseline volatility
- `omega_3` (-10 to 0): Higher = volatility changes faster

### 2. Interpretation
- High `hgf_volatility` → Environment is changing rapidly
- High `hgf_learning_rate` → Adapting quickly to changes
- High `hgf_state_uncertainty` → Unsure about current state

### 3. Validation
- Check if volatility increases during rule changes
- Verify learning rates adapt appropriately
- Compare with simple Bayesian as baseline

---

## 🎯 Integration with Your Pipeline

### Add HGF Step (Optional)

You can add HGF as an alternative to Step 2:

```bash
# Original Step 2: Simple Bayesian
python3 step2_estimate_uncertainty.py

# OR: New Step 2b: HGF-based uncertainty
python3 step2b_estimate_uncertainty_hgf.py  # Create this!
```

### Or: Use HGF in Analysis

Integrate HGF directly in your analysis scripts:

```python
# In step3_fit_traditional_evc.py or step4_fit_bayesian_evc.py
from models.hgf_uncertainty import HGFSequentialEstimator

# Add HGF estimates
estimator = HGFSequentialEstimator()
data_with_hgf = estimator.process_subject_data(data)

# Use HGF uncertainty in model fitting
model.fit(data_with_hgf, uncertainty_col='hgf_combined_uncertainty')
```

---

## ✨ Summary

**YES, absolutely use HGF for your Bayesian EVC project!**

### What You Get:
✅ **Working implementation** - Ready to use  
✅ **Complete documentation** - Detailed guide  
✅ **Tested code** - Demo shows it works  
✅ **Integration examples** - Easy to incorporate  
✅ **Literature review** - Key papers and resources  

### Why It's Better:
- **Adaptive learning** matches human behavior
- **Multi-level uncertainty** captures complexity
- **Clinically validated** for real applications
- **Neural correlates** for validation

### How to Start:
1. Run the demo: `python3 models/hgf_uncertainty.py`
2. Read the guide: `HGF_IMPLEMENTATION_GUIDE.md`
3. Try on your data: Use `HGFSequentialEstimator`
4. Integrate with EVC: Add HGF uncertainty to model

**Ready to implement?** The code is ready to go! 🚀

---

**Files to check:**
- `HGF_IMPLEMENTATION_GUIDE.md` - Complete guide
- `models/hgf_uncertainty.py` - Working code
- `BAYESIAN_INFERENCE_EXPLAINED.md` - Related approaches

