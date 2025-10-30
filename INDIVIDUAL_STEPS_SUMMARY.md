# Individual Steps - Complete Summary

## ✅ What Was Created

I've broken down the full pipeline into **6 independent scripts** that you can run individually. This makes debugging much easier!

## 📋 The 6 Steps

### Step 1: Generate Data
**File:** `step1_generate_data.py`  
**Runtime:** ~2 minutes  
**What it does:** Creates experimental dataset with 6,000 trials  
**Output:** `data/behavioral_data.csv`, `data/neural_data.csv`

### Step 2: Estimate Uncertainty (Optional)
**File:** `step2_estimate_uncertainty.py`  
**Runtime:** ~1 minute  
**What it does:** Adds Bayesian uncertainty estimates  
**Output:** `data/behavioral_data_with_uncertainties.csv`

### Step 3: Fit Traditional EVC
**File:** `step3_fit_traditional_evc.py`  
**Runtime:** ~2-3 minutes  
**What it does:** Trains baseline model (Reward - Effort)  
**Output:** `results/traditional_evc_model.pkl`, `results/traditional_evc_results.csv`

### Step 4: Fit Bayesian EVC
**File:** `step4_fit_bayesian_evc.py`  
**Runtime:** ~2-3 minutes  
**What it does:** Trains uncertainty-aware model  
**Output:** `results/bayesian_evc_model.pkl`, `results/bayesian_evc_results.csv`

### Step 5: Compare Models
**File:** `step5_compare_models.py`  
**Runtime:** <1 minute  
**What it does:** Statistical comparison of both models  
**Output:** `results/model_comparison.csv`

### Step 6: Visualize
**File:** `step6_visualize.py`  
**Runtime:** ~1-2 minutes  
**What it does:** Creates 6 publication-ready plots  
**Output:** `results/figures/*.png` (6 files)

## 🚀 How to Run

### Run All Steps in Order:
```bash
python3 step1_generate_data.py
python3 step2_estimate_uncertainty.py  # Optional
python3 step3_fit_traditional_evc.py
python3 step4_fit_bayesian_evc.py
python3 step5_compare_models.py
python3 step6_visualize.py
```

### Or Run Specific Steps:
```bash
# Just regenerate data
python3 step1_generate_data.py

# Just refit models
python3 step3_fit_traditional_evc.py
python3 step4_fit_bayesian_evc.py

# Just update visualizations
python3 step6_visualize.py
```

## 📊 What You Get

After running all steps:

**Data Files (3):**
- Behavioral data with all trial variables
- Neural activity data
- Summary statistics by block

**Model Files (2):**
- Traditional EVC model (saved as .pkl)
- Bayesian EVC model (saved as .pkl)

**Results Files (5):**
- Traditional EVC results and predictions
- Bayesian EVC results and predictions
- Model comparison

**Figures (6):**
1. Model comparison (predicted vs observed)
2. Uncertainty effects on control
3. Block-wise behavioral metrics
4. Model fit metrics comparison
5. Neural-behavioral correlations
6. Individual differences

## 🎯 Key Advantages

### Compared to Full Pipeline:

✅ **Easier Debugging** - See exactly where errors occur  
✅ **Faster Iteration** - Re-run only what you need  
✅ **Better Understanding** - See each step's output  
✅ **More Control** - Modify individual steps easily  
✅ **Inspect Intermediate Results** - Check data at each stage  

## 📖 Documentation

- **Quick Reference:** `STEP_BY_STEP.txt` (1 page)
- **Detailed Guide:** `RUN_STEPS.md` (comprehensive)
- **Full Index:** `INDEX.md` (navigation)

## 🔍 Debugging Tips

### If a step fails:

1. **Check console output** - Error messages are detailed
2. **Verify previous steps completed** - Each step needs prior outputs
3. **Check file existence** - Make sure input files exist
4. **Look at the data** - Open CSV files to inspect

### Common Issues:

**"File not found"**  
→ Run previous steps first

**Negative R² values**  
→ Normal for small samples; check model fitting

**Import errors**  
→ Run `pip install -r requirements.txt`

**Plots don't show**  
→ They're saved to `results/figures/` automatically

## 💡 Pro Tips

1. **First time:** Run all steps in order
2. **Debugging:** Run steps individually
3. **Experimenting:** Modify and re-run specific steps
4. **Quick check:** Look at console output after each step
5. **Verify results:** Check the "What to check" section in RUN_STEPS.md

## 🎓 Learning Path

### For Quick Results:
1. Run all 6 steps
2. Check `results/model_comparison.csv`
3. View plots in `results/figures/`

### For Understanding:
1. Run Step 1, examine the data
2. Run Step 3, check traditional model results
3. Run Step 4, compare with Bayesian results
4. Run Step 5, see the comparison
5. Run Step 6, visualize everything

### For Development:
1. Modify parameters in step files
2. Re-run specific steps
3. Compare results
4. Iterate

## 📝 Next Steps

After running all steps:

1. ✅ Examine `results/model_comparison.csv`
2. ✅ View all plots in `results/figures/`
3. ✅ Check model parameters in results files
4. ✅ Analyze predictions in `*_predictions.csv`
5. ✅ Modify parameters and re-run
6. ✅ Apply to your own data

## 🆘 Getting Help

- **Quick reference:** `STEP_BY_STEP.txt`
- **Detailed guide:** `RUN_STEPS.md`
- **Implementation details:** `IMPLEMENTATION_GUIDE.md`
- **Project overview:** `PROJECT_SUMMARY.md`
- **Navigation:** `INDEX.md`

## ✨ Summary

Instead of one monolithic pipeline, you now have:
- ✅ 6 independent, modular scripts
- ✅ Clear inputs and outputs for each
- ✅ Easy debugging and iteration
- ✅ Complete documentation
- ✅ Tested and working

**Ready to start?** Run: `python3 step1_generate_data.py`

---

**Total runtime for all steps: ~10 minutes**  
**All results automatically saved to `data/` and `results/` directories**

