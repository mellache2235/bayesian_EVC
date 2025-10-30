# Run Individual Steps Guide

## ⚠️ IMPORTANT: Run Steps in Order

**You MUST run the steps in numerical order (step1 → step2 → step3 → step4 → step5 → step6).**

Each step depends on output files created by previous steps.

## Step Execution Order

```
STEP 1 → STEP 2 → STEP 3 → STEP 4 → STEP 5 → STEP 6
  ↓         ↓         ↓         ↓         ↓         ↓
Data    Unc Est   Trad Fit  Bayes Fit  Compare  Visualize
```

Instead of running the full pipeline, you can run each step individually for easier debugging and inspection.

## Overview

The analysis is broken into 6 independent steps:

1. **Generate Data** - Create experimental dataset
2. **Estimate Uncertainty** - Add Bayesian uncertainty estimates
3. **Fit Traditional EVC** - Train baseline model
4. **Fit Bayesian EVC** - Train uncertainty-aware model
5. **Compare Models** - Statistical comparison
6. **Visualize** - Create all plots

## Quick Start

**Run each step in this exact order:**

```bash
# Step 1: Generate data (MUST RUN FIRST)
python3 step1_generate_data.py

# Step 2: Estimate uncertainty (OPTIONAL - can skip)
python3 step2_estimate_uncertainty.py

# Step 3: Fit traditional EVC model (requires Step 1)
python3 step3_fit_traditional_evc.py

# Step 4: Fit Bayesian EVC model (requires Step 1)
python3 step4_fit_bayesian_evc.py

# Step 5: Compare models (requires Steps 3 and 4)
python3 step5_compare_models.py

# Step 6: Create visualizations (requires Steps 1, 3, 4, 5)
python3 step6_visualize.py
```

## Detailed Instructions

### Step 1: Generate Data (2 minutes)

```bash
python3 step1_generate_data.py
```

**What it does:**
- Generates 6,000 trials (30 subjects × 200 trials)
- Creates 4 experimental blocks with varying uncertainty
- Simulates neural data (DLPFC, ACC, striatum)

**Output files:**
- `data/behavioral_data.csv`
- `data/neural_data.csv`
- `data/summary_statistics.csv`

**What to check:**
- Look at summary statistics by block
- Verify uncertainty varies across blocks
- Check that files were created in `data/` directory

---

### Step 2: Estimate Uncertainty (1 minute)

```bash
python3 step2_estimate_uncertainty.py
```

**What it does:**
- Processes behavioral data with Bayesian estimator
- Adds decision uncertainty (from evidence)
- Adds state uncertainty (from belief updating)
- Computes combined uncertainty and confidence

**Output files:**
- `data/behavioral_data_with_uncertainties.csv`

**What to check:**
- Verify new columns were added
- Check uncertainty statistics (mean, std)
- Ensure values are in reasonable ranges (0-1)

**Note:** This step is optional - the original data already has uncertainty measures. This step adds *additional* Bayesian estimates.

---

### Step 3: Fit Traditional EVC (2-3 minutes)

```bash
python3 step3_fit_traditional_evc.py
```

**What it does:**
- Splits data into train/test sets (70/30 by subject)
- Fits Traditional EVC model (Reward - Effort)
- Evaluates on test set
- Saves model and predictions

**Output files:**
- `results/traditional_evc_model.pkl`
- `results/traditional_evc_results.csv`
- `results/traditional_evc_predictions.csv`

**What to check:**
- Training R² (should be positive, ideally > 0.1)
- Test R² (should be similar to training)
- RMSE (lower is better)
- Correlation (should be positive)

**Troubleshooting:**
- If R² is very negative, the model isn't fitting well
- Check that control_signal has variance in the data
- Try adjusting parameter bounds in the model code

---

### Step 4: Fit Bayesian EVC (2-3 minutes)

```bash
python3 step4_fit_bayesian_evc.py
```

**What it does:**
- Uses same train/test split as Step 3
- Fits Bayesian EVC model (Reward - Effort + Uncertainty Reduction)
- Evaluates on test set
- Saves model and predictions

**Output files:**
- `results/bayesian_evc_model.pkl`
- `results/bayesian_evc_results.csv`
- `results/bayesian_evc_predictions.csv`

**What to check:**
- Compare R² to Traditional EVC (should be higher)
- Check uncertainty_weight parameter (should be > 0)
- Verify RMSE is lower than Traditional EVC
- Correlation should be higher

**Key insight:**
The uncertainty_weight parameter shows how much people value reducing uncertainty beyond just getting rewards.

---

### Step 5: Compare Models (< 1 minute)

```bash
python3 step5_compare_models.py
```

**What it does:**
- Loads results from both models
- Compares performance metrics
- Calculates percentage improvements
- Determines which model is better

**Output files:**
- `results/model_comparison.csv`

**What to check:**
- Which model wins on each metric
- Percentage improvements
- Overall assessment (should favor Bayesian EVC)

**Expected result:**
Bayesian EVC should show superior performance, especially:
- Higher R²
- Lower RMSE
- Higher correlation

---

### Step 6: Visualize (1-2 minutes)

```bash
python3 step6_visualize.py
```

**What it does:**
- Creates 6 publication-ready plots
- Saves all figures as PNG files
- Uses non-interactive backend (no display needed)

**Output files:**
- `results/figures/01_model_comparison.png`
- `results/figures/02_uncertainty_effects.png`
- `results/figures/03_block_effects.png`
- `results/figures/04_model_fit_metrics.png`
- `results/figures/05_neural_correlates.png`
- `results/figures/06_individual_differences.png`

**What to check:**
- Open each PNG file to view
- Model comparison: Bayesian should be closer to diagonal
- Uncertainty effects: Should show positive relationship
- Block effects: Metrics should vary by block
- Model fit: Bayesian bars should be higher (R², corr) or lower (RMSE)

---

## Running Specific Steps Only

You can run any subset of steps. For example:

### Just generate new data:
```bash
python3 step1_generate_data.py
```

### Just refit models (after changing parameters):
```bash
python3 step3_fit_traditional_evc.py
python3 step4_fit_bayesian_evc.py
python3 step5_compare_models.py
```

### Just regenerate visualizations:
```bash
python3 step6_visualize.py
```

## Debugging Tips

### If Step 1 fails:
- Check that `utils/data_generator.py` exists
- Verify `data/` directory can be created
- Try with fewer subjects: edit the script

### If Step 3 or 4 fails:
- Ensure Step 1 completed successfully
- Check that `data/behavioral_data.csv` exists
- Look for optimization warnings
- Try with smaller dataset for testing

### If Step 5 fails:
- Ensure Steps 3 and 4 completed
- Check that result CSV files exist
- Verify no NaN values in results

### If Step 6 fails:
- Ensure all previous steps completed
- Check matplotlib backend (should be 'Agg')
- Verify `results/figures/` directory exists
- Look for specific plot that's failing

## Customization

### Change number of subjects/trials:
Edit `step1_generate_data.py`:
```python
behavioral_data = generator.generate_task_data(
    n_subjects=50,        # Change this
    n_trials_per_subject=300,  # Change this
    n_blocks=4
)
```

### Change train/test split:
Edit `step3_fit_traditional_evc.py` and `step4_fit_bayesian_evc.py`:
```python
n_train = int(len(subjects) * 0.8)  # Change 0.7 to 0.8 for 80/20 split
```

### Change model parameters:
Edit the model initialization in Steps 3 or 4:
```python
model = TraditionalEVC(
    reward_weight=2.0,      # Change these
    effort_cost_weight=0.5,
    effort_exponent=2.5
)
```

## File Dependencies & Execution Order

**CRITICAL: Steps must be run in numerical order (1 → 2 → 3 → 4 → 5 → 6)**

```
Step 1: step1_generate_data.py
  ↓ Creates: data/behavioral_data.csv, data/neural_data.csv
  ↓ Required by: Steps 2, 3, 4, 6
  
Step 2: step2_estimate_uncertainty.py [OPTIONAL]
  ↓ Creates: data/behavioral_data_with_uncertainties.csv
  ↓ Requires: Step 1
  ↓ Can be skipped if using original uncertainty columns
  
Step 3: step3_fit_traditional_evc.py
  ↓ Creates: results/traditional_evc_model.pkl, results/traditional_evc_results.csv
  ↓ Requires: Step 1
  ↓ Required by: Steps 5, 6
  
Step 4: step4_fit_bayesian_evc.py
  ↓ Creates: results/bayesian_evc_model.pkl, results/bayesian_evc_results.csv
  ↓ Requires: Step 1
  ↓ Required by: Steps 5, 6
  
Step 5: step5_compare_models.py
  ↓ Creates: results/model_comparison.csv
  ↓ Requires: Steps 3 and 4
  ↓ Required by: Step 6
  
Step 6: step6_visualize.py
  ↓ Creates: results/figures/*.png (6 plots)
  ↓ Requires: Steps 1, 3, 4, 5
```

**Minimum required sequence:**
- Steps 1, 3, 4, 5, 6 (Step 2 is optional)

## Expected Runtime

- Step 1: ~2 minutes
- Step 2: ~1 minute
- Step 3: ~2-3 minutes
- Step 4: ~2-3 minutes
- Step 5: <1 minute
- Step 6: ~1-2 minutes

**Total: ~10 minutes**

## Output Summary

After running all steps, you'll have:

### Data (3 files)
- Behavioral data
- Neural data
- Summary statistics

### Results (7 files)
- 2 model files (.pkl)
- 4 results CSVs
- 1 comparison CSV

### Figures (6 files)
- All publication-ready plots

## Next Steps

After running all steps:

1. **Examine results** in `results/model_comparison.csv`
2. **View figures** in `results/figures/`
3. **Analyze predictions** in `results/*_predictions.csv`
4. **Modify parameters** and re-run specific steps
5. **Apply to real data** by replacing Step 1

## Getting Help

- Check console output for error messages
- Verify all dependencies installed: `pip install -r requirements.txt`
- Read `README.md` for overview
- Check `INDEX.md` for navigation
- See `BAYESIAN_INFERENCE_EXPLAINED.md` for Bayesian details

---

**Ready to start?** Run: `python3 step1_generate_data.py`

