# Bayesian EVC Pipeline

## Quick Start

**Run steps in order (1 → 2 → 3 → 4 → 5 → 6):**

```bash
python3 step1_generate_data.py
python3 step2_estimate_uncertainty.py
python3 step3_fit_traditional_evc.py
python3 step4_fit_bayesian_evc.py
python3 step5_compare_models.py
python3 step6_visualize.py
```

**Important:** You must run steps in numerical order. Each step depends on output from previous steps.

## Where is Bayesian Inference?

See `BAYESIAN_INFERENCE_EXPLAINED.md` for details.

Bayesian inference is implemented in:
- `models/bayesian_uncertainty.py` - Bayesian uncertainty estimation
- `models/bayesian_evc.py` - Bayesian EVC model with uncertainty reduction

## Files

- `step1_generate_data.py` - **RUN FIRST** - Generate experimental data
- `step2_estimate_uncertainty.py` - **RUN SECOND** (optional) - Add Bayesian uncertainty estimates
- `step3_fit_traditional_evc.py` - **RUN THIRD** - Fit baseline EVC model
- `step4_fit_bayesian_evc.py` - **RUN FOURTH** - Fit Bayesian EVC model
- `step5_compare_models.py` - **RUN FIFTH** - Compare model performance
- `step6_visualize.py` - **RUN SIXTH** - Create visualizations
- `BAYESIAN_INFERENCE_EXPLAINED.md` - Detailed explanation of Bayesian inference
- `HGF_IMPLEMENTATION_GUIDE.md` - Guide for Hierarchical Gaussian Filter
- `HGF_SUMMARY.md` - Quick HGF overview
- `proposal.md` - Original research proposal

## Requirements

```bash
pip install -r requirements.txt
```

