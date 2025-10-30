# Bayesian EVC Pipeline

## Quick Start

Run the complete pipeline:
```bash
python run_pipeline.py
```

Or run individual steps:
```bash
python step1_load_data.py
python step2_run_bayesian_updates.py
python step3_compute_evc.py
python step4_visualize.py
```

## Where is Bayesian Inference?

See `BAYESIAN_INFERENCE_EXPLAINED.md` for details.

**Short answer:** In `src/pipeline.py`:
- Lines 114-129: `BayesianUncertaintyTracker.update()` - Bayesian belief updates
- Lines 111-113: Belief decay and evidence accumulation
- Lines 89-96: Uncertainty quantification from beliefs

## Files

- `run_pipeline.py` - Run complete pipeline
- `step1_load_data.py` - Load and validate data
- `step2_run_bayesian_updates.py` - Run Bayesian belief updates
- `step3_compute_evc.py` - Compute EVC scores
- `step4_visualize.py` - Create visualizations
- `BAYESIAN_INFERENCE_EXPLAINED.md` - Detailed explanation of Bayesian inference
- `LITERATURE_REVIEW.md` - Related approaches (DDM, HGF, etc.)
- `HGF_IMPLEMENTATION_GUIDE.md` - Guide for Hierarchical Gaussian Filter
- `proposal.md` - Original research proposal

## Requirements

```bash
pip install -r requirements.txt
```

