# Quick Start Guide

## Installation (1 minute)

```bash
pip install -r requirements.txt
```

## Test Implementation (2 minutes)

```bash
python3 test_implementation.py
```

Expected output: "ALL TESTS PASSED! ✓"

## Run Full Analysis (5-10 minutes)

```bash
python3 main_pipeline.py
```

This generates:
- `data/` - Experimental data (6,000 trials)
- `results/` - Model comparisons and parameters
- `results/figures/` - 6 publication-ready plots

## Key Files

| File | Purpose |
|------|---------|
| `main_pipeline.py` | Run complete analysis |
| `test_implementation.py` | Quick verification |
| `models/bayesian_evc.py` | Main Bayesian EVC model |
| `models/traditional_evc.py` | Baseline comparison model |
| `utils/data_generator.py` | Generate dummy data |
| `utils/visualization.py` | Create plots |

## Quick Examples

### Generate Data Only
```python
from utils.data_generator import ExperimentalDataGenerator

generator = ExperimentalDataGenerator(seed=42)
data = generator.generate_task_data(n_subjects=30, n_trials_per_subject=200)
generator.save_data(data, neural_data, output_dir='data')
```

### Fit Models Only
```python
import pandas as pd
from models.traditional_evc import TraditionalEVC
from models.bayesian_evc import BayesianEVC

data = pd.read_csv('data/behavioral_data.csv')

# Traditional
trad = TraditionalEVC()
trad_results = trad.fit(data)

# Bayesian
bayes = BayesianEVC()
bayes_results = bayes.fit(data, uncertainty_col='total_uncertainty')

print(f"Traditional R²: {trad_results['r2']:.3f}")
print(f"Bayesian R²: {bayes_results['r2']:.3f}")
```

### Visualize Only
```python
from utils.visualization import EVCVisualizer

viz = EVCVisualizer()
viz.plot_uncertainty_effects(data, save_path='uncertainty.png')
```

## Understanding the Output

### Console Output
```
STEP 1: GENERATING EXPERIMENTAL DATA
  ✓ Data generation complete! (6000 trials)

STEP 3: FITTING TRADITIONAL EVC MODEL
  Training R²: 0.XXX
  Test R²: 0.XXX

STEP 4: FITTING BAYESIAN EVC MODEL
  Training R²: 0.XXX (should be higher)
  Test R²: 0.XXX (should be higher)

STEP 5: MODEL COMPARISON
  ✓ Bayesian EVC shows superior predictive performance!
```

### Generated Files
- `behavioral_data.csv`: All trial data
- `model_comparison.csv`: Performance metrics
- `model_parameters.csv`: Fitted parameters
- `predictions.csv`: Model predictions
- `figures/*.png`: 6 visualization plots

## Troubleshooting

**Problem**: Import errors
**Solution**: `pip install -r requirements.txt`

**Problem**: "python not found"
**Solution**: Use `python3` instead

**Problem**: Negative R² values
**Solution**: Normal for dummy data; increase sample size or adjust parameters

**Problem**: Plots don't show
**Solution**: Check `results/figures/` directory for saved images

## Next Steps

1. ✅ Run test script
2. ✅ Run full pipeline
3. 📖 Read `IMPLEMENTATION_GUIDE.md` for details
4. 🔧 Customize parameters in `main_pipeline.py`
5. 📊 Analyze results in `results/` directory
6. 🎨 Modify visualizations in `utils/visualization.py`

## Key Concepts

**Traditional EVC**: `Reward - Effort`
**Bayesian EVC**: `Confidence×Reward - Effort + Uncertainty_Reduction`

The Bayesian model explicitly values reducing uncertainty, leading to better predictions of control allocation.

## Support

- Check `README.md` for overview
- Read `IMPLEMENTATION_GUIDE.md` for detailed usage
- Review `PROJECT_SUMMARY.md` for complete description
- Examine code comments and docstrings

---

**Ready to start?** Run: `python3 test_implementation.py`

