# Bayesian EVC Project - Complete Index

## 📚 Documentation Files (Start Here!)

| File | Purpose | Read Time |
|------|---------|-----------|
| **[QUICK_START.md](QUICK_START.md)** | Get running in 5 minutes | 2 min |
| **[README.md](README.md)** | Project overview and structure | 5 min |
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | What was implemented and why | 10 min |
| **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** | Detailed usage guide with examples | 20 min |
| **[proposal.md](proposal.md)** | Original research proposal | 10 min |

## 🚀 Executable Scripts

| File | Command | Purpose |
|------|---------|---------|
| **test_implementation.py** | `python3 test_implementation.py` | Quick test (2 min) |
| **main_pipeline.py** | `python3 main_pipeline.py` | Full analysis (10 min) |

## 💻 Source Code

### Models (`models/`)
- **bayesian_uncertainty.py** - Bayesian uncertainty estimation (decision + state)
- **traditional_evc.py** - Baseline EVC model (reward - effort)
- **bayesian_evc.py** - Bayesian EVC model (reward - effort + uncertainty reduction)

### Utilities (`utils/`)
- **data_generator.py** - Generate experimental data with varying uncertainty
- **visualization.py** - Create publication-ready plots

### Configuration
- **requirements.txt** - Python dependencies

## 📊 Data & Results

### Generated Data (`data/`)
- **behavioral_data.csv** - Trial-level behavioral data (6,000 trials)
- **neural_data.csv** - Simulated neural activity (DLPFC, ACC, striatum)
- **summary_statistics.csv** - Block-level summaries

### Analysis Results (`results/`)
- **model_comparison.csv** - Performance metrics (R², RMSE, correlation)
- **model_parameters.csv** - Fitted model parameters
- **predictions.csv** - Trial-level model predictions

### Figures (`results/figures/`)
1. **model_comparison.png** - Predicted vs. observed control
2. **uncertainty_effects.png** - Uncertainty-control relationship
3. **block_effects.png** - Behavioral metrics by block
4. **model_fit_metrics.png** - Model comparison bar charts
5. **neural_correlates.png** - Neural-behavioral correlations
6. **individual_differences.png** - Uncertainty tolerance effects

## 🎯 Quick Navigation

### I want to...

**...understand what this project does**
→ Read [README.md](README.md) then [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

**...run the code immediately**
→ Follow [QUICK_START.md](QUICK_START.md)

**...learn how to use the code**
→ Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)

**...understand the theory**
→ Read [proposal.md](proposal.md)

**...modify the models**
→ Check `models/` directory and [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) "Customization" section

**...generate custom data**
→ See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) "Example 1: Generate Custom Data"

**...create custom visualizations**
→ See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) "Example 4: Custom Visualizations"

**...understand the results**
→ Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) "Expected Outputs" section

## 📖 Reading Order

### For Quick Start (15 minutes)
1. [QUICK_START.md](QUICK_START.md) - 2 min
2. Run `python3 test_implementation.py` - 2 min
3. Run `python3 main_pipeline.py` - 10 min
4. Browse `results/figures/` - 1 min

### For Understanding (45 minutes)
1. [README.md](README.md) - 5 min
2. [proposal.md](proposal.md) - 10 min
3. [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 10 min
4. [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - 20 min

### For Development (2+ hours)
1. All documentation above
2. Read source code with comments
3. Experiment with examples
4. Modify and extend

## 🔑 Key Concepts

### Traditional EVC
```
EVC = Expected_Reward - Effort_Cost
```
Control allocation based on reward-effort trade-off.

### Bayesian EVC (This Project)
```
Bayesian_EVC = Confidence×Expected_Reward - Effort_Cost + Uncertainty_Reduction_Benefit
```
Control allocation that explicitly values uncertainty reduction.

### Two Types of Uncertainty
1. **Decision Uncertainty** - From evidence clarity
2. **State Uncertainty** - From beliefs about task rules

## 📈 Project Statistics

- **Total Lines of Code**: ~2,500
- **Number of Functions**: 50+
- **Number of Classes**: 6
- **Documentation Pages**: 5
- **Example Plots**: 6
- **Test Coverage**: Core functionality tested

## ✅ Implementation Checklist

- [x] Dummy data generation
- [x] Bayesian uncertainty estimation
- [x] Traditional EVC model
- [x] Bayesian EVC model
- [x] Model comparison pipeline
- [x] Comprehensive visualizations
- [x] Complete documentation
- [x] Working test suite
- [x] Usage examples

## 🎓 Learning Path

### Beginner
1. Read [QUICK_START.md](QUICK_START.md)
2. Run test script
3. Examine generated plots
4. Read [README.md](README.md)

### Intermediate
1. Read [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
2. Run examples
3. Modify parameters
4. Create custom analyses

### Advanced
1. Read all documentation
2. Study source code
3. Extend models
4. Integrate real data

## 🔧 Customization Points

| What to Customize | Where to Look |
|-------------------|---------------|
| Data generation parameters | `utils/data_generator.py` |
| Model equations | `models/traditional_evc.py`, `models/bayesian_evc.py` |
| Uncertainty estimation | `models/bayesian_uncertainty.py` |
| Visualizations | `utils/visualization.py` |
| Analysis pipeline | `main_pipeline.py` |
| Fitting parameters | Model `fit()` methods |

## 🐛 Debugging Guide

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| "python not found" | Use `python3` |
| Negative R² | Normal for small samples; increase data |
| Plots don't show | Check `results/figures/` directory |
| Slow fitting | Reduce subjects/trials for testing |

## 📞 Support Resources

1. **Documentation**: Read the 5 markdown files
2. **Code Comments**: Extensive docstrings in all modules
3. **Examples**: See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)
4. **Test Script**: Run `test_implementation.py` to verify setup

## 🎯 Success Criteria

You'll know everything is working when:
- ✅ `test_implementation.py` shows "ALL TESTS PASSED!"
- ✅ `main_pipeline.py` completes without errors
- ✅ 6 plots appear in `results/figures/`
- ✅ Bayesian EVC shows better R² than Traditional EVC
- ✅ Results match expected patterns

## 🚀 Next Steps After Setup

1. **Validate**: Run test script and full pipeline
2. **Explore**: Examine generated data and plots
3. **Learn**: Read documentation thoroughly
4. **Experiment**: Try examples from implementation guide
5. **Customize**: Modify parameters and models
6. **Extend**: Add new features or analyses
7. **Apply**: Use with real experimental data

## 📝 Citation

If using this implementation, please cite:
- Shenhav, A., et al. (2013). The expected value of control. *Neuron*.
- Your research paper using this implementation

## 🏁 Getting Started Right Now

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test everything works
python3 test_implementation.py

# 3. Run full analysis
python3 main_pipeline.py

# 4. Check results
ls results/figures/
```

---

**Status**: ✅ Complete and tested
**Version**: 1.0
**Last Updated**: 2025

**Ready to begin?** → Start with [QUICK_START.md](QUICK_START.md)

