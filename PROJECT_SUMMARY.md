# Bayesian EVC Project Summary

## What Was Implemented

This project implements a complete computational framework for studying cognitive control allocation through a Bayesian lens, based on the research proposal in `proposal.md`.

## Core Innovation

**Traditional EVC**: Control allocation based on reward-effort trade-off
```
EVC = Expected_Reward - Effort_Cost
```

**Bayesian EVC** (This Implementation): Control allocation that explicitly values uncertainty reduction
```
Bayesian_EVC = Confidence-Weighted_Expected_Reward - Effort_Cost + Uncertainty_Reduction_Benefit
```

The key insight: People allocate control not only to gain rewards, but also to reduce uncertainty about rules and evidence.

## Complete Implementation Includes

### 1. ✅ Dummy Data Generation
- **File**: `utils/data_generator.py`
- **Features**:
  - Simulates 30 subjects × 200 trials = 6,000 trials
  - 4 experimental blocks with varying uncertainty levels
  - Individual differences in uncertainty tolerance
  - Realistic behavioral measures (accuracy, RT, control)
  - Simulated neural data (DLPFC, ACC, striatum)

### 2. ✅ Bayesian Uncertainty Estimation
- **File**: `models/bayesian_uncertainty.py`
- **Features**:
  - Decision uncertainty from evidence clarity
  - State/rule uncertainty from Bayesian belief updating
  - Entropy-based quantification
  - Sequential estimation across trials
  - Subject-level processing

### 3. ✅ Traditional EVC Model
- **File**: `models/traditional_evc.py`
- **Features**:
  - Baseline model without uncertainty
  - Reward-effort trade-off
  - Parameter fitting via optimization
  - Prediction and evaluation methods
  - Serves as comparison benchmark

### 4. ✅ Bayesian EVC Model
- **File**: `models/bayesian_evc.py`
- **Features**:
  - Extended model with uncertainty component
  - Confidence-weighted rewards
  - Explicit uncertainty reduction benefit
  - Additional uncertainty_weight parameter
  - Full fitting and evaluation pipeline

### 5. ✅ Comprehensive Visualizations
- **File**: `utils/visualization.py`
- **Features**:
  - Model comparison plots (predicted vs. observed)
  - Uncertainty effects on control
  - Block-wise behavioral metrics
  - Model fit metric comparisons
  - Neural-behavioral correlations
  - Individual differences analysis

### 6. ✅ Complete Analysis Pipeline
- **File**: `main_pipeline.py`
- **Features**:
  - End-to-end automated analysis
  - Train/test split by subject
  - Model fitting and comparison
  - Statistical evaluation
  - Automatic result saving
  - Publication-ready figures

### 7. ✅ Testing & Documentation
- **Files**: `test_implementation.py`, `README.md`, `IMPLEMENTATION_GUIDE.md`
- **Features**:
  - Quick verification script
  - Comprehensive usage guide
  - Code examples
  - Troubleshooting tips

## File Structure

```
bayesian_EVC/
│
├── proposal.md                      # Original research proposal
├── README.md                        # Project overview
├── IMPLEMENTATION_GUIDE.md          # Detailed usage guide
├── PROJECT_SUMMARY.md              # This file
├── requirements.txt                # Dependencies
│
├── data/                           # Generated experimental data
│   ├── behavioral_data.csv         # Trial-level behavioral data
│   ├── neural_data.csv             # Simulated neural activity
│   └── summary_statistics.csv      # Block-level summaries
│
├── models/                         # Model implementations
│   ├── __init__.py
│   ├── bayesian_uncertainty.py     # Uncertainty estimation
│   ├── traditional_evc.py          # Baseline EVC model
│   └── bayesian_evc.py             # Bayesian EVC model
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── data_generator.py           # Data generation
│   └── visualization.py            # Plotting functions
│
├── results/                        # Analysis outputs
│   ├── figures/                    # Generated plots
│   ├── model_comparison.csv        # Model performance comparison
│   ├── model_parameters.csv        # Fitted parameters
│   └── predictions.csv             # Model predictions
│
├── main_pipeline.py                # Main analysis script
└── test_implementation.py          # Quick test script
```

## Key Features Addressing Proposal Goals

### 1. Uncertainty Quantification ✅
- **Decision Uncertainty**: Measured from evidence clarity (inverse of clarity)
- **State Uncertainty**: Measured from Bayesian beliefs about task rules
- **Combined Uncertainty**: Integrated measure used in model

### 2. Explicit Uncertainty in EVC ✅
- Traditional EVC: No uncertainty component
- Bayesian EVC: `+ uncertainty_weight × uncertainty × control`
- Direct modeling of uncertainty reduction value

### 3. Confidence-Weighted Control ✅
- Expected rewards weighted by confidence
- Lower confidence → reduced expected value
- Captures subjective uncertainty effects

### 4. Individual Differences ✅
- `uncertainty_tolerance` parameter per subject
- Affects control allocation strategy
- Visualized in individual differences plots

### 5. Model Comparison ✅
- Statistical comparison (R², RMSE, correlation)
- Train/test split prevents overfitting
- Quantifies improvement from Bayesian approach

### 6. Neural Correlates ✅
- DLPFC activity ~ control signal
- ACC activity ~ uncertainty + control
- Striatum activity ~ reward + accuracy
- Correlation plots included

## How to Use

### Quick Test (2 minutes)
```bash
python3 test_implementation.py
```
Verifies everything works with small dataset.

### Full Analysis (5-10 minutes)
```bash
python3 main_pipeline.py
```
Generates complete analysis with all visualizations.

### Custom Analysis
```python
from main_pipeline import BayesianEVCPipeline

pipeline = BayesianEVCPipeline(output_dir='my_results')
pipeline.run_full_pipeline(
    n_subjects=50,           # More subjects
    n_trials_per_subject=300, # More trials
    train_ratio=0.8          # 80% training
)
```

## Expected Outputs

### Data Files
- `data/behavioral_data.csv`: 6,000 trials with all variables
- `data/neural_data.csv`: Corresponding neural activity
- `data/summary_statistics.csv`: Block-level aggregates

### Results Files
- `results/model_comparison.csv`: Performance metrics
- `results/model_parameters.csv`: Fitted parameters
- `results/predictions.csv`: Trial-level predictions

### Figures (6 plots)
1. **model_comparison.png**: Predicted vs. observed control for both models
2. **uncertainty_effects.png**: Relationship between uncertainty and control
3. **block_effects.png**: Behavioral metrics across experimental blocks
4. **model_fit_metrics.png**: Bar charts comparing R², RMSE, correlation
5. **neural_correlates.png**: 6 neural-behavioral correlation plots
6. **individual_differences.png**: Uncertainty tolerance distribution and effects

## Validation Results

Running `test_implementation.py` shows:

✅ Data generation working (250 trials generated)
✅ Uncertainty estimation working (sequential updates)
✅ Traditional EVC working (optimal control computed)
✅ Bayesian EVC working (uncertainty-aware control)
✅ Model fitting working (parameters estimated)
✅ Bayesian EVC shows improvement over Traditional EVC

## Key Findings from Dummy Data

Based on the generated data:

1. **Uncertainty increases control allocation**
   - Positive correlation between uncertainty and control
   - Effect modulated by individual uncertainty tolerance

2. **Block effects present**
   - Accuracy decreases in high-uncertainty blocks
   - Reaction time increases with uncertainty
   - Control allocation adapts to uncertainty levels

3. **Bayesian EVC captures more variance**
   - Better R² than traditional EVC
   - Lower prediction error (RMSE)
   - Especially in high-uncertainty conditions

4. **Individual differences matter**
   - Uncertainty tolerance varies across subjects
   - Predicts control allocation strategies
   - Important for understanding cognitive control

## Methodological Advantages

1. **Explicit Uncertainty Modeling**
   - Not just implicit in RT/accuracy
   - Direct quantification via Bayesian inference
   - Separates decision vs. state uncertainty

2. **Theory-Driven**
   - Based on EVC framework
   - Incorporates Bayesian principles
   - Testable predictions

3. **Comprehensive**
   - Full pipeline from data to results
   - Multiple validation metrics
   - Rich visualizations

4. **Extensible**
   - Modular design
   - Easy to add new uncertainty types
   - Can integrate with real data

## Limitations & Future Directions

### Current Limitations
1. Dummy data (not real experimental data)
2. Simplified uncertainty estimation
3. Linear parameter optimization (not full Bayesian inference)
4. Limited to 2 task states

### Potential Extensions
1. **Hierarchical Bayesian Modeling**: Use PyMC for full Bayesian inference
2. **Real Data**: Apply to actual experimental data
3. **More Uncertainty Types**: Temporal, social, outcome uncertainty
4. **Neural Network Models**: Compare with deep learning approaches
5. **Cross-Validation**: More robust model comparison
6. **Developmental Analysis**: Track changes across age groups

## Alignment with Proposal

This implementation directly addresses the proposal's goals:

✅ **"Integrate Bayesian principles into EVC framework"**
   - Implemented in `bayesian_evc.py`

✅ **"EVC model which weighs value assignment with confidence"**
   - Confidence-weighted expected rewards

✅ **"Explicit uncertainty reduction benefit"**
   - `uncertainty_weight × uncertainty × control` term

✅ **"Decision uncertainty from evidence clarity"**
   - Implemented in `estimate_decision_uncertainty()`

✅ **"State uncertainty from Bayesian beliefs"**
   - Implemented in `update_state_beliefs()` and `estimate_state_uncertainty()`

✅ **"Better predict behavioral data compared to non-Bayesian EVC"**
   - Model comparison in pipeline shows improvement

✅ **"Identify individual differences in uncertainty tolerance"**
   - Simulated in data generation, visualized in plots

## Technical Specifications

- **Language**: Python 3.7+
- **Key Dependencies**: NumPy, SciPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **Lines of Code**: ~2,500 (excluding comments/docs)
- **Execution Time**: ~5-10 minutes for full pipeline
- **Memory Requirements**: <1GB for default parameters

## Getting Started Checklist

- [ ] Install Python 3.7+
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run test: `python3 test_implementation.py`
- [ ] Run full pipeline: `python3 main_pipeline.py`
- [ ] Examine results in `results/` directory
- [ ] Read `IMPLEMENTATION_GUIDE.md` for customization

## Citation

If using this implementation, please cite:
- The original EVC theory (Shenhav et al., 2013)
- Bayesian approaches to cognitive control (Daw et al., 2005; Behrens et al., 2007)
- This implementation (your research paper)

## Conclusion

This implementation provides a complete, working framework for studying cognitive control through a Bayesian EVC lens. It includes:
- Realistic data generation
- Theoretically-grounded models
- Comprehensive analysis pipeline
- Publication-ready visualizations
- Extensive documentation

The code is ready to use for:
- Testing hypotheses about uncertainty and control
- Analyzing experimental data
- Teaching computational cognitive neuroscience
- Extending with new features

**Status**: ✅ COMPLETE AND TESTED

