# Bayesian EVC Implementation Guide

## Overview

This implementation provides a complete framework for analyzing cognitive control allocation using a Bayesian approach to the Expected Value of Control (EVC) theory. The project includes dummy data generation, model implementations, and comprehensive analysis pipelines.

## Project Structure

```
.
├── proposal.md                    # Original research proposal
├── README.md                      # Project overview
├── requirements.txt               # Python dependencies
├── IMPLEMENTATION_GUIDE.md        # This file
│
├── data/                          # Generated experimental data
│   ├── behavioral_data.csv
│   ├── neural_data.csv
│   └── summary_statistics.csv
│
├── models/                        # Model implementations
│   ├── __init__.py
│   ├── bayesian_uncertainty.py    # Bayesian uncertainty estimation
│   ├── traditional_evc.py         # Traditional EVC model
│   └── bayesian_evc.py            # Bayesian EVC with uncertainty
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── data_generator.py          # Dummy data generation
│   └── visualization.py           # Plotting utilities
│
├── results/                       # Analysis outputs
│   ├── figures/                   # Generated plots
│   ├── model_comparison.csv
│   ├── model_parameters.csv
│   └── predictions.csv
│
├── main_pipeline.py               # Main analysis pipeline
└── test_implementation.py         # Quick test script
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Required packages:
- numpy, scipy, pandas (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (model evaluation)
- pymc, arviz (Bayesian inference - optional for advanced features)

## Quick Start

### 1. Test the Implementation

Run the test script to verify everything works:

```bash
python3 test_implementation.py
```

This will:
- Generate a small dataset (5 subjects, 50 trials each)
- Test uncertainty estimation
- Test both EVC models
- Fit models and compare performance

### 2. Run the Full Pipeline

Execute the complete analysis pipeline:

```bash
python3 main_pipeline.py
```

This will:
1. Generate experimental data (30 subjects, 200 trials each)
2. Split data into train/test sets (70/30)
3. Fit traditional EVC model
4. Fit Bayesian EVC model
5. Compare model performance
6. Generate visualizations
7. Save all results

Results will be saved to:
- `data/` - Raw experimental data
- `results/` - Model outputs and comparisons
- `results/figures/` - All visualizations

## Key Components

### 1. Data Generation (`utils/data_generator.py`)

The `ExperimentalDataGenerator` creates realistic experimental data with:

**Experimental Manipulations:**
- **Block 1**: Low uncertainty (high evidence clarity, stable rules)
- **Block 2**: High evidence uncertainty (low clarity, stable rules)
- **Block 3**: High rule uncertainty (high clarity, unstable rules)
- **Block 4**: High both uncertainties

**Generated Variables:**
- `evidence_clarity`: How clear the evidence is (0-1)
- `rule_stability`: How stable the task rules are (0-1)
- `decision_uncertainty`: Uncertainty about the decision (1 - clarity)
- `state_uncertainty`: Uncertainty about task rules (1 - stability)
- `total_uncertainty`: Combined uncertainty measure
- `control_signal`: Amount of cognitive control allocated (0-1)
- `accuracy`: Trial outcome (0 or 1)
- `reaction_time`: Response time in milliseconds
- `confidence`: Participant's confidence (0-1)

**Individual Differences:**
- `uncertainty_tolerance`: How well each participant handles uncertainty (0-1)

**Neural Data:**
- `dlpfc_activity`: Dorsolateral prefrontal cortex (control-related)
- `acc_activity`: Anterior cingulate cortex (conflict/uncertainty-related)
- `striatal_activity`: Striatum (reward-related)

### 2. Bayesian Uncertainty Estimation (`models/bayesian_uncertainty.py`)

The `BayesianUncertaintyEstimator` implements two types of uncertainty:

**Decision Uncertainty:**
- Based on evidence clarity
- Inspired by drift-diffusion models
- Quantifies difficulty of accumulating evidence
- Measured via entropy and confidence

**State/Rule Uncertainty:**
- Based on beliefs about task rules
- Uses Bayesian belief updating
- Tracks uncertainty about which rule is active
- Measured via entropy of belief distribution

**Key Methods:**
- `estimate_decision_uncertainty()`: Compute evidence-based uncertainty
- `update_state_beliefs()`: Bayesian update of rule beliefs
- `estimate_state_uncertainty()`: Compute rule-based uncertainty
- `estimate_combined_uncertainty()`: Integrate both sources

### 3. Traditional EVC Model (`models/traditional_evc.py`)

The `TraditionalEVC` class implements the baseline model:

**Formula:**
```
EVC = (Reward_Weight × Expected_Reward) - (Effort_Cost_Weight × Effort_Cost)

where:
  Expected_Reward = P(success) × Reward_Magnitude
  P(success) = baseline_accuracy + (1 - baseline_accuracy) × control × 0.5
  Effort_Cost = control^effort_exponent
```

**Parameters:**
- `reward_weight`: Weight for reward component (default: 1.0)
- `effort_cost_weight`: Weight for effort cost (default: 1.0)
- `effort_exponent`: Exponent for effort function (default: 2.0)

**Key Methods:**
- `compute_evc()`: Calculate EVC for given control level
- `optimal_control()`: Find control that maximizes EVC
- `fit()`: Fit parameters to observed data
- `predict_control()`: Predict optimal control for trials
- `evaluate()`: Evaluate model performance

### 4. Bayesian EVC Model (`models/bayesian_evc.py`)

The `BayesianEVC` class extends traditional EVC with uncertainty:

**Formula:**
```
Bayesian_EVC = (Reward_Weight × Confidence-Weighted_Expected_Reward) 
               - (Effort_Cost_Weight × Effort_Cost)
               + (Uncertainty_Weight × Uncertainty_Reduction_Benefit)

where:
  Confidence-Weighted_Expected_Reward = P(success) × Reward × Confidence
  Uncertainty_Reduction_Benefit = Uncertainty × Control
```

**Key Innovation:**
The model explicitly values uncertainty reduction. When uncertainty is high, allocating more control provides additional benefit beyond just improving accuracy—it also reduces uncertainty itself.

**Parameters:**
- `reward_weight`: Weight for reward component (default: 1.0)
- `effort_cost_weight`: Weight for effort cost (default: 1.0)
- `uncertainty_weight`: Weight for uncertainty reduction (default: 0.5) **[NEW]**
- `effort_exponent`: Exponent for effort function (default: 2.0)

**Key Methods:**
- `compute_bayesian_evc()`: Calculate Bayesian EVC
- `optimal_control()`: Find control that maximizes Bayesian EVC
- `fit()`: Fit parameters including uncertainty weight
- `predict_control()`: Predict optimal control with uncertainty
- `evaluate()`: Evaluate model performance

### 5. Visualization (`utils/visualization.py`)

The `EVCVisualizer` class provides comprehensive plotting:

**Available Plots:**
1. `plot_model_comparison()`: Compare model predictions vs. observed
2. `plot_uncertainty_effects()`: Relationship between uncertainty and control
3. `plot_block_effects()`: Behavioral metrics across experimental blocks
4. `plot_model_fit_metrics()`: Bar charts comparing model fit (R², RMSE, correlation)
5. `plot_neural_correlates()`: Correlations between behavioral and neural measures
6. `plot_individual_differences()`: Distribution and effects of uncertainty tolerance

### 6. Main Pipeline (`main_pipeline.py`)

The `BayesianEVCPipeline` class orchestrates the complete analysis:

**Pipeline Steps:**
1. **Generate Data**: Create experimental dataset
2. **Split Data**: Separate train/test sets by subject
3. **Fit Traditional EVC**: Train baseline model
4. **Fit Bayesian EVC**: Train uncertainty-aware model
5. **Compare Models**: Statistical comparison of performance
6. **Generate Visualizations**: Create all plots
7. **Save Results**: Export data, parameters, and predictions

## Usage Examples

### Example 1: Generate Custom Data

```python
from utils.data_generator import ExperimentalDataGenerator

generator = ExperimentalDataGenerator(seed=42)

# Generate data
behavioral_data = generator.generate_task_data(
    n_subjects=50,
    n_trials_per_subject=300,
    n_blocks=4
)

neural_data = generator.generate_neural_data(behavioral_data)

# Save
generator.save_data(behavioral_data, neural_data, output_dir='my_data')
```

### Example 2: Fit Models Independently

```python
import pandas as pd
from models.traditional_evc import TraditionalEVC
from models.bayesian_evc import BayesianEVC

# Load data
data = pd.read_csv('data/behavioral_data.csv')

# Traditional EVC
trad_model = TraditionalEVC()
trad_results = trad_model.fit(data)
print(f"Traditional R²: {trad_results['r2']:.3f}")

# Bayesian EVC
bayes_model = BayesianEVC()
bayes_results = bayes_model.fit(
    data,
    uncertainty_col='total_uncertainty',
    confidence_col='confidence'
)
print(f"Bayesian R²: {bayes_results['r2']:.3f}")
```

### Example 3: Analyze Uncertainty Over Time

```python
from models.bayesian_uncertainty import SequentialBayesianEstimator

estimator = SequentialBayesianEstimator(n_states=2, learning_rate=0.1)

# Process all subjects
results = estimator.process_subject_data(
    data,
    subject_col='subject_id',
    evidence_col='evidence_clarity',
    outcome_col='accuracy'
)

# Now results contains trial-by-trial uncertainty estimates
print(results[['trial', 'decision_uncertainty', 'state_uncertainty', 'combined_uncertainty']].head())
```

### Example 4: Custom Visualizations

```python
from utils.visualization import EVCVisualizer

viz = EVCVisualizer(figsize=(12, 8))

# Plot specific relationships
viz.plot_uncertainty_effects(
    data,
    uncertainty_col='total_uncertainty',
    control_col='control_signal',
    save_path='my_uncertainty_plot.png'
)
```

### Example 5: Run Partial Pipeline

```python
from main_pipeline import BayesianEVCPipeline

pipeline = BayesianEVCPipeline(output_dir='my_results', seed=123)

# Run only specific steps
pipeline.step1_generate_data(n_subjects=20, n_trials_per_subject=100)
pipeline.step2_split_data(train_ratio=0.8)
pipeline.step3_fit_traditional_evc()
pipeline.step4_fit_bayesian_evc()
pipeline.step5_compare_models()
```

## Model Comparison Metrics

The pipeline evaluates models using:

1. **R² (Coefficient of Determination)**
   - Proportion of variance explained
   - Higher is better
   - Range: -∞ to 1 (1 = perfect fit)

2. **RMSE (Root Mean Squared Error)**
   - Average prediction error
   - Lower is better
   - Same units as control signal (0-1)

3. **Correlation**
   - Linear relationship between predicted and observed
   - Range: -1 to 1 (1 = perfect positive correlation)

## Expected Results

Based on the proposal's hypotheses, you should observe:

1. **Bayesian EVC shows better predictive performance** than traditional EVC
   - Higher R² on test set
   - Lower RMSE
   - Better captures control allocation patterns

2. **Uncertainty effects on control**
   - Positive relationship between uncertainty and control allocation
   - Individual differences in uncertainty tolerance predict control strategies

3. **Block effects**
   - Accuracy decreases in high-uncertainty blocks
   - Reaction time increases with uncertainty
   - Control allocation increases with uncertainty

4. **Neural correlates**
   - DLPFC activity correlates with control signal
   - ACC activity correlates with uncertainty
   - Striatal activity correlates with reward

## Customization

### Adding New Uncertainty Types

To add a new uncertainty source:

1. Modify `BayesianUncertaintyEstimator` to include new estimation method
2. Update `estimate_combined_uncertainty()` to integrate it
3. Adjust weights in the combination

### Modifying the EVC Formula

To change how EVC is computed:

1. Edit `compute_evc()` in `TraditionalEVC` or `compute_bayesian_evc()` in `BayesianEVC`
2. Update `optimal_control()` if needed
3. Adjust parameter bounds in `fit()` method

### Adding New Visualizations

To create new plots:

1. Add method to `EVCVisualizer` class
2. Call it from `step6_generate_visualizations()` in pipeline
3. Follow existing patterns for consistency

## Troubleshooting

### Issue: Negative R² values

**Cause:** Model predictions are worse than simply predicting the mean.

**Solutions:**
- Increase sample size
- Check if data generation parameters are reasonable
- Adjust model parameter bounds during fitting
- Use more training data

### Issue: Model fitting is slow

**Cause:** Optimization over many trials.

**Solutions:**
- Reduce number of subjects/trials for testing
- Use subset of data for initial exploration
- Consider parallel processing for subject-level fits

### Issue: Visualizations don't display

**Cause:** Running in headless environment or matplotlib backend issues.

**Solutions:**
- Save plots to files instead of showing
- Set matplotlib backend: `matplotlib.use('Agg')`
- Run in environment with display support

## Extensions

Potential extensions to the implementation:

1. **Hierarchical Bayesian Modeling**
   - Use PyMC to implement full hierarchical model
   - Estimate population-level and individual-level parameters
   - Better handle individual differences

2. **Real Data Integration**
   - Replace dummy data with actual experimental data
   - Adjust column names and preprocessing as needed
   - Validate model assumptions

3. **Additional Uncertainty Sources**
   - Temporal uncertainty (timing of events)
   - Social uncertainty (others' intentions)
   - Outcome uncertainty (reward variability)

4. **Neural Network Implementation**
   - Use deep learning to predict control from features
   - Compare with theory-driven models
   - Explore non-linear relationships

5. **Cross-Validation**
   - Implement k-fold cross-validation
   - More robust model comparison
   - Better generalization estimates

## References

This implementation is based on:

- **Expected Value of Control (EVC) Theory**: Shenhav et al. (2013)
- **Drift-Diffusion Models**: Ratcliff & McKoon (2008)
- **Bayesian Inference in Cognition**: Daw et al. (2005)
- **Uncertainty and Control**: Behrens et al. (2007)

## Contact & Support

For questions about the implementation:
1. Check this guide and README.md
2. Review code comments and docstrings
3. Run test_implementation.py to verify setup
4. Examine example outputs in results/

## License

This is a research implementation for educational purposes.

