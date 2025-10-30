# Run Steps - Bayesian EVC Pipeline

## Quick Start

Run the complete pipeline:
```bash
python main_pipeline.py
```

## Step-by-Step Execution

### Step 1: Generate Data
```bash
python step1_generate_data.py
```
Generates experimental behavioral and neural data.

### Step 2: Estimate Uncertainty
```bash
python step2_estimate_uncertainty.py
```
Computes Bayesian uncertainty estimates from the data.

### Step 3: Fit Traditional EVC Model
```bash
python step3_fit_traditional_evc.py
```
Fits the traditional Expected Value of Control model.

### Step 4: Fit Bayesian EVC Model
```bash
python step4_fit_bayesian_evc.py
```
Fits the Bayesian EVC model with uncertainty integration.

### Step 5: Compare Models
```bash
python step5_compare_models.py
```
Compares performance between traditional and Bayesian models.

### Step 6: Visualize Results
```bash
python step6_visualize.py
```
Generates comprehensive visualizations of results.

## Output

- `data/` - Generated behavioral and neural data
- `results/` - Model comparison metrics and predictions
- `results/figures/` - Visualization plots

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

