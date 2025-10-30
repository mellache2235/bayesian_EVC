# Bayesian Expected Value of Control (EVC) Modeling Framework

This repository implements a Bayesian Expected Value of Control (EVC) framework that integrates uncertainty into cognitive control allocation decisions. The framework extends traditional EVC models by explicitly incorporating decision uncertainty and state/rule uncertainty into the control allocation computation.

## Overview

The Bayesian EVC model addresses a key limitation of traditional EVC approaches: the assumption of clear and stable task conditions. By incorporating Bayesian uncertainty estimation, the model better captures how individuals allocate cognitive effort in dynamic and uncertain environments.

### Key Components

1. **Bayesian EVC Model**: Extends traditional EVC by adding uncertainty reduction as an explicit benefit component
2. **Drift Diffusion Model (DDM)**: Estimates decision confidence and uncertainty from behavioral data
3. **MCMC Parameter Estimation**: Bayesian inference for model parameters using PyMC
4. **Model Comparison**: Comparison with traditional EVC models

## Project Structure

```
.
├── generate_dummy_data.py      # Data generation script
├── pipeline.py                  # Main pipeline script
├── requirements.txt             # Python dependencies
├── models/                      # Model implementations
│   ├── __init__.py
│   ├── ddm.py                   # Drift Diffusion Model
│   ├── bayesian_evc.py          # Bayesian EVC model
│   └── traditional_evc.py       # Traditional EVC model (baseline)
├── estimation/                  # Parameter estimation
│   ├── __init__.py
│   └── mcmc_fitting.py          # MCMC parameter estimation
└── data/                        # Data directory (generated)
    ├── behavioral_data.csv
    └── neural_data.csv
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bayesian_EVC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Generate Dummy Data

First, generate synthetic behavioral and neural data:

```bash
python generate_dummy_data.py
```

This creates:
- `data/behavioral_data.csv`: Behavioral trial data with uncertainty measures
- `data/neural_data.csv`: Neural activity data (fNIRS/fMRI-like)

### Run Main Pipeline

Run the complete analysis pipeline:

```bash
python pipeline.py
```

The pipeline performs:
1. Data loading/generation
2. Uncertainty estimation using DDM
3. Model fitting (Bayesian and Traditional EVC)
4. Model comparison
5. Visualization generation

### Individual Components

#### Bayesian EVC Model

```python
from models.bayesian_evc import BayesianEVC

model = BayesianEVC(
    reward_sensitivity=1.0,
    effort_cost=0.5,
    uncertainty_reduction_weight=0.3,
    uncertainty_tolerance=0.5,
    control_efficiency=1.0
)

# Compute EVC
evc, components = model.compute_evc(
    expected_reward,
    control_level,
    decision_uncertainty,
    state_uncertainty
)

# Optimal control allocation
optimal_control = model.optimal_control(
    expected_reward,
    decision_uncertainty,
    state_uncertainty
)
```

#### DDM Confidence Estimation

```python
from models.ddm import DriftDiffusionModel

ddm = DriftDiffusionModel()
confidence, uncertainty = ddm.compute_confidence(
    reaction_times,
    choices,
    evidence_clarity
)
```

#### MCMC Parameter Estimation

```python
from estimation.mcmc_fitting import fit_bayesian_evc_mcmc

model, trace = fit_bayesian_evc_mcmc(
    behavioral_data,
    participant_id=0,
    n_samples=1000
)
```

## Model Formulation

### Bayesian EVC

The Bayesian EVC extends the traditional formulation:

**Traditional EVC:**
```
EVC = β_r × E[Reward|Control] - c × Control²
```

**Bayesian EVC:**
```
EVC = β_r × E[Reward|Control] - c × Control² + λ × UncertaintyReduction
```

Where:
- `β_r`: Reward sensitivity
- `c`: Effort cost parameter
- `λ`: Uncertainty reduction weight
- `UncertaintyReduction = η × Control × TotalUncertainty`
- `TotalUncertainty = DecisionUncertainty + StateUncertainty`

### Uncertainty Types

1. **Decision Uncertainty**: Trial-to-trial variability in evidence clarity, estimated from DDM
2. **State Uncertainty**: Uncertainty about task rules/conditions, estimated from rule stability

## Data Format

### Behavioral Data

| Column | Description |
|--------|-------------|
| `participant_id` | Participant identifier |
| `trial` | Trial number |
| `evidence_clarity` | Evidence clarity (0-1, higher = clearer) |
| `rule_stability` | Rule stability (0-1, higher = more stable) |
| `decision_uncertainty` | Decision uncertainty (computed) |
| `state_uncertainty` | State/rule uncertainty (computed) |
| `reaction_time` | Reaction time in seconds |
| `choice` | Choice made (0 or 1) |
| `correct` | Correctness (0 or 1) |
| `reward` | Reward received |
| `difficulty` | Trial difficulty |

### Neural Data

| Column | Description |
|--------|-------------|
| `participant_id` | Participant identifier |
| `trial` | Trial number |
| `control_signal` | Aggregated control signal |
| `channel_0` ... `channel_N` | Neural activity per channel |

## Model Comparison

The framework includes comparison metrics:
- R² (coefficient of determination)
- Correlation with observed behavior
- Mean squared error (MSE)
- Improvement over traditional EVC

## Output

Running the pipeline generates:
- `results/model_results.png`: Visualization of model fits and comparisons
- Model parameter estimates
- Comparison metrics

## Future Directions

- Hierarchical modeling for group-level parameters
- Neural data integration (fNIRS/fMRI)
- Model selection and comparison (WAIC, LOO)
- Individual differences analysis
- Developmental trajectory modeling

## Citation

If you use this framework, please cite:

```
Bayesian Expected Value of Control Framework
[Your citation information]
```

## License

[Specify license]

## Contact

[Your contact information]

