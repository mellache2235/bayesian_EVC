# Bayesian Expected Value of Control (EVC) Framework

Bayesian EVC model incorporating uncertainty into cognitive control allocation.

## Quick Start

```bash
pip install -r requirements.txt
python pipeline.py
```

## Components

- **Bayesian EVC**: EVC with uncertainty reduction benefit
- **Traditional EVC**: Baseline model without uncertainty
- **DDM**: Confidence estimation from behavioral data
- **MCMC**: Parameter estimation (optional)

## Structure

```
├── pipeline.py              # Main pipeline
├── generate_dummy_data.py   # Data generation
├── models/                  # EVC and DDM models
└── estimation/             # MCMC fitting
```

## Usage

```python
from models.bayesian_evc import BayesianEVC

model = BayesianEVC()
control = model.optimal_control(reward, decision_unc, state_unc)
evc, components = model.compute_evc(reward, control, decision_unc, state_unc)
```
