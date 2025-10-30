# Structured Synthetic Dataset Overview

## Motivation

The original dummy dataset relied on independent uniform draws, which made it
hard to see theoretically meaningful patterns. The new generator encodes
behavioural structure inspired by canonical executive-function tasks that
manipulate evidence quality and rule volatility. It enables quick prototyping of
the uncertainty-aware Expected Value of Control (EVC) pipeline while preserving
qualitative effects reported in the literature (e.g., Wiecki et al., 2013; Ueltzhöffer et al., 2015).

## Key Design Choices

- **Condition factors** – Each trial belongs to a volatility block (`stable` vs.
  `volatile`) and an evidence clarity tier (`high`, `medium`, `low`). These
  combinations reproduce predictable changes in uncertainty, reward, and costs.
- **Participant heterogeneity** – Every child has a latent profile with
  parameters for baseline evidence accumulation ability, decision caution,
  reward/effort sensitivities, and how efficiently additional control increases
  drift rate. This creates realistic between-subject differences.
- **Control dynamics** – Observed control allocation reflects a logistic
  combination of reward incentives, effort costs, and uncertainty reduction
  motives. Increased volatility elevates both rewards and control investment.
- **Drift-diffusion inspired outcomes** – Reaction times and accuracies derive
  from approximate drift–diffusion relationships with a non-decision time and
  threshold component. Harder and more volatile trials slow responses and reduce
  accuracy unless control is increased.

## Columns Added Compared to the Original Dummy Data

- `volatility_condition`, `clarity_condition` – categorical labels for trial
  manipulations.
- `observed_control` – simulated control allocation (0–1) used to modulate
  drift rates.
- `observed_drift` – latent drift rate after combining clarity, volatility, and
  control effects.

All original columns consumed by `EVCPipeline` are preserved (`evidence_clarity`,
`rule_stability`, `reward`, `effort_cost`, `baseline_uncertainty`,
`observed_rt`, `observed_accuracy`).

## Typical Ranges

- Evidence clarity ~0.3–0.9 with mean ≈0.60; rule stability ~0.2–0.9.
- Rewards progress from ≈1.0 in stable blocks to ≈1.6 in volatile blocks, while
  effort costs rise as clarity drops.
- Reaction times cluster between 0.55–1.35 s (clipped at 0.35–1.6 s).
- Accuracy averages around 0.68 with expected drops in low-clarity, volatile
  conditions.

## Using the Generator

- Run `python3 scripts/generate_data.py` to create a dataset with the default
  configuration.
- Add flags such as `--children 60 --trials 120 --seed 42 --output data/custom.csv`
  to customise the size, randomness, or output path.
- For programmatic control, import `StructuredDatasetGenerator` directly from
  `src.data_generation`.

