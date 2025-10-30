"""Structured synthetic data generator for the Bayesian EVC pipeline.

The generator aims to produce behavioural traces that mimic qualitative
patterns reported in executive-function tasks that manipulate evidence clarity
and rule volatility (e.g., Stroop/Flanker paradigms, task-switching designs).

Key ingredients inspired by typical findings:

* Reaction times concentrate between 550–1,100 ms, with harder/volatile trials
  yielding longer latencies (see, e.g., DDM fits summarised in Wiecki et al.,
  2013; Ueltzhöffer et al., 2015).
* Accuracy improves with higher evidence clarity and sustained rule stability,
  but can be rescued by exerting additional cognitive control.
* Individual differences are expressed via baseline drift rates (ability),
  response caution (threshold), and sensitivity to rewards vs. uncertainty.

The produced dataset stays small enough to run quickly, yet embeds structure
that the Bayesian EVC pipeline can leverage when comparing to more traditional
reward–effort models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _logistic(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


@dataclass
class ChildProfile:
    """Latent characteristics for one participant."""

    child_id: int
    ability: float  # baseline drift-rate multiplier
    caution: float  # decision threshold (higher => slower, more accurate)
    reward_sensitivity: float
    effort_sensitivity: float
    uncertainty_intolerance: float
    control_efficiency: float
    non_decision: float


@dataclass
class GenerationConfig:
    n_children: int = 40
    trials_per_child: int = 80
    seed: int = 123
    output_path: Path = Path("data/structured_evc_trials.csv")


class StructuredDatasetGenerator:
    """Creates structured synthetic datasets for downstream modelling."""

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        profiles = [self._sample_child_profile(idx + 1) for idx in range(self.config.n_children)]
        records: List[dict] = []

        volatility_conditions = ["stable", "volatile"]
        clarity_levels = ["high", "medium", "low"]

        for profile in profiles:
            block_offsets = self._rng.uniform(0, 1, size=3)
            for trial_id in range(1, self.config.trials_per_child + 1):
                vol_condition = self._rng.choice(volatility_conditions, p=[0.55, 0.45])
                clarity_condition = self._rng.choice(clarity_levels, p=[0.4, 0.35, 0.25])

                trial = self._simulate_trial(profile, trial_id, vol_condition, clarity_condition, block_offsets)
                records.append(trial)

        df = pd.DataFrame.from_records(records)
        return df

    def save(self, df: pd.DataFrame) -> Path:
        path = self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_child_profile(self, child_id: int) -> ChildProfile:
        rng = self._rng
        ability = rng.normal(0.85, 0.12)
        caution = rng.normal(1.05, 0.08)
        reward_sens = rng.normal(1.2, 0.2)
        effort_sens = rng.normal(0.9, 0.15)
        uncertainty_intolerance = rng.normal(1.1, 0.2)
        control_efficiency = rng.normal(0.35, 0.05)
        non_decision = rng.normal(0.32, 0.03)
        return ChildProfile(
            child_id=child_id,
            ability=max(0.4, ability),
            caution=max(0.85, caution),
            reward_sensitivity=max(0.6, reward_sens),
            effort_sensitivity=max(0.4, effort_sens),
            uncertainty_intolerance=max(0.5, uncertainty_intolerance),
            control_efficiency=max(0.2, control_efficiency),
            non_decision=max(0.2, non_decision),
        )

    def _simulate_trial(
        self,
        profile: ChildProfile,
        trial_id: int,
        volatility_condition: str,
        clarity_condition: str,
        block_offsets: np.ndarray,
    ) -> dict:
        rng = self._rng

        clarity_lookup = {
            "high": rng.normal(0.82, 0.06),
            "medium": rng.normal(0.58, 0.07),
            "low": rng.normal(0.34, 0.08),
        }
        stability_lookup = {
            "stable": rng.normal(0.78, 0.08),
            "volatile": rng.normal(0.38, 0.1),
        }

        evidence_clarity = float(np.clip(clarity_lookup[clarity_condition], 0.05, 0.95))
        rule_stability = float(np.clip(stability_lookup[volatility_condition], 0.05, 0.95))

        # Reward schedules: higher rewards in volatile/high effort blocks
        reward_mean = 1.4 if volatility_condition == "volatile" else 1.1
        reward = float(np.clip(rng.normal(reward_mean, 0.25), 0.4, 2.2))
        effort_cost = float(np.clip(rng.normal(0.65 + (1 - evidence_clarity) * 0.4, 0.18), 0.2, 1.5))

        baseline_uncertainty = 1 - (0.55 * evidence_clarity + 0.45 * rule_stability)
        baseline_uncertainty = float(np.clip(baseline_uncertainty, 0.05, 0.95))

        control_drive = (
            profile.reward_sensitivity * (reward - 0.6)
            - profile.effort_sensitivity * effort_cost
            + profile.uncertainty_intolerance * (baseline_uncertainty - rule_stability + 0.2)
            + block_offsets[0] * 0.15
        )
        control_latent = control_drive + rng.normal(0, 0.25)
        control_allocation = float(np.clip(_logistic(control_latent), 0.01, 0.99))

        drift_base = profile.ability * (0.35 + 0.6 * evidence_clarity)
        volatility_penalty = (1 - rule_stability) * 0.25
        drift_rate = float(np.clip(drift_base + profile.control_efficiency * control_allocation - volatility_penalty, 0.08, 2.0))

        threshold = profile.caution + block_offsets[1] * 0.05
        non_decision = profile.non_decision + block_offsets[2] * 0.02

        rt_mean = non_decision + threshold / drift_rate
        rt_noise = 0.07 + (1 - evidence_clarity) * 0.12 + (1 - rule_stability) * 0.05
        observed_rt = float(np.clip(rng.normal(rt_mean, rt_noise), 0.35, 1.6))

        accuracy_prob = _logistic((drift_rate - 0.45) * 3 - (threshold - 1.0))
        observed_accuracy = int(rng.random() < accuracy_prob)

        return {
            "child_id": profile.child_id,
            "trial_id": trial_id,
            "volatility_condition": volatility_condition,
            "clarity_condition": clarity_condition,
            "evidence_clarity": round(evidence_clarity, 3),
            "rule_stability": round(rule_stability, 3),
            "reward": round(reward, 3),
            "effort_cost": round(effort_cost, 3),
            "baseline_uncertainty": round(baseline_uncertainty, 3),
            "observed_control": round(control_allocation, 3),
            "observed_drift": round(drift_rate, 3),
            "observed_rt": round(observed_rt, 3),
            "observed_accuracy": observed_accuracy,
        }


def main() -> None:
    generator = StructuredDatasetGenerator()
    df = generator.generate()
    path = generator.save(df)
    print(f"Wrote structured dataset with {len(df)} rows to {path}")


if __name__ == "__main__":
    main()

