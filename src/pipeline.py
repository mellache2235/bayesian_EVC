"""Starter implementation of a Bayesian Expected Value of Control pipeline.

This module wires together the core steps outlined in ``proposal.md``:

1. Load task-level observations that separately manipulate evidence clarity
   (decision uncertainty) and rule stability (state uncertainty).
2. Run light-weight Bayesian belief updates to maintain per-participant
   estimates of uncertainty about task rules and evidence reliability.
3. Combine those estimates with reward and effort-cost information to compute
   an Expected Value of Control (EVC) signal that explicitly prices
   uncertainty reduction benefits.

The code is intentionally scaffolded to make it easy to extend with richer
latent-state models (e.g., full hierarchical Bayesian inference or MCMC)
while remaining executable with the dummy data included in ``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BeliefPrior:
    """Hyperparameters for Bayesian belief tracking.

    Attributes
    ----------
    rule_alpha : float
        Prior pseudo-count for the hypothesis that the current rule mapping
        still holds. Higher values encode stronger initial confidence.
    rule_beta : float
        Prior pseudo-count for the hypothesis that the rule has changed.
    evidence_strength : float
        Scale factor applied to evidence clarity scores when updating
        evidence-confidence beliefs.
    volatility_discount : float
        Discount factor (0-1) that gradually forgets past beliefs to model
        volatile environments. Values closer to 0 emphasise recent evidence.
    """

    rule_alpha: float = 2.0
    rule_beta: float = 2.0
    evidence_strength: float = 2.5
    volatility_discount: float = 0.85


@dataclass
class EVCWeights:
    """Linear weights mapping model signals to an EVC score."""

    reward: float = 1.0
    effort_cost: float = 1.0
    uncertainty_reduction: float = 0.8


@dataclass
class ModelConfig:
    """Container for all model hyperparameters."""

    belief_prior: BeliefPrior = BeliefPrior()
    evc_weights: EVCWeights = EVCWeights()
    control_temperature: float = 1.5
    seed: Optional[int] = 123


# ---------------------------------------------------------------------------
# Core model components
# ---------------------------------------------------------------------------


@dataclass
class BeliefState:
    """Sufficient statistics representing the agent's current beliefs."""

    rule_alpha: float
    rule_beta: float
    evidence_precision: float

    def rule_confidence(self) -> float:
        total = self.rule_alpha + self.rule_beta
        return 0.0 if total == 0 else self.rule_alpha / total

    def rule_uncertainty(self) -> float:
        # Normalised Beta entropy proxy (0=no uncertainty, 1=max uncertainty)
        p = self.rule_confidence()
        return 4 * p * (1 - p)


class BayesianUncertaintyTracker:
    """Tracks rule and evidence uncertainty across trials."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def initialise_state(self) -> BeliefState:
        prior = self.config.belief_prior
        return BeliefState(
            rule_alpha=prior.rule_alpha,
            rule_beta=prior.rule_beta,
            evidence_precision=prior.evidence_strength,
        )

    def update(self, state: BeliefState, trial: pd.Series) -> BeliefState:
        prior = self.config.belief_prior
        # Discount past beliefs to model volatility before adding new evidence
        decay = prior.volatility_discount
        state.rule_alpha *= decay
        state.rule_beta *= decay
        state.evidence_precision *= decay

        stability_evidence = trial["rule_stability"] * prior.evidence_strength
        clarity_evidence = trial["evidence_clarity"] * prior.evidence_strength

        state.rule_alpha += stability_evidence
        state.rule_beta += max(0.0, prior.evidence_strength - stability_evidence)

        state.evidence_precision += clarity_evidence
        return state

    def expected_uncertainty_reduction(
        self, state: BeliefState, trial: pd.Series
    ) -> float:
        # Higher clarity and stability produce larger reductions.
        rule_uncertainty = state.rule_uncertainty()
        evidence_uncertainty = 1 / (1 + state.evidence_precision)
        signal = rule_uncertainty * trial["rule_stability"] + evidence_uncertainty * trial[
            "evidence_clarity"
        ]
        return float(signal / 2)


class BayesianEVCModel:
    """Computes control allocation decisions from uncertainty-aware signals."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.tracker = BayesianUncertaintyTracker(config)

    def initialise_state(self) -> BeliefState:
        return self.tracker.initialise_state()

    def evaluate_trial(self, state: BeliefState, trial: pd.Series) -> Dict[str, float]:
        state = self.tracker.update(state, trial)

        reward = trial["reward"]
        effort_cost = trial["effort_cost"]
        reduction_signal = self.tracker.expected_uncertainty_reduction(state, trial)

        weights = self.config.evc_weights
        evc_score = (
            weights.reward * reward
            - weights.effort_cost * effort_cost
            + weights.uncertainty_reduction * reduction_signal
        )

        control_allocation = self._squashed_policy(evc_score)
        predicted_accuracy = self._predict_accuracy(state, control_allocation)

        return {
            "posterior_rule_confidence": state.rule_confidence(),
            "posterior_rule_uncertainty": state.rule_uncertainty(),
            "posterior_evidence_precision": state.evidence_precision,
            "expected_uncertainty_reduction": reduction_signal,
            "evc_score": evc_score,
            "predicted_control_allocation": control_allocation,
            "predicted_accuracy": predicted_accuracy,
        }

    def _squashed_policy(self, evc_score: float) -> float:
        temperature = self.config.control_temperature
        return 1 / (1 + np.exp(-evc_score / max(1e-3, temperature)))

    def _predict_accuracy(self, state: BeliefState, control: float) -> float:
        base = state.rule_confidence()
        precision = state.evidence_precision / (state.evidence_precision + 1)
        return float(np.clip(base * 0.4 + precision * 0.3 + control * 0.3, 0, 1))


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


class EVCPipeline:
    """End-to-end orchestration for running the Bayesian EVC model."""

    def __init__(self, data_path: str, config: Optional[ModelConfig] = None):
        self.data_path = data_path
        self.config = config or ModelConfig()
        self.model = BayesianEVCModel(self.config)
        self.results_: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        expected_cols = {
            "child_id",
            "trial_id",
            "evidence_clarity",
            "rule_stability",
            "reward",
            "effort_cost",
        }
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
        return df

    def run(self) -> pd.DataFrame:
        df = self.load_data()
        records: List[Dict[str, float]] = []

        for child_id, child_trials in df.groupby("child_id"):
            state = self.model.initialise_state()
            child_trials = child_trials.sort_values("trial_id")
            for _, trial in child_trials.iterrows():
                estimates = self.model.evaluate_trial(state, trial)
                output = {**trial.to_dict(), **estimates, "child_id": child_id}
                records.append(output)

        self.results_ = pd.DataFrame(records)
        return self.results_

    def summarise(self) -> pd.DataFrame:
        if self.results_ is None:
            raise RuntimeError("Pipeline must be run before requesting a summary.")

        summary = self.results_.groupby("child_id").agg(
            mean_control=("predicted_control_allocation", "mean"),
            mean_uncertainty=("posterior_rule_uncertainty", "mean"),
            mean_accuracy=("predicted_accuracy", "mean"),
            mean_reward=("reward", "mean"),
        )
        return summary.reset_index()


# ---------------------------------------------------------------------------
# Utility entry point
# ---------------------------------------------------------------------------


def main(data_path: str = "data/structured_evc_trials.csv") -> None:
    """Convenience entry point for running the pipeline from the CLI."""

    pipeline = EVCPipeline(data_path=data_path)
    results = pipeline.run()
    summary = pipeline.summarise()

    print("=== Example Results (first 5 rows) ===")
    print(results.head())
    print("\n=== Per-child Summary ===")
    print(summary.head())


if __name__ == "__main__":
    main()

