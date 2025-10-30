#!/usr/bin/env python3
"""CLI for generating structured Bayesian EVC synthetic datasets.

Usage examples::

    python scripts/generate_data.py
    python scripts/generate_data.py --children 60 --trials 120 --seed 42 \
        --output data/custom_trials.csv

The script wraps ``src.data_generation.StructuredDatasetGenerator`` and exposes
the key configuration parameters via command-line flags so you can quickly
produce datasets with different sample sizes or random seeds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_repo_root_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_add_repo_root_to_path()

from src.data_generation import GenerationConfig, StructuredDatasetGenerator  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate structured Bayesian EVC synthetic datasets.",
    )
    parser.add_argument(
        "--children",
        type=int,
        default=40,
        help="Number of simulated participants (default: 40).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=80,
        help="Number of trials per participant (default: 80).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/structured_evc_trials.csv"),
        help="Output CSV path (default: data/structured_evc_trials.csv).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = GenerationConfig(
        n_children=args.children,
        trials_per_child=args.trials,
        seed=args.seed,
        output_path=args.output,
    )
    generator = StructuredDatasetGenerator(config)
    df = generator.generate()
    path = generator.save(df)

    print(
        "Generated structured dataset",  # noqa: T201
        f"for {config.n_children} children × {config.trials_per_child} trials",
        f"at {path}",
    )


if __name__ == "__main__":
    main()

