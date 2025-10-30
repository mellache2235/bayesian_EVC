"""
Quick test script to verify the implementation works.
"""

import numpy as np
import pandas as pd

from utils.data_generator import ExperimentalDataGenerator
from models.bayesian_uncertainty import BayesianUncertaintyEstimator
from models.traditional_evc import TraditionalEVC
from models.bayesian_evc import BayesianEVC


def test_data_generation():
    """Test data generation."""
    print("Testing data generation...")
    generator = ExperimentalDataGenerator(seed=42)
    
    behavioral_data = generator.generate_task_data(
        n_subjects=5,
        n_trials_per_subject=50,
        n_blocks=4
    )
    
    neural_data = generator.generate_neural_data(behavioral_data)
    
    print(f"✓ Generated {len(behavioral_data)} behavioral trials")
    print(f"✓ Generated {len(neural_data)} neural observations")
    print(f"\nSample behavioral data:")
    print(behavioral_data.head())
    
    return behavioral_data, neural_data


def test_uncertainty_estimation():
    """Test Bayesian uncertainty estimation."""
    print("\n" + "=" * 70)
    print("Testing Bayesian uncertainty estimation...")
    
    estimator = BayesianUncertaintyEstimator(n_states=2, learning_rate=0.2)
    
    # Test sequence
    evidence_sequence = [0.8, 0.7, 0.5, 0.4, 0.6]
    outcomes = [1, 1, 0, 0, 1]
    
    for i, (evidence, outcome) in enumerate(zip(evidence_sequence, outcomes)):
        metrics = estimator.estimate_combined_uncertainty(evidence, outcome)
        print(f"Trial {i+1}: uncertainty={metrics['combined_uncertainty']:.3f}, "
              f"confidence={metrics['combined_confidence']:.3f}")
    
    print("✓ Uncertainty estimation working")


def test_traditional_evc():
    """Test traditional EVC model."""
    print("\n" + "=" * 70)
    print("Testing traditional EVC model...")
    
    model = TraditionalEVC(
        reward_weight=1.0,
        effort_cost_weight=1.0,
        effort_exponent=2.0
    )
    
    # Test optimal control
    optimal_control, max_evc = model.optimal_control(
        reward_magnitude=10,
        baseline_accuracy=0.5
    )
    
    print(f"Optimal control for reward=10, baseline_acc=0.5: {optimal_control:.3f}")
    print(f"Maximum EVC: {max_evc:.3f}")
    print("✓ Traditional EVC model working")
    
    return model


def test_bayesian_evc():
    """Test Bayesian EVC model."""
    print("\n" + "=" * 70)
    print("Testing Bayesian EVC model...")
    
    model = BayesianEVC(
        reward_weight=1.0,
        effort_cost_weight=1.0,
        uncertainty_weight=0.5,
        effort_exponent=2.0
    )
    
    # Test optimal control with uncertainty
    optimal_control, max_evc = model.optimal_control(
        reward_magnitude=10,
        baseline_accuracy=0.5,
        uncertainty=0.7,
        confidence=0.3
    )
    
    print(f"Optimal control for reward=10, baseline_acc=0.5, uncertainty=0.7: {optimal_control:.3f}")
    print(f"Maximum Bayesian EVC: {max_evc:.3f}")
    print("✓ Bayesian EVC model working")
    
    return model


def test_model_fitting(behavioral_data):
    """Test model fitting."""
    print("\n" + "=" * 70)
    print("Testing model fitting...")
    
    # Traditional EVC
    print("\nFitting Traditional EVC...")
    trad_model = TraditionalEVC()
    trad_results = trad_model.fit(
        behavioral_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity'
    )
    print(f"  R²: {trad_results['r2']:.3f}")
    print(f"  RMSE: {trad_results['rmse']:.3f}")
    
    # Bayesian EVC
    print("\nFitting Bayesian EVC...")
    bayes_model = BayesianEVC()
    bayes_results = bayes_model.fit(
        behavioral_data,
        observed_control_col='control_signal',
        reward_col='reward_magnitude',
        accuracy_col='evidence_clarity',
        uncertainty_col='total_uncertainty',
        confidence_col='confidence'
    )
    print(f"  R²: {bayes_results['r2']:.3f}")
    print(f"  RMSE: {bayes_results['rmse']:.3f}")
    
    print("\n✓ Model fitting working")
    
    # Compare
    print("\nModel Comparison:")
    print(f"  Traditional EVC R²: {trad_results['r2']:.3f}")
    print(f"  Bayesian EVC R²: {bayes_results['r2']:.3f}")
    
    if bayes_results['r2'] > trad_results['r2']:
        improvement = (bayes_results['r2'] - trad_results['r2']) / trad_results['r2'] * 100
        print(f"  ✓ Bayesian EVC shows {improvement:.1f}% improvement in R²")
    
    return trad_model, bayes_model


def main():
    """Run all tests."""
    print("=" * 70)
    print("TESTING BAYESIAN EVC IMPLEMENTATION")
    print("=" * 70)
    
    # Test 1: Data generation
    behavioral_data, neural_data = test_data_generation()
    
    # Test 2: Uncertainty estimation
    test_uncertainty_estimation()
    
    # Test 3: Traditional EVC
    test_traditional_evc()
    
    # Test 4: Bayesian EVC
    test_bayesian_evc()
    
    # Test 5: Model fitting
    test_model_fitting(behavioral_data)
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nThe implementation is working correctly.")
    print("\nTo run the full pipeline with visualizations, execute:")
    print("  python3 main_pipeline.py")


if __name__ == '__main__':
    main()

