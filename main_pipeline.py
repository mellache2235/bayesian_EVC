"""
Main pipeline for Bayesian EVC analysis.

This script:
1. Generates dummy experimental data
2. Fits both traditional and Bayesian EVC models
3. Compares model performance
4. Generates visualizations
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from utils.data_generator import ExperimentalDataGenerator
from utils.visualization import EVCVisualizer
from models.traditional_evc import TraditionalEVC
from models.bayesian_evc import BayesianEVC


class BayesianEVCPipeline:
    """Main analysis pipeline for Bayesian EVC project."""
    
    def __init__(self, output_dir: str = 'results', seed: int = 42):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Directory for saving results
            seed: Random seed for reproducibility
        """
        self.output_dir = output_dir
        self.seed = seed
        self.visualizer = EVCVisualizer()
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/figures', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        # Data containers
        self.behavioral_data = None
        self.neural_data = None
        self.train_data = None
        self.test_data = None
        
        # Models
        self.traditional_model = None
        self.bayesian_model = None
        
        # Results
        self.results = {}
    
    def step1_generate_data(
        self,
        n_subjects: int = 30,
        n_trials_per_subject: int = 200,
        n_blocks: int = 4
    ):
        """
        Step 1: Generate experimental data.
        
        Args:
            n_subjects: Number of participants
            n_trials_per_subject: Trials per participant
            n_blocks: Number of experimental blocks
        """
        print("=" * 70)
        print("STEP 1: GENERATING EXPERIMENTAL DATA")
        print("=" * 70)
        
        generator = ExperimentalDataGenerator(seed=self.seed)
        
        print(f"\nGenerating data for {n_subjects} subjects...")
        print(f"  - {n_trials_per_subject} trials per subject")
        print(f"  - {n_blocks} experimental blocks")
        
        self.behavioral_data = generator.generate_task_data(
            n_subjects=n_subjects,
            n_trials_per_subject=n_trials_per_subject,
            n_blocks=n_blocks
        )
        
        print(f"\nGenerating neural data...")
        self.neural_data = generator.generate_neural_data(self.behavioral_data)
        
        print(f"\nSaving data...")
        generator.save_data(self.behavioral_data, self.neural_data, output_dir='data')
        
        print(f"\n✓ Data generation complete!")
        print(f"  Total trials: {len(self.behavioral_data)}")
        print(f"\nData summary by block:")
        print(self.behavioral_data.groupby('block').agg({
            'total_uncertainty': 'mean',
            'control_signal': 'mean',
            'accuracy': 'mean',
            'reaction_time': 'mean'
        }).round(3))
    
    def step2_split_data(self, train_ratio: float = 0.7):
        """
        Step 2: Split data into training and test sets.
        
        Args:
            train_ratio: Proportion of data for training
        """
        print("\n" + "=" * 70)
        print("STEP 2: SPLITTING DATA")
        print("=" * 70)
        
        # Split by subject to avoid data leakage
        subjects = self.behavioral_data['subject_id'].unique()
        np.random.seed(self.seed)
        np.random.shuffle(subjects)
        
        n_train = int(len(subjects) * train_ratio)
        train_subjects = subjects[:n_train]
        test_subjects = subjects[n_train:]
        
        self.train_data = self.behavioral_data[
            self.behavioral_data['subject_id'].isin(train_subjects)
        ].copy()
        
        self.test_data = self.behavioral_data[
            self.behavioral_data['subject_id'].isin(test_subjects)
        ].copy()
        
        print(f"\nTraining set:")
        print(f"  - {len(train_subjects)} subjects")
        print(f"  - {len(self.train_data)} trials")
        
        print(f"\nTest set:")
        print(f"  - {len(test_subjects)} subjects")
        print(f"  - {len(self.test_data)} trials")
        
        print(f"\n✓ Data split complete!")
    
    def step3_fit_traditional_evc(self):
        """Step 3: Fit traditional EVC model."""
        print("\n" + "=" * 70)
        print("STEP 3: FITTING TRADITIONAL EVC MODEL")
        print("=" * 70)
        
        print("\nInitializing traditional EVC model...")
        self.traditional_model = TraditionalEVC(
            reward_weight=1.0,
            effort_cost_weight=1.0,
            effort_exponent=2.0
        )
        
        print("Fitting model to training data...")
        train_results = self.traditional_model.fit(
            self.train_data,
            observed_control_col='control_signal',
            reward_col='reward_magnitude',
            accuracy_col='evidence_clarity'
        )
        
        print("\nFitted parameters:")
        print(f"  - Reward weight: {train_results['reward_weight']:.3f}")
        print(f"  - Effort cost weight: {train_results['effort_cost_weight']:.3f}")
        print(f"  - Effort exponent: {train_results['effort_exponent']:.3f}")
        
        print("\nTraining set performance:")
        print(f"  - R²: {train_results['r2']:.3f}")
        print(f"  - RMSE: {train_results['rmse']:.3f}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = self.traditional_model.evaluate(
            self.test_data,
            observed_control_col='control_signal',
            reward_col='reward_magnitude',
            accuracy_col='evidence_clarity'
        )
        
        print("\nTest set performance:")
        print(f"  - R²: {test_results['r2']:.3f}")
        print(f"  - RMSE: {test_results['rmse']:.3f}")
        print(f"  - Correlation: {test_results['correlation']:.3f}")
        
        self.results['traditional_evc'] = {
            'train': train_results,
            'test': test_results
        }
        
        print(f"\n✓ Traditional EVC model fitting complete!")
    
    def step4_fit_bayesian_evc(self):
        """Step 4: Fit Bayesian EVC model."""
        print("\n" + "=" * 70)
        print("STEP 4: FITTING BAYESIAN EVC MODEL")
        print("=" * 70)
        
        print("\nInitializing Bayesian EVC model...")
        self.bayesian_model = BayesianEVC(
            reward_weight=1.0,
            effort_cost_weight=1.0,
            uncertainty_weight=0.5,
            effort_exponent=2.0,
            n_states=2,
            learning_rate=0.1
        )
        
        print("Fitting model to training data...")
        train_results = self.bayesian_model.fit(
            self.train_data,
            observed_control_col='control_signal',
            reward_col='reward_magnitude',
            accuracy_col='evidence_clarity',
            uncertainty_col='total_uncertainty',
            confidence_col='confidence'
        )
        
        print("\nFitted parameters:")
        print(f"  - Reward weight: {train_results['reward_weight']:.3f}")
        print(f"  - Effort cost weight: {train_results['effort_cost_weight']:.3f}")
        print(f"  - Uncertainty weight: {train_results['uncertainty_weight']:.3f}")
        print(f"  - Effort exponent: {train_results['effort_exponent']:.3f}")
        
        print("\nTraining set performance:")
        print(f"  - R²: {train_results['r2']:.3f}")
        print(f"  - RMSE: {train_results['rmse']:.3f}")
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        test_results = self.bayesian_model.evaluate(
            self.test_data,
            observed_control_col='control_signal',
            reward_col='reward_magnitude',
            accuracy_col='evidence_clarity',
            uncertainty_col='total_uncertainty',
            confidence_col='confidence'
        )
        
        print("\nTest set performance:")
        print(f"  - R²: {test_results['r2']:.3f}")
        print(f"  - RMSE: {test_results['rmse']:.3f}")
        print(f"  - Correlation: {test_results['correlation']:.3f}")
        
        self.results['bayesian_evc'] = {
            'train': train_results,
            'test': test_results
        }
        
        print(f"\n✓ Bayesian EVC model fitting complete!")
    
    def step5_compare_models(self):
        """Step 5: Compare model performance."""
        print("\n" + "=" * 70)
        print("STEP 5: MODEL COMPARISON")
        print("=" * 70)
        
        trad_test = self.results['traditional_evc']['test']
        bayes_test = self.results['bayesian_evc']['test']
        
        print("\nTest Set Performance Comparison:")
        print("-" * 50)
        print(f"{'Metric':<20} {'Traditional':<15} {'Bayesian':<15} {'Δ':<10}")
        print("-" * 50)
        
        metrics = ['r2', 'rmse', 'correlation']
        for metric in metrics:
            trad_val = trad_test[metric]
            bayes_val = bayes_test[metric]
            delta = bayes_val - trad_val
            
            # For RMSE, lower is better
            if metric == 'rmse':
                delta = -delta
            
            improvement = "✓" if delta > 0 else "✗"
            
            print(f"{metric.upper():<20} {trad_val:<15.4f} {bayes_val:<15.4f} "
                  f"{delta:+.4f} {improvement}")
        
        print("-" * 50)
        
        # Statistical comparison
        print("\nModel Improvement Analysis:")
        r2_improvement = (bayes_test['r2'] - trad_test['r2']) / trad_test['r2'] * 100
        rmse_improvement = (trad_test['rmse'] - bayes_test['rmse']) / trad_test['rmse'] * 100
        
        print(f"  - R² improvement: {r2_improvement:+.2f}%")
        print(f"  - RMSE improvement: {rmse_improvement:+.2f}%")
        
        if bayes_test['r2'] > trad_test['r2']:
            print("\n✓ Bayesian EVC shows superior predictive performance!")
        else:
            print("\n✗ Traditional EVC shows better performance.")
        
        # Save comparison results
        comparison_df = pd.DataFrame({
            'Model': ['Traditional EVC', 'Bayesian EVC'],
            'R²': [trad_test['r2'], bayes_test['r2']],
            'RMSE': [trad_test['rmse'], bayes_test['rmse']],
            'Correlation': [trad_test['correlation'], bayes_test['correlation']]
        })
        
        comparison_df.to_csv(f'{self.output_dir}/model_comparison.csv', index=False)
        print(f"\n✓ Comparison results saved to {self.output_dir}/model_comparison.csv")
    
    def step6_generate_visualizations(self):
        """Step 6: Generate visualizations."""
        print("\n" + "=" * 70)
        print("STEP 6: GENERATING VISUALIZATIONS")
        print("=" * 70)
        
        # Get predictions
        observed = self.test_data['control_signal'].values
        
        traditional_pred = self.traditional_model.predict_control(
            self.test_data,
            reward_col='reward_magnitude',
            accuracy_col='evidence_clarity'
        )
        
        bayesian_pred = self.bayesian_model.predict_control(
            self.test_data,
            reward_col='reward_magnitude',
            accuracy_col='evidence_clarity',
            uncertainty_col='total_uncertainty',
            confidence_col='confidence'
        )
        
        # 1. Model comparison
        print("\n1. Plotting model comparison...")
        self.visualizer.plot_model_comparison(
            observed,
            traditional_pred,
            bayesian_pred,
            save_path=f'{self.output_dir}/figures/model_comparison.png'
        )
        
        # 2. Uncertainty effects
        print("2. Plotting uncertainty effects...")
        self.visualizer.plot_uncertainty_effects(
            self.behavioral_data,
            uncertainty_col='total_uncertainty',
            control_col='control_signal',
            save_path=f'{self.output_dir}/figures/uncertainty_effects.png'
        )
        
        # 3. Block effects
        print("3. Plotting block effects...")
        self.visualizer.plot_block_effects(
            self.behavioral_data,
            block_col='block',
            metrics=['accuracy', 'reaction_time', 'control_signal'],
            save_path=f'{self.output_dir}/figures/block_effects.png'
        )
        
        # 4. Model fit metrics
        print("4. Plotting model fit metrics...")
        metrics_dict = {
            'Traditional EVC': self.results['traditional_evc']['test'],
            'Bayesian EVC': self.results['bayesian_evc']['test']
        }
        self.visualizer.plot_model_fit_metrics(
            metrics_dict,
            save_path=f'{self.output_dir}/figures/model_fit_metrics.png'
        )
        
        # 5. Neural correlates
        print("5. Plotting neural correlates...")
        self.visualizer.plot_neural_correlates(
            self.behavioral_data,
            self.neural_data,
            save_path=f'{self.output_dir}/figures/neural_correlates.png'
        )
        
        # 6. Individual differences
        print("6. Plotting individual differences...")
        self.visualizer.plot_individual_differences(
            self.behavioral_data,
            subject_col='subject_id',
            uncertainty_tolerance_col='uncertainty_tolerance',
            control_col='control_signal',
            save_path=f'{self.output_dir}/figures/individual_differences.png'
        )
        
        print(f"\n✓ All visualizations saved to {self.output_dir}/figures/")
    
    def step7_save_results(self):
        """Step 7: Save comprehensive results."""
        print("\n" + "=" * 70)
        print("STEP 7: SAVING RESULTS")
        print("=" * 70)
        
        # Save model parameters
        params_df = pd.DataFrame({
            'Model': ['Traditional EVC', 'Bayesian EVC'],
            'Reward Weight': [
                self.results['traditional_evc']['train']['reward_weight'],
                self.results['bayesian_evc']['train']['reward_weight']
            ],
            'Effort Cost Weight': [
                self.results['traditional_evc']['train']['effort_cost_weight'],
                self.results['bayesian_evc']['train']['effort_cost_weight']
            ],
            'Uncertainty Weight': [
                0.0,  # Traditional doesn't have this
                self.results['bayesian_evc']['train']['uncertainty_weight']
            ],
            'Effort Exponent': [
                self.results['traditional_evc']['train']['effort_exponent'],
                self.results['bayesian_evc']['train']['effort_exponent']
            ]
        })
        
        params_df.to_csv(f'{self.output_dir}/model_parameters.csv', index=False)
        print(f"\n✓ Model parameters saved to {self.output_dir}/model_parameters.csv")
        
        # Save predictions
        predictions_df = self.test_data.copy()
        predictions_df['traditional_pred'] = self.traditional_model.predict_control(
            self.test_data,
            reward_col='reward_magnitude',
            accuracy_col='evidence_clarity'
        )
        predictions_df['bayesian_pred'] = self.bayesian_model.predict_control(
            self.test_data,
            reward_col='reward_magnitude',
            accuracy_col='evidence_clarity',
            uncertainty_col='total_uncertainty',
            confidence_col='confidence'
        )
        
        predictions_df.to_csv(f'{self.output_dir}/predictions.csv', index=False)
        print(f"✓ Predictions saved to {self.output_dir}/predictions.csv")
        
        print(f"\n✓ All results saved!")
    
    def run_full_pipeline(
        self,
        n_subjects: int = 30,
        n_trials_per_subject: int = 200,
        n_blocks: int = 4,
        train_ratio: float = 0.7
    ):
        """
        Run the complete analysis pipeline.
        
        Args:
            n_subjects: Number of participants
            n_trials_per_subject: Trials per participant
            n_blocks: Number of experimental blocks
            train_ratio: Proportion of data for training
        """
        print("\n" + "=" * 70)
        print("BAYESIAN EVC ANALYSIS PIPELINE")
        print("=" * 70)
        print("\nThis pipeline will:")
        print("  1. Generate experimental data")
        print("  2. Split data into train/test sets")
        print("  3. Fit traditional EVC model")
        print("  4. Fit Bayesian EVC model")
        print("  5. Compare model performance")
        print("  6. Generate visualizations")
        print("  7. Save results")
        print("\n" + "=" * 70)
        
        # Run all steps
        self.step1_generate_data(n_subjects, n_trials_per_subject, n_blocks)
        self.step2_split_data(train_ratio)
        self.step3_fit_traditional_evc()
        self.step4_fit_bayesian_evc()
        self.step5_compare_models()
        self.step6_generate_visualizations()
        self.step7_save_results()
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved to: {self.output_dir}/")
        print(f"Figures saved to: {self.output_dir}/figures/")
        print("\n" + "=" * 70)


def main():
    """Run the main pipeline."""
    # Initialize pipeline
    pipeline = BayesianEVCPipeline(output_dir='results', seed=42)
    
    # Run complete analysis
    pipeline.run_full_pipeline(
        n_subjects=30,
        n_trials_per_subject=200,
        n_blocks=4,
        train_ratio=0.7
    )


if __name__ == '__main__':
    main()

