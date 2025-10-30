"""
Main pipeline for Bayesian EVC modeling.

This script orchestrates the full pipeline:
1. Load/generate data
2. Estimate uncertainties using DDM
3. Fit Bayesian EVC models
4. Compare with traditional EVC
5. Evaluate model performance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from scipy.interpolate import UnivariateSpline

from models.ddm import DriftDiffusionModel
from models.bayesian_evc import BayesianEVC
from models.traditional_evc import TraditionalEVC
from estimation.mcmc_fitting import fit_participants, summarize_posterior


def load_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load behavioral and neural data.
    
    Parameters:
    -----------
    data_dir : str
        Data directory
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        Behavioral and neural data
    """
    behavioral_data = pd.read_csv(f'{data_dir}/behavioral_data.csv')
    neural_data = pd.read_csv(f'{data_dir}/neural_data.csv')
    
    return behavioral_data, neural_data


def estimate_uncertainties(
    behavioral_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Estimate decision uncertainties using DDM.
    
    Parameters:
    -----------
    behavioral_data : pd.DataFrame
        Behavioral data
    
    Returns:
    --------
    pd.DataFrame
        Data with updated uncertainty estimates
    """
    ddm = DriftDiffusionModel()
    
    # Estimate DDM parameters
    ddm_params = ddm.estimate_parameters(
        behavioral_data['reaction_time'].values,
        behavioral_data['choice'].values,
        behavioral_data['correct'].values
    )
    
    # Update DDM with estimated parameters
    ddm.set_parameters(**ddm_params)
    
    # Compute confidence and decision uncertainty
    confidence, decision_uncertainty = ddm.compute_confidence(
        behavioral_data['reaction_time'].values,
        behavioral_data['choice'].values,
        behavioral_data['evidence_clarity'].values
    )
    
    # Update behavioral data
    behavioral_data = behavioral_data.copy()
    behavioral_data['ddm_confidence'] = confidence
    behavioral_data['ddm_decision_uncertainty'] = decision_uncertainty
    
    return behavioral_data


def compare_models(
    behavioral_data: pd.DataFrame,
    bayesian_params: Dict[str, float],
    traditional_params: Dict[str, float]
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Compare Bayesian EVC vs Traditional EVC models.
    
    Parameters:
    -----------
    behavioral_data : pd.DataFrame
        Behavioral data
    bayesian_params : Dict[str, float]
        Bayesian EVC parameters
    traditional_params : Dict[str, float]
        Traditional EVC parameters
    
    Returns:
    --------
    Tuple[Dict[str, float], Dict[str, np.ndarray]]
        Comparison metrics and prediction data for plotting
    """
    # Initialize models
    bayesian_model = BayesianEVC(**bayesian_params)
    traditional_model = TraditionalEVC(**traditional_params)
    
    # Prepare data
    expected_reward = behavioral_data['reward'].values
    decision_uncertainty = behavioral_data['decision_uncertainty'].values
    state_uncertainty = behavioral_data['state_uncertainty'].values
    observed_control = behavioral_data['reaction_time'].values / behavioral_data['reaction_time'].max()
    
    # Predictions
    bayesian_control, bayesian_perf = bayesian_model.predict_behavior(
        expected_reward,
        decision_uncertainty,
        state_uncertainty,
        add_noise=False
    )
    
    traditional_control, traditional_perf = traditional_model.predict_behavior(
        expected_reward,
        add_noise=False
    )
    
    # Compute EVC
    bayesian_evc, bayesian_components = bayesian_model.compute_evc(
        expected_reward,
        observed_control,
        decision_uncertainty,
        state_uncertainty
    )
    
    traditional_evc, traditional_components = traditional_model.compute_evc(
        expected_reward,
        observed_control
    )
    
    # Evaluation metrics
    # Correlation with observed behavior
    bayesian_corr = np.corrcoef(bayesian_control, observed_control)[0, 1]
    traditional_corr = np.corrcoef(traditional_control, observed_control)[0, 1]
    
    # MSE
    bayesian_mse = np.mean((bayesian_control - observed_control) ** 2)
    traditional_mse = np.mean((traditional_control - observed_control) ** 2)
    
    # R-squared
    ss_res_bayesian = np.sum((observed_control - bayesian_control) ** 2)
    ss_tot = np.sum((observed_control - observed_control.mean()) ** 2)
    bayesian_r2 = 1 - (ss_res_bayesian / ss_tot)
    
    ss_res_traditional = np.sum((observed_control - traditional_control) ** 2)
    traditional_r2 = 1 - (ss_res_traditional / ss_tot)
    
    metrics = {
        'bayesian_correlation': bayesian_corr,
        'traditional_correlation': traditional_corr,
        'bayesian_mse': bayesian_mse,
        'traditional_mse': traditional_mse,
        'bayesian_r2': bayesian_r2,
        'traditional_r2': traditional_r2,
        'improvement_r2': bayesian_r2 - traditional_r2,
        'bayesian_evc_mean': bayesian_evc.mean(),
        'traditional_evc_mean': traditional_evc.mean()
    }
    
    # Package prediction data for plotting
    prediction_data = {
        'observed_control': observed_control,
        'bayesian_control': bayesian_control,
        'traditional_control': traditional_control,
        'bayesian_evc': bayesian_evc,
        'traditional_evc': traditional_evc,
        'bayesian_components': bayesian_components,
        'traditional_components': traditional_components,
        'expected_reward': expected_reward,
        'decision_uncertainty': decision_uncertainty,
        'state_uncertainty': state_uncertainty,
        'total_uncertainty': decision_uncertainty + state_uncertainty
    }
    
    return metrics, prediction_data


def visualize_results(
    behavioral_data: pd.DataFrame,
    comparison_metrics: Dict[str, float],
    prediction_data: Dict[str, np.ndarray],
    output_dir: str = 'results'
):
    """
    Create comprehensive visualizations of results.
    
    Parameters:
    -----------
    behavioral_data : pd.DataFrame
        Behavioral data
    comparison_metrics : Dict[str, float]
        Model comparison metrics
    prediction_data : Dict[str, np.ndarray]
        Model predictions and components
    output_dir : str
        Output directory
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # ===== Plot 1: Model Predictions vs Observed =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bayesian EVC predictions
    axes[0, 0].scatter(
        prediction_data['observed_control'],
        prediction_data['bayesian_control'],
        alpha=0.5,
        s=20
    )
    # Add diagonal line
    min_val = min(prediction_data['observed_control'].min(), prediction_data['bayesian_control'].min())
    max_val = max(prediction_data['observed_control'].max(), prediction_data['bayesian_control'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Observed Control (RT)')
    axes[0, 0].set_ylabel('Predicted Control')
    axes[0, 0].set_title(f'Bayesian EVC Predictions (R² = {comparison_metrics["bayesian_r2"]:.3f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Traditional EVC predictions
    axes[0, 1].scatter(
        prediction_data['observed_control'],
        prediction_data['traditional_control'],
        alpha=0.5,
        s=20,
        color='orange'
    )
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('Observed Control (RT)')
    axes[0, 1].set_ylabel('Predicted Control')
    axes[0, 1].set_title(f'Traditional EVC Predictions (R² = {comparison_metrics["traditional_r2"]:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty vs Observed Control
    axes[1, 0].scatter(
        prediction_data['total_uncertainty'],
        prediction_data['observed_control'],
        alpha=0.5,
        s=20,
        label='Observed'
    )
    axes[1, 0].scatter(
        prediction_data['total_uncertainty'],
        prediction_data['bayesian_control'],
        alpha=0.5,
        s=20,
        label='Bayesian EVC'
    )
    axes[1, 0].set_xlabel('Total Uncertainty')
    axes[1, 0].set_ylabel('Control Level')
    axes[1, 0].set_title('Uncertainty vs Control Allocation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Model comparison metrics
    metrics_to_plot = ['bayesian_r2', 'traditional_r2']
    metric_labels = ['Bayesian EVC', 'Traditional EVC']
    values = [comparison_metrics[m] for m in metrics_to_plot]
    colors = ['#2ecc71', '#e74c3c']
    bars = axes[1, 1].bar(metric_labels, values, color=colors, alpha=0.7)
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Model Fit Comparison')
    axes[1, 1].set_ylim([0, max(1.0, max(values) * 1.1)])
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_model_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== Plot 2: EVC Components Breakdown =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bayesian EVC components
    components = prediction_data['bayesian_components']
    trial_idx = np.arange(len(prediction_data['observed_control']))
    sample_idx = np.random.choice(trial_idx, size=min(200, len(trial_idx)), replace=False)
    
    axes[0, 0].plot(trial_idx[sample_idx], components['reward_benefit'][sample_idx], 
                   label='Reward Benefit', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(trial_idx[sample_idx], -components['effort_cost'][sample_idx], 
                   label='Effort Cost', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(trial_idx[sample_idx], components['uncertainty_benefit'][sample_idx], 
                   label='Uncertainty Benefit', alpha=0.7, linewidth=1.5)
    axes[0, 0].plot(trial_idx[sample_idx], prediction_data['bayesian_evc'][sample_idx], 
                   label='Total EVC', alpha=0.9, linewidth=2, color='black')
    axes[0, 0].set_xlabel('Trial (sampled)')
    axes[0, 0].set_ylabel('EVC Value')
    axes[0, 0].set_title('Bayesian EVC Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Component contributions (pie chart)
    mean_reward = components['reward_benefit'].mean()
    mean_effort = components['effort_cost'].mean()
    mean_uncertainty = components['uncertainty_benefit'].mean()
    mean_evc = prediction_data['bayesian_evc'].mean()
    
    # Normalize components for visualization
    total_abs = abs(mean_reward) + abs(mean_effort) + abs(mean_uncertainty)
    if total_abs > 0:
        sizes = [abs(mean_reward)/total_abs, abs(mean_effort)/total_abs, abs(mean_uncertainty)/total_abs]
        labels = ['Reward\nBenefit', 'Effort\nCost', 'Uncertainty\nBenefit']
        colors_pie = ['#3498db', '#e74c3c', '#2ecc71']
        axes[0, 1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Mean Component Contributions\n(Bayesian EVC)')
    
    # Uncertainty vs EVC
    axes[1, 0].scatter(
        prediction_data['total_uncertainty'],
        prediction_data['bayesian_evc'],
        alpha=0.5,
        s=20,
        label='Bayesian EVC'
    )
    axes[1, 0].scatter(
        prediction_data['total_uncertainty'],
        prediction_data['traditional_evc'],
        alpha=0.5,
        s=20,
        color='orange',
        label='Traditional EVC'
    )
    axes[1, 0].set_xlabel('Total Uncertainty')
    axes[1, 0].set_ylabel('Expected Value of Control')
    axes[1, 0].set_title('Uncertainty Effect on EVC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # EVC comparison histogram
    axes[1, 1].hist(prediction_data['bayesian_evc'], bins=30, alpha=0.6, 
                   label='Bayesian EVC', color='#3498db', density=True)
    axes[1, 1].hist(prediction_data['traditional_evc'], bins=30, alpha=0.6, 
                   label='Traditional EVC', color='#e74c3c', density=True)
    axes[1, 1].set_xlabel('EVC Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('EVC Distribution Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_evc_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== Plot 3: Uncertainty Effects =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Decision uncertainty vs Control
    axes[0, 0].scatter(
        prediction_data['decision_uncertainty'],
        prediction_data['observed_control'],
        alpha=0.4,
        s=15,
        label='Observed'
    )
    # Add trend line
    z = np.polyfit(prediction_data['decision_uncertainty'], prediction_data['observed_control'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(prediction_data['decision_uncertainty'].min(), 
                         prediction_data['decision_uncertainty'].max(), 100)
    axes[0, 0].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[0, 0].set_xlabel('Decision Uncertainty')
    axes[0, 0].set_ylabel('Control Level')
    axes[0, 0].set_title('Decision Uncertainty vs Control')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # State uncertainty vs Control
    axes[0, 1].scatter(
        prediction_data['state_uncertainty'],
        prediction_data['observed_control'],
        alpha=0.4,
        s=15,
        label='Observed'
    )
    z = np.polyfit(prediction_data['state_uncertainty'], prediction_data['observed_control'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(prediction_data['state_uncertainty'].min(), 
                         prediction_data['state_uncertainty'].max(), 100)
    axes[0, 1].plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[0, 1].set_xlabel('State Uncertainty')
    axes[0, 1].set_ylabel('Control Level')
    axes[0, 1].set_title('State Uncertainty vs Control')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Uncertainty vs Accuracy
    axes[1, 0].scatter(
        prediction_data['total_uncertainty'],
        behavioral_data['correct'].values,
        alpha=0.4,
        s=15
    )
    # Add smoothed line
    from scipy.interpolate import UnivariateSpline
    sorted_idx = np.argsort(prediction_data['total_uncertainty'])
    sorted_unc = prediction_data['total_uncertainty'][sorted_idx]
    sorted_acc = behavioral_data['correct'].values[sorted_idx]
    if len(sorted_unc) > 10:
        spline = UnivariateSpline(sorted_unc, sorted_acc, s=len(sorted_unc))
        axes[1, 0].plot(sorted_unc, spline(sorted_unc), "r-", alpha=0.8, linewidth=2, label='Smoothed')
    axes[1, 0].set_xlabel('Total Uncertainty')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Uncertainty vs Performance')
    axes[1, 0].set_ylim([-0.1, 1.1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uncertainty bins and control
    n_bins = 10
    uncertainty_bins = pd.cut(prediction_data['total_uncertainty'], bins=n_bins)
    bin_centers = [interval.mid for interval in uncertainty_bins.cat.categories]
    bin_control_obs = [prediction_data['observed_control'][uncertainty_bins == cat].mean() 
                       for cat in uncertainty_bins.cat.categories]
    bin_control_bayesian = [prediction_data['bayesian_control'][uncertainty_bins == cat].mean() 
                            for cat in uncertainty_bins.cat.categories]
    
    axes[1, 1].plot(bin_centers, bin_control_obs, 'o-', label='Observed', linewidth=2, markersize=8)
    axes[1, 1].plot(bin_centers, bin_control_bayesian, 's-', label='Bayesian EVC', 
                   linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Total Uncertainty (binned)')
    axes[1, 1].set_ylabel('Mean Control Level')
    axes[1, 1].set_title('Control Allocation by Uncertainty Level')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_uncertainty_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== Plot 4: Individual Differences (if multiple participants) =====
    if 'participant_id' in behavioral_data.columns:
        n_participants = behavioral_data['participant_id'].nunique()
        if n_participants > 1:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Participant-level correlations
            participant_corrs_bayesian = []
            participant_corrs_traditional = []
            participant_ids = []
            
            for pid in behavioral_data['participant_id'].unique():
                pid_mask = behavioral_data['participant_id'] == pid
                if pid_mask.sum() > 10:  # Need enough trials
                    obs_pid = prediction_data['observed_control'][pid_mask]
                    bayesian_pid = prediction_data['bayesian_control'][pid_mask]
                    traditional_pid = prediction_data['traditional_control'][pid_mask]
                    
                    corr_bayesian = np.corrcoef(obs_pid, bayesian_pid)[0, 1]
                    corr_traditional = np.corrcoef(obs_pid, traditional_pid)[0, 1]
                    
                    if not np.isnan(corr_bayesian) and not np.isnan(corr_traditional):
                        participant_corrs_bayesian.append(corr_bayesian)
                        participant_corrs_traditional.append(corr_traditional)
                        participant_ids.append(pid)
            
            if len(participant_corrs_bayesian) > 0:
                x_pos = np.arange(len(participant_ids))
                width = 0.35
                
                axes[0, 0].bar(x_pos - width/2, participant_corrs_bayesian, width, 
                              label='Bayesian EVC', alpha=0.7, color='#3498db')
                axes[0, 0].bar(x_pos + width/2, participant_corrs_traditional, width, 
                              label='Traditional EVC', alpha=0.7, color='#e74c3c')
                axes[0, 0].set_xlabel('Participant ID')
                axes[0, 0].set_ylabel('Correlation (Predicted vs Observed)')
                axes[0, 0].set_title('Model Fit by Participant')
                axes[0, 0].set_xticks(x_pos)
                axes[0, 0].set_xticklabels([str(int(pid)) for pid in participant_ids], rotation=45)
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3, axis='y')
                axes[0, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            # Mean control by participant
            mean_control_by_participant = behavioral_data.groupby('participant_id')['reaction_time'].mean()
            axes[0, 1].bar(range(len(mean_control_by_participant)), mean_control_by_participant.values, 
                          alpha=0.7, color='#9b59b6')
            axes[0, 1].set_xlabel('Participant ID')
            axes[0, 1].set_ylabel('Mean Reaction Time')
            axes[0, 1].set_title('Individual Differences in Control')
            axes[0, 1].set_xticks(range(len(mean_control_by_participant)))
            axes[0, 1].set_xticklabels([str(int(pid)) for pid in mean_control_by_participant.index], 
                                       rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Accuracy by participant
            accuracy_by_participant = behavioral_data.groupby('participant_id')['correct'].mean()
            axes[1, 0].bar(range(len(accuracy_by_participant)), accuracy_by_participant.values, 
                         alpha=0.7, color='#2ecc71')
            axes[1, 0].set_xlabel('Participant ID')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_title('Individual Differences in Performance')
            axes[1, 0].set_xticks(range(len(accuracy_by_participant)))
            axes[1, 0].set_xticklabels([str(int(pid)) for pid in accuracy_by_participant.index], 
                                       rotation=45)
            axes[1, 0].set_ylim([0, 1])
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Uncertainty tolerance (if computed)
            total_uncertainty_by_participant = behavioral_data.groupby('participant_id').apply(
                lambda x: (x['decision_uncertainty'] + x['state_uncertainty']).mean()
            )
            axes[1, 1].bar(range(len(total_uncertainty_by_participant)), 
                          total_uncertainty_by_participant.values, 
                          alpha=0.7, color='#f39c12')
            axes[1, 1].set_xlabel('Participant ID')
            axes[1, 1].set_ylabel('Mean Total Uncertainty')
            axes[1, 1].set_title('Individual Differences in Uncertainty')
            axes[1, 1].set_xticks(range(len(total_uncertainty_by_participant)))
            axes[1, 1].set_xticklabels([str(int(pid)) for pid in total_uncertainty_by_participant.index], 
                                      rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/04_individual_differences.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # ===== Plot 5: Comprehensive Model Comparison =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Correlation comparison
    corr_metrics = ['bayesian_correlation', 'traditional_correlation']
    corr_values = [comparison_metrics[m] for m in corr_metrics]
    axes[0, 0].bar(['Bayesian EVC', 'Traditional EVC'], corr_values, 
                  color=['#3498db', '#e74c3c'], alpha=0.7)
    axes[0, 0].set_ylabel('Correlation')
    axes[0, 0].set_title('Prediction Correlation Comparison')
    axes[0, 0].set_ylim([0, max(1.0, max(corr_values) * 1.1)])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(corr_values):
        axes[0, 0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MSE comparison
    mse_metrics = ['bayesian_mse', 'traditional_mse']
    mse_values = [comparison_metrics[m] for m in mse_metrics]
    axes[0, 1].bar(['Bayesian EVC', 'Traditional EVC'], mse_values, 
                  color=['#3498db', '#e74c3c'], alpha=0.7)
    axes[0, 1].set_ylabel('Mean Squared Error')
    axes[0, 1].set_title('Prediction Error Comparison')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mse_values):
        axes[0, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Improvement metrics
    improvement_metrics = {
        'R² Improvement': comparison_metrics['improvement_r2'],
        'Corr Improvement': comparison_metrics['bayesian_correlation'] - comparison_metrics['traditional_correlation'],
        'MSE Reduction': comparison_metrics['traditional_mse'] - comparison_metrics['bayesian_mse']
    }
    axes[1, 0].bar(range(len(improvement_metrics)), list(improvement_metrics.values()), 
                  color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.7)
    axes[1, 0].set_xticks(range(len(improvement_metrics)))
    axes[1, 0].set_xticklabels(list(improvement_metrics.keys()), rotation=15, ha='right')
    axes[1, 0].set_ylabel('Improvement')
    axes[1, 0].set_title('Bayesian EVC Improvements')
    axes[1, 0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(improvement_metrics.values()):
        axes[1, 0].text(i, v, f'{v:.3f}', ha='center', 
                       va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    # Summary statistics table
    axes[1, 1].axis('off')
    summary_data = [
        ['Metric', 'Bayesian EVC', 'Traditional EVC'],
        ['R²', f"{comparison_metrics['bayesian_r2']:.3f}", f"{comparison_metrics['traditional_r2']:.3f}"],
        ['Correlation', f"{comparison_metrics['bayesian_correlation']:.3f}", 
         f"{comparison_metrics['traditional_correlation']:.3f}"],
        ['MSE', f"{comparison_metrics['bayesian_mse']:.3f}", f"{comparison_metrics['traditional_mse']:.3f}"],
        ['Mean EVC', f"{comparison_metrics['bayesian_evc_mean']:.3f}", 
         f"{comparison_metrics['traditional_evc_mean']:.3f}"]
    ]
    table = axes[1, 1].table(cellText=summary_data, cellLoc='center', loc='center',
                             colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    axes[1, 1].set_title('Model Comparison Summary', pad=20, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")
    print(f"  - 01_model_predictions.png")
    print(f"  - 02_evc_components.png")
    print(f"  - 03_uncertainty_effects.png")
    if 'participant_id' in behavioral_data.columns and behavioral_data['participant_id'].nunique() > 1:
        print(f"  - 04_individual_differences.png")
    print(f"  - 05_model_comparison.png")


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("Bayesian EVC Modeling Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n1. Loading data...")
    try:
        behavioral_data, neural_data = load_data()
        print(f"   Loaded {len(behavioral_data)} behavioral trials")
        print(f"   Loaded {len(neural_data)} neural data points")
    except FileNotFoundError:
        print("   Data not found. Generating dummy data...")
        from generate_dummy_data import generate_behavioral_data, generate_neural_data, save_data
        
        behavioral_data = generate_behavioral_data(n_participants=20, n_trials_per_participant=200)
        neural_data = generate_neural_data(behavioral_data)
        save_data(behavioral_data, neural_data)
        print("   Data generated and saved.")
    
    # Step 2: Estimate uncertainties using DDM
    print("\n2. Estimating uncertainties using DDM...")
    behavioral_data = estimate_uncertainties(behavioral_data)
    print("   Uncertainties estimated.")
    
    # Step 3: Fit models (simplified - using MLE for demonstration)
    print("\n3. Fitting models...")
    print("   (Using simplified parameter estimation for demonstration)")
    
    # For demonstration, use reasonable default parameters
    # In full implementation, use MCMC fitting
    bayesian_params = {
        'reward_sensitivity': 1.0,
        'effort_cost': 0.5,
        'uncertainty_reduction_weight': 0.3,
        'uncertainty_tolerance': 0.5,
        'control_efficiency': 1.0
    }
    
    traditional_params = {
        'reward_sensitivity': 1.0,
        'effort_cost': 0.5
    }
    
    # Step 4: Compare models
    print("\n4. Comparing models...")
    comparison_metrics, prediction_data = compare_models(
        behavioral_data,
        bayesian_params,
        traditional_params
    )
    
    print("\n   Model Comparison Results:")
    print(f"   Bayesian EVC R²:      {comparison_metrics['bayesian_r2']:.3f}")
    print(f"   Traditional EVC R²:   {comparison_metrics['traditional_r2']:.3f}")
    print(f"   Improvement:          {comparison_metrics['improvement_r2']:.3f}")
    print(f"   Bayesian Correlation: {comparison_metrics['bayesian_correlation']:.3f}")
    print(f"   Traditional Corr:     {comparison_metrics['traditional_correlation']:.3f}")
    
    # Step 5: Visualize results
    print("\n5. Creating visualizations...")
    visualize_results(behavioral_data, comparison_metrics, prediction_data)
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

