"""Main pipeline for Bayesian EVC modeling."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from models.ddm import DriftDiffusionModel
from models.bayesian_evc import BayesianEVC
from models.traditional_evc import TraditionalEVC


def load_data(data_dir='data'):
    return pd.read_csv(f'{data_dir}/behavioral_data.csv'), pd.read_csv(f'{data_dir}/neural_data.csv')


def estimate_uncertainties(behavioral_data):
    ddm = DriftDiffusionModel()
    ddm.set_parameters(**ddm.estimate_parameters(
        behavioral_data['reaction_time'].values,
        behavioral_data['choice'].values,
        behavioral_data['correct'].values))
    confidence, decision_uncertainty = ddm.compute_confidence(
        behavioral_data['reaction_time'].values,
        behavioral_data['choice'].values,
        behavioral_data['evidence_clarity'].values)
    behavioral_data = behavioral_data.copy()
    behavioral_data['ddm_confidence'] = confidence
    behavioral_data['ddm_decision_uncertainty'] = decision_uncertainty
    return behavioral_data


def compare_models(behavioral_data, bayesian_params, traditional_params):
    bayesian_model = BayesianEVC(**bayesian_params)
    traditional_model = TraditionalEVC(**traditional_params)
    
    reward = behavioral_data['reward'].values
    dec_unc = behavioral_data['decision_uncertainty'].values
    state_unc = behavioral_data['state_uncertainty'].values
    obs_control = behavioral_data['reaction_time'].values / behavioral_data['reaction_time'].max()
    
    bay_control, _ = bayesian_model.predict_behavior(reward, dec_unc, state_unc)
    trad_control, _ = traditional_model.predict_behavior(reward)
    
    bay_evc, bay_comp = bayesian_model.compute_evc(reward, obs_control, dec_unc, state_unc)
    trad_evc, _ = traditional_model.compute_evc(reward, obs_control)
    
    ss_res_bay = np.sum((obs_control - bay_control) ** 2)
    ss_res_trad = np.sum((obs_control - trad_control) ** 2)
    ss_tot = np.sum((obs_control - obs_control.mean()) ** 2)
    
    metrics = {
        'bayesian_r2': 1 - (ss_res_bay / ss_tot),
        'traditional_r2': 1 - (ss_res_trad / ss_tot),
        'bayesian_correlation': np.corrcoef(bay_control, obs_control)[0, 1],
        'traditional_correlation': np.corrcoef(trad_control, obs_control)[0, 1],
        'bayesian_evc_mean': bay_evc.mean(),
        'traditional_evc_mean': trad_evc.mean(),
        'improvement_r2': 0
    }
    metrics['improvement_r2'] = metrics['bayesian_r2'] - metrics['traditional_r2']
    
    return metrics, {
        'observed_control': obs_control,
        'bayesian_control': bay_control,
        'traditional_control': trad_control,
        'bayesian_evc': bay_evc,
        'traditional_evc': trad_evc,
        'bayesian_components': bay_comp,
        'total_uncertainty': dec_unc + state_unc
    }


def visualize_results(comparison_metrics, prediction_data, output_dir='results'):
    Path(output_dir).mkdir(exist_ok=True)
    sns.set_style('whitegrid')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    obs, bay, trad = prediction_data['observed_control'], prediction_data['bayesian_control'], prediction_data['traditional_control']
    
    for ax, pred, title, r2 in zip([axes[0, 0], axes[0, 1]], [bay, trad],
                                   ['Bayesian EVC', 'Traditional EVC'],
                                   [comparison_metrics['bayesian_r2'], comparison_metrics['traditional_r2']]):
        ax.scatter(obs, pred, alpha=0.5, s=15)
        m = min(obs.min(), pred.min())
        M = max(obs.max(), pred.max())
        ax.plot([m, M], [m, M], 'r--', lw=2)
        ax.set_xlabel('Observed Control')
        ax.set_ylabel('Predicted Control')
        ax.set_title(f'{title} (R² = {r2:.3f})')
        ax.grid(True, alpha=0.3)
    
    axes[1, 0].scatter(prediction_data['total_uncertainty'], obs, alpha=0.4, s=15, label='Observed')
    axes[1, 0].scatter(prediction_data['total_uncertainty'], bay, alpha=0.4, s=15, label='Bayesian EVC')
    axes[1, 0].set_xlabel('Total Uncertainty')
    axes[1, 0].set_ylabel('Control Level')
    axes[1, 0].set_title('Uncertainty vs Control')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    bars = axes[1, 1].bar(['Bayesian', 'Traditional'],
                         [comparison_metrics['bayesian_r2'], comparison_metrics['traditional_r2']],
                         color=['#2ecc71', '#e74c3c'], alpha=0.7)
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Model Comparison')
    axes[1, 1].set_ylim([0, 1])
    for bar, val in zip(bars, [comparison_metrics['bayesian_r2'], comparison_metrics['traditional_r2']]):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    comp = prediction_data['bayesian_components']
    idx = np.random.choice(len(obs), size=min(200, len(obs)), replace=False)
    
    axes[0].plot(idx, comp['reward_benefit'][idx], label='Reward', alpha=0.7)
    axes[0].plot(idx, -comp['effort_cost'][idx], label='Effort Cost', alpha=0.7)
    axes[0].plot(idx, comp['uncertainty_benefit'][idx], label='Uncertainty', alpha=0.7)
    axes[0].plot(idx, prediction_data['bayesian_evc'][idx], label='EVC', color='black', linewidth=2)
    axes[0].set_xlabel('Trial')
    axes[0].set_ylabel('Value')
    axes[0].set_title('EVC Components')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(prediction_data['total_uncertainty'], prediction_data['bayesian_evc'],
                   alpha=0.5, s=15, label='Bayesian')
    axes[1].scatter(prediction_data['total_uncertainty'], prediction_data['traditional_evc'],
                   alpha=0.5, s=15, color='orange', label='Traditional')
    axes[1].set_xlabel('Total Uncertainty')
    axes[1].set_ylabel('EVC')
    axes[1].set_title('Uncertainty Effect on EVC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/evc_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}/")


def main():
    print("Bayesian EVC Modeling Pipeline")
    print("=" * 50)
    
    print("\n1. Loading data...")
    try:
        behavioral_data, _ = load_data()
        print(f"   Loaded {len(behavioral_data)} trials")
    except FileNotFoundError:
        print("   Generating dummy data...")
        from generate_dummy_data import generate_behavioral_data, generate_neural_data, save_data
        behavioral_data = generate_behavioral_data(n_participants=20, n_trials_per_participant=200)
        save_data(behavioral_data, generate_neural_data(behavioral_data))
    
    print("\n2. Estimating uncertainties...")
    behavioral_data = estimate_uncertainties(behavioral_data)
    
    print("\n3. Comparing models...")
    bayesian_params = {'reward_sensitivity': 1.0, 'effort_cost': 0.5,
                      'uncertainty_reduction_weight': 0.3, 'uncertainty_tolerance': 0.5,
                      'control_efficiency': 1.0}
    traditional_params = {'reward_sensitivity': 1.0, 'effort_cost': 0.5}
    
    comparison_metrics, prediction_data = compare_models(behavioral_data, bayesian_params, traditional_params)
    
    print(f"\n   Bayesian EVC R²:    {comparison_metrics['bayesian_r2']:.3f}")
    print(f"   Traditional EVC R²:  {comparison_metrics['traditional_r2']:.3f}")
    print(f"   Improvement:         {comparison_metrics['improvement_r2']:.3f}")
    
    print("\n4. Creating visualizations...")
    visualize_results(comparison_metrics, prediction_data)
    
    print("\n" + "=" * 50)
    print("Pipeline completed!")


if __name__ == '__main__':
    main()
