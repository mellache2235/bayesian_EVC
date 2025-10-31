"""
Visualize Parameter Convergence During Model Fitting

This script shows how model parameters evolve during the optimization process.
It helps you understand:
1. Whether parameters converge to stable values
2. How quickly they converge
3. If there are any optimization issues (oscillations, getting stuck)

This is useful for:
- Debugging model fitting
- Understanding parameter identifiability
- Checking if optimization is working properly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from models.traditional_evc import TraditionalEVC
from models.bayesian_evc import BayesianEVC

sns.set_style("whitegrid")
sns.set_palette("husl")


class ParameterTracker:
    """Tracks parameter values during optimization."""
    
    def __init__(self):
        self.history = []
        self.iteration = 0
    
    def callback(self, params):
        """Called after each optimization iteration."""
        self.history.append({
            'iteration': self.iteration,
            'params': params.copy()
        })
        self.iteration += 1
    
    def get_history_df(self, param_names):
        """Convert history to DataFrame."""
        if not self.history:
            return None
        
        data = []
        for record in self.history:
            row = {'iteration': record['iteration']}
            for i, name in enumerate(param_names):
                row[name] = record['params'][i]
            data.append(row)
        
        return pd.DataFrame(data)


def fit_traditional_with_tracking(data, observed_control_col='control_signal',
                                  reward_col='reward_magnitude', 
                                  accuracy_col='evidence_clarity'):
    """
    Fit Traditional EVC model while tracking parameter evolution.
    
    Returns:
        model: Fitted model
        history_df: DataFrame with parameter values at each iteration
    """
    print("Fitting Traditional EVC model with parameter tracking...")
    
    model = TraditionalEVC()
    observed_control = data[observed_control_col].values
    
    # Create tracker
    tracker = ParameterTracker()
    
    # Objective function
    def objective(params):
        model.baseline = params[0]
        model.reward_weight = params[1]
        model.effort_cost_weight = params[2]
        model.effort_exponent = params[3]
        
        predicted = model.predict_control(data, reward_col, accuracy_col)
        mse = np.mean((predicted - observed_control) ** 2)
        
        # Track parameters
        tracker.callback(params)
        
        return mse
    
    # Optimize
    initial_params = [model.baseline, model.reward_weight, 
                     model.effort_cost_weight, model.effort_exponent]
    bounds = [(0.0, 1.0), (0.01, 10.0), (0.01, 10.0), (1.0, 3.0)]
    
    result = minimize(objective, x0=initial_params, bounds=bounds, 
                     method='L-BFGS-B', options={'maxiter': 100})
    
    # Update final parameters
    model.baseline = result.x[0]
    model.reward_weight = result.x[1]
    model.effort_cost_weight = result.x[2]
    model.effort_exponent = result.x[3]
    
    # Get history
    param_names = ['baseline', 'reward_weight', 'effort_cost_weight', 'effort_exponent']
    history_df = tracker.get_history_df(param_names)
    
    print(f"✓ Optimization completed in {len(tracker.history)} iterations")
    
    return model, history_df


def fit_bayesian_with_tracking(data, observed_control_col='control_signal',
                               reward_col='reward_magnitude',
                               accuracy_col='evidence_clarity',
                               uncertainty_col='total_uncertainty',
                               confidence_col='confidence'):
    """
    Fit Bayesian EVC model while tracking parameter evolution.
    
    Returns:
        model: Fitted model
        history_df: DataFrame with parameter values at each iteration
    """
    print("Fitting Bayesian EVC model with parameter tracking...")
    
    model = BayesianEVC()
    observed_control = data[observed_control_col].values
    
    # Create tracker
    tracker = ParameterTracker()
    
    # Objective function
    def objective(params):
        model.baseline = params[0]
        model.reward_weight = params[1]
        model.effort_cost_weight = params[2]
        model.uncertainty_weight = params[3]
        model.effort_exponent = params[4]
        
        predicted = model.predict_control(data, reward_col, accuracy_col,
                                         uncertainty_col, confidence_col)
        mse = np.mean((predicted - observed_control) ** 2)
        
        # Track parameters
        tracker.callback(params)
        
        return mse
    
    # Optimize
    initial_params = [model.baseline, model.reward_weight, 
                     model.effort_cost_weight, model.uncertainty_weight,
                     model.effort_exponent]
    bounds = [(0.0, 1.0), (0.01, 10.0), (0.01, 10.0), (0.0, 5.0), (1.0, 3.0)]
    
    result = minimize(objective, x0=initial_params, bounds=bounds,
                     method='L-BFGS-B', options={'maxiter': 100})
    
    # Update final parameters
    model.baseline = result.x[0]
    model.reward_weight = result.x[1]
    model.effort_cost_weight = result.x[2]
    model.uncertainty_weight = result.x[3]
    model.effort_exponent = result.x[4]
    
    # Get history
    param_names = ['baseline', 'reward_weight', 'effort_cost_weight', 
                   'uncertainty_weight', 'effort_exponent']
    history_df = tracker.get_history_df(param_names)
    
    print(f"✓ Optimization completed in {len(tracker.history)} iterations")
    
    return model, history_df


def plot_parameter_convergence(history_df, model_name, save_path=None):
    """
    Plot how parameters evolve during optimization.
    
    Args:
        history_df: DataFrame with parameter history
        model_name: Name of the model (for title)
        save_path: Optional path to save figure
    """
    param_cols = [col for col in history_df.columns if col != 'iteration']
    n_params = len(param_cols)
    
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params))
    if n_params == 1:
        axes = [axes]
    
    for i, param in enumerate(param_cols):
        ax = axes[i]
        
        # Plot parameter trajectory
        ax.plot(history_df['iteration'], history_df[param], 
               'o-', linewidth=2, markersize=4, alpha=0.7)
        
        # Mark final value
        final_value = history_df[param].iloc[-1]
        ax.axhline(final_value, color='red', linestyle='--', 
                  linewidth=2, alpha=0.5, label=f'Final: {final_value:.4f}')
        
        ax.set_xlabel('Optimization Iteration', fontsize=12)
        ax.set_ylabel(param.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{param.replace("_", " ").title()} Convergence', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Parameter Convergence During Optimization',
                fontsize=16, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved convergence plot to {save_path}")
    
    return fig


def plot_convergence_comparison(trad_history, bayes_history, save_path=None):
    """
    Compare parameter convergence between Traditional and Bayesian models.
    
    Shows side-by-side how parameters evolve in both models.
    """
    # Common parameters
    common_params = ['baseline', 'reward_weight', 'effort_cost_weight', 'effort_exponent']
    
    fig, axes = plt.subplots(len(common_params), 2, figsize=(14, 3*len(common_params)))
    
    for i, param in enumerate(common_params):
        # Traditional model
        ax_trad = axes[i, 0]
        ax_trad.plot(trad_history['iteration'], trad_history[param],
                    'o-', linewidth=2, markersize=4, alpha=0.7, color='blue')
        final_trad = trad_history[param].iloc[-1]
        ax_trad.axhline(final_trad, color='red', linestyle='--', 
                       linewidth=2, alpha=0.5, label=f'Final: {final_trad:.4f}')
        ax_trad.set_ylabel(param.replace('_', ' ').title(), fontsize=12)
        ax_trad.set_title(f'Traditional EVC - {param.replace("_", " ").title()}',
                         fontsize=12, fontweight='bold')
        ax_trad.legend(fontsize=9)
        ax_trad.grid(True, alpha=0.3)
        
        # Bayesian model
        ax_bayes = axes[i, 1]
        ax_bayes.plot(bayes_history['iteration'], bayes_history[param],
                     'o-', linewidth=2, markersize=4, alpha=0.7, color='green')
        final_bayes = bayes_history[param].iloc[-1]
        ax_bayes.axhline(final_bayes, color='red', linestyle='--',
                        linewidth=2, alpha=0.5, label=f'Final: {final_bayes:.4f}')
        ax_bayes.set_ylabel(param.replace('_', ' ').title(), fontsize=12)
        ax_bayes.set_title(f'Bayesian EVC - {param.replace("_", " ").title()}',
                          fontsize=12, fontweight='bold')
        ax_bayes.legend(fontsize=9)
        ax_bayes.grid(True, alpha=0.3)
        
        # Set x-label only on bottom row
        if i == len(common_params) - 1:
            ax_trad.set_xlabel('Optimization Iteration', fontsize=12)
            ax_bayes.set_xlabel('Optimization Iteration', fontsize=12)
    
    # Add uncertainty weight for Bayesian model
    if 'uncertainty_weight' in bayes_history.columns:
        # Add extra row for uncertainty weight
        fig.set_figheight(fig.get_figheight() + 3)
        ax_unc = fig.add_subplot(len(common_params)+1, 2, (len(common_params)*2)+2)
        ax_unc.plot(bayes_history['iteration'], bayes_history['uncertainty_weight'],
                   'o-', linewidth=2, markersize=4, alpha=0.7, color='purple')
        final_unc = bayes_history['uncertainty_weight'].iloc[-1]
        ax_unc.axhline(final_unc, color='red', linestyle='--',
                      linewidth=2, alpha=0.5, label=f'Final: {final_unc:.4f}')
        ax_unc.set_xlabel('Optimization Iteration', fontsize=12)
        ax_unc.set_ylabel('Uncertainty Weight', fontsize=12)
        ax_unc.set_title('Bayesian EVC - Uncertainty Weight (KEY PARAMETER!)',
                        fontsize=12, fontweight='bold')
        ax_unc.legend(fontsize=9)
        ax_unc.grid(True, alpha=0.3)
    
    plt.suptitle('Parameter Convergence Comparison: Traditional vs. Bayesian EVC',
                fontsize=16, fontweight='bold', y=0.999)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison plot to {save_path}")
    
    return fig


def main():
    """
    Main function to visualize parameter convergence.
    """
    print("=" * 70)
    print("PARAMETER CONVERGENCE VISUALIZATION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv('data/behavioral_data.csv')
        print(f"✓ Loaded {len(data)} trials")
    except FileNotFoundError:
        print("✗ Error: data/behavioral_data.csv not found!")
        print("  Please run 'python3 step1_generate_data.py' first.")
        return
    
    # Split data
    print("\nSplitting data (70/30 train/test)...")
    subjects = data['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    
    n_train = int(len(subjects) * 0.7)
    train_subjects = subjects[:n_train]
    train_data = data[data['subject_id'].isin(train_subjects)].copy()
    
    print(f"  Using {len(train_subjects)} subjects, {len(train_data)} trials for training")
    
    # Create results directory
    import os
    os.makedirs('results/convergence', exist_ok=True)
    
    # Fit Traditional EVC with tracking
    print("\n" + "-" * 70)
    trad_model, trad_history = fit_traditional_with_tracking(train_data)
    
    print("\nTraditional EVC Final Parameters:")
    print(f"  Baseline: {trad_model.baseline:.4f}")
    print(f"  Reward weight: {trad_model.reward_weight:.4f}")
    print(f"  Effort cost weight: {trad_model.effort_cost_weight:.4f}")
    print(f"  Effort exponent: {trad_model.effort_exponent:.4f}")
    
    # Plot Traditional convergence
    plot_parameter_convergence(trad_history, 'Traditional EVC',
                              save_path='results/convergence/traditional_convergence.png')
    
    # Fit Bayesian EVC with tracking
    print("\n" + "-" * 70)
    bayes_model, bayes_history = fit_bayesian_with_tracking(train_data)
    
    print("\nBayesian EVC Final Parameters:")
    print(f"  Baseline: {bayes_model.baseline:.4f}")
    print(f"  Reward weight: {bayes_model.reward_weight:.4f}")
    print(f"  Effort cost weight: {bayes_model.effort_cost_weight:.4f}")
    print(f"  Uncertainty weight: {bayes_model.uncertainty_weight:.4f} ← KEY!")
    print(f"  Effort exponent: {bayes_model.effort_exponent:.4f}")
    
    # Plot Bayesian convergence
    plot_parameter_convergence(bayes_history, 'Bayesian EVC',
                              save_path='results/convergence/bayesian_convergence.png')
    
    # Plot comparison
    print("\n" + "-" * 70)
    print("Creating comparison plot...")
    plot_convergence_comparison(trad_history, bayes_history,
                               save_path='results/convergence/convergence_comparison.png')
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY")
    print("=" * 70)
    
    print("\nTraditional EVC:")
    print(f"  Iterations to converge: {len(trad_history)}")
    print(f"  Parameter stability (last 5 iterations):")
    for param in ['baseline', 'reward_weight', 'effort_cost_weight', 'effort_exponent']:
        last_5 = trad_history[param].tail(5)
        std = last_5.std()
        print(f"    {param}: std = {std:.6f} {'✓ Stable' if std < 0.01 else '⚠ Still changing'}")
    
    print("\nBayesian EVC:")
    print(f"  Iterations to converge: {len(bayes_history)}")
    print(f"  Parameter stability (last 5 iterations):")
    for param in ['baseline', 'reward_weight', 'effort_cost_weight', 
                  'uncertainty_weight', 'effort_exponent']:
        last_5 = bayes_history[param].tail(5)
        std = last_5.std()
        print(f"    {param}: std = {std:.6f} {'✓ Stable' if std < 0.01 else '⚠ Still changing'}")
    
    print("\n" + "=" * 70)
    print("✓ CONVERGENCE VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated plots:")
    print("  1. results/convergence/traditional_convergence.png")
    print("  2. results/convergence/bayesian_convergence.png")
    print("  3. results/convergence/convergence_comparison.png")
    
    print("\nInterpretation:")
    print("  - Flat lines at end = parameters converged")
    print("  - Oscillations = optimization struggling")
    print("  - Monotonic change = smooth convergence")
    print("  - Early plateau = fast convergence")


if __name__ == '__main__':
    main()

