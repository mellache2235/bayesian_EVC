"""
Visualization utilities for Bayesian EVC analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")


class EVCVisualizer:
    """Visualization tools for EVC model analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
    
    def plot_model_comparison(
        self,
        observed: np.ndarray,
        traditional_pred: np.ndarray,
        bayesian_pred: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of model predictions.
        
        Args:
            observed: Observed control signals
            traditional_pred: Traditional EVC predictions
            bayesian_pred: Bayesian EVC predictions
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Traditional EVC
        axes[0].scatter(observed, traditional_pred, alpha=0.5, s=20)
        axes[0].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect prediction')
        axes[0].set_xlabel('Observed Control', fontsize=12)
        axes[0].set_ylabel('Predicted Control', fontsize=12)
        axes[0].set_title('Traditional EVC', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        
        # Bayesian EVC
        axes[1].scatter(observed, bayesian_pred, alpha=0.5, s=20, color='orange')
        axes[1].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect prediction')
        axes[1].set_xlabel('Observed Control', fontsize=12)
        axes[1].set_ylabel('Predicted Control', fontsize=12)
        axes[1].set_title('Bayesian EVC', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        
        # Residuals comparison
        trad_residuals = observed - traditional_pred
        bayes_residuals = observed - bayesian_pred
        
        axes[2].hist(trad_residuals, bins=30, alpha=0.5, label='Traditional EVC')
        axes[2].hist(bayes_residuals, bins=30, alpha=0.5, label='Bayesian EVC')
        axes[2].axvline(0, color='red', linestyle='--', lw=2)
        axes[2].set_xlabel('Residuals', fontsize=12)
        axes[2].set_ylabel('Frequency', fontsize=12)
        axes[2].set_title('Prediction Residuals', fontsize=14, fontweight='bold')
        axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_uncertainty_effects(
        self,
        data: pd.DataFrame,
        uncertainty_col: str = 'total_uncertainty',
        control_col: str = 'control_signal',
        save_path: Optional[str] = None
    ):
        """
        Plot relationship between uncertainty and control allocation.
        
        Args:
            data: DataFrame with trial data
            uncertainty_col: Column name for uncertainty
            control_col: Column name for control signal
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Scatter plot
        axes[0].scatter(
            data[uncertainty_col],
            data[control_col],
            alpha=0.3,
            s=20
        )
        
        # Add trend line
        z = np.polyfit(data[uncertainty_col], data[control_col], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(data[uncertainty_col].min(), data[uncertainty_col].max(), 100)
        axes[0].plot(x_trend, p(x_trend), 'r-', lw=2, label='Trend')
        
        axes[0].set_xlabel('Uncertainty', fontsize=12)
        axes[0].set_ylabel('Control Signal', fontsize=12)
        axes[0].set_title('Uncertainty vs Control Allocation', fontsize=14, fontweight='bold')
        axes[0].legend()
        
        # Binned analysis
        data['uncertainty_bin'] = pd.cut(data[uncertainty_col], bins=5)
        binned = data.groupby('uncertainty_bin')[control_col].agg(['mean', 'sem'])
        
        bin_centers = [interval.mid for interval in binned.index]
        axes[1].errorbar(
            bin_centers,
            binned['mean'],
            yerr=binned['sem'],
            fmt='o-',
            capsize=5,
            markersize=8,
            linewidth=2
        )
        axes[1].set_xlabel('Uncertainty (binned)', fontsize=12)
        axes[1].set_ylabel('Mean Control Signal', fontsize=12)
        axes[1].set_title('Control by Uncertainty Level', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_block_effects(
        self,
        data: pd.DataFrame,
        block_col: str = 'block',
        metrics: List[str] = ['accuracy', 'reaction_time', 'control_signal'],
        save_path: Optional[str] = None
    ):
        """
        Plot behavioral metrics across experimental blocks.
        
        Args:
            data: DataFrame with trial data
            block_col: Column name for blocks
            metrics: List of metric columns to plot
            save_path: Optional path to save figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            block_stats = data.groupby(block_col)[metric].agg(['mean', 'sem'])
            
            ax.errorbar(
                block_stats.index,
                block_stats['mean'],
                yerr=block_stats['sem'],
                fmt='o-',
                capsize=5,
                markersize=10,
                linewidth=2
            )
            ax.set_xlabel('Block', fontsize=12)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric.replace("_", " ").title()} by Block', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(block_stats.index)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_fit_metrics(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of model fit metrics.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metric dicts as values
            save_path: Optional path to save figure
        """
        metrics_to_plot = ['r2', 'rmse', 'correlation']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        
        models = list(metrics_dict.keys())
        
        for ax, metric in zip(axes, metrics_to_plot):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            
            bars = ax.bar(models, values, alpha=0.7, edgecolor='black', linewidth=2)
            
            # Color bars
            colors = ['steelblue', 'orange']
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'Model Comparison: {metric.upper()}', 
                        fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
            
            # Add value labels on bars
            for i, (model, value) in enumerate(zip(models, values)):
                ax.text(i, value + max(values) * 0.02, f'{value:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_neural_correlates(
        self,
        behavioral_data: pd.DataFrame,
        neural_data: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """
        Plot correlations between behavioral and neural measures.
        
        Args:
            behavioral_data: DataFrame with behavioral data
            neural_data: DataFrame with neural data
            save_path: Optional path to save figure
        """
        # Merge data
        merged = pd.merge(
            behavioral_data,
            neural_data,
            on=['subject_id', 'trial']
        )
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Define relationships to plot
        relationships = [
            ('control_signal', 'dlpfc_activity', 'Control vs DLPFC'),
            ('total_uncertainty', 'acc_activity', 'Uncertainty vs ACC'),
            ('obtained_reward', 'striatal_activity', 'Reward vs Striatum'),
            ('confidence', 'dlpfc_activity', 'Confidence vs DLPFC'),
            ('effort_cost', 'acc_activity', 'Effort vs ACC'),
            ('accuracy', 'striatal_activity', 'Accuracy vs Striatum')
        ]
        
        for ax, (x_col, y_col, title) in zip(axes, relationships):
            ax.scatter(merged[x_col], merged[y_col], alpha=0.3, s=20)
            
            # Add correlation
            corr = merged[[x_col, y_col]].corr().iloc[0, 1]
            
            # Trend line
            z = np.polyfit(merged[x_col], merged[y_col], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(merged[x_col].min(), merged[x_col].max(), 100)
            ax.plot(x_trend, p(x_trend), 'r-', lw=2, alpha=0.7)
            
            ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel(y_col.replace('_', ' ').title(), fontsize=10)
            ax.set_title(f'{title}\nr = {corr:.3f}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_individual_differences(
        self,
        data: pd.DataFrame,
        subject_col: str = 'subject_id',
        uncertainty_tolerance_col: str = 'uncertainty_tolerance',
        control_col: str = 'control_signal',
        save_path: Optional[str] = None
    ):
        """
        Plot individual differences in uncertainty tolerance and control.
        
        Args:
            data: DataFrame with subject-level data
            subject_col: Column identifying subjects
            uncertainty_tolerance_col: Column with uncertainty tolerance
            control_col: Column with control signals
            save_path: Optional path to save figure
        """
        # Aggregate by subject
        subject_data = data.groupby(subject_col).agg({
            uncertainty_tolerance_col: 'first',
            control_col: 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution of uncertainty tolerance
        axes[0].hist(subject_data[uncertainty_tolerance_col], bins=15, 
                    edgecolor='black', alpha=0.7)
        axes[0].axvline(subject_data[uncertainty_tolerance_col].mean(), 
                       color='red', linestyle='--', lw=2, label='Mean')
        axes[0].set_xlabel('Uncertainty Tolerance', fontsize=12)
        axes[0].set_ylabel('Number of Subjects', fontsize=12)
        axes[0].set_title('Distribution of Uncertainty Tolerance', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        
        # Relationship with control
        axes[1].scatter(
            subject_data[uncertainty_tolerance_col],
            subject_data[control_col],
            s=100,
            alpha=0.6,
            edgecolor='black'
        )
        
        # Trend line
        z = np.polyfit(subject_data[uncertainty_tolerance_col], 
                      subject_data[control_col], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(
            subject_data[uncertainty_tolerance_col].min(),
            subject_data[uncertainty_tolerance_col].max(),
            100
        )
        axes[1].plot(x_trend, p(x_trend), 'r-', lw=2, label='Trend')
        
        corr = subject_data[[uncertainty_tolerance_col, control_col]].corr().iloc[0, 1]
        
        axes[1].set_xlabel('Uncertainty Tolerance', fontsize=12)
        axes[1].set_ylabel('Mean Control Signal', fontsize=12)
        axes[1].set_title(f'Uncertainty Tolerance vs Control\nr = {corr:.3f}',
                         fontsize=14, fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def main():
    """Demonstrate visualization utilities."""
    print("Visualization utilities loaded successfully!")
    print("\nAvailable plotting functions:")
    print("  - plot_model_comparison()")
    print("  - plot_uncertainty_effects()")
    print("  - plot_block_effects()")
    print("  - plot_model_fit_metrics()")
    print("  - plot_neural_correlates()")
    print("  - plot_individual_differences()")


if __name__ == '__main__':
    main()

