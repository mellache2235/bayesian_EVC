"""
Step 4: Visualize results.

Creates comprehensive visualizations of:
- Belief evolution over trials
- EVC components
- Control allocation patterns
- Uncertainty dynamics
"""

import sys
sys.path.insert(0, '/Users/hari/.cursor/worktrees/bayesian_EVC/645wx')

from src.pipeline import EVCPipeline, ModelConfig
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

sns.set_style("whitegrid")
sns.set_palette("husl")


def main():
    print("=" * 70)
    print("STEP 4: VISUALIZE RESULTS")
    print("=" * 70)
    
    # Load results
    print("\nLoading results...")
    try:
        results = pd.read_csv('results/evc_scores.csv')
        print(f"✓ Loaded {len(results)} trials")
    except FileNotFoundError:
        print("Running pipeline to generate results...")
        data_path = "data/structured_evc_trials.csv"
        pipeline = EVCPipeline(data_path=data_path, config=ModelConfig())
        results = pipeline.run()
        results.to_csv('results/evc_scores.csv', index=False)
    
    # Create figures directory
    os.makedirs('results/figures', exist_ok=True)
    
    # 1. Belief evolution over trials
    print("\n1. Creating belief evolution plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select first child for detailed view
    child_id = results['child_id'].iloc[0]
    child_data = results[results['child_id'] == child_id]
    
    # Rule confidence
    axes[0, 0].plot(child_data['trial_id'], child_data['posterior_rule_confidence'], 'o-', alpha=0.7)
    axes[0, 0].set_xlabel('Trial')
    axes[0, 0].set_ylabel('Rule Confidence')
    axes[0, 0].set_title(f'Rule Confidence Evolution (Child {child_id})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rule uncertainty
    axes[0, 1].plot(child_data['trial_id'], child_data['posterior_rule_uncertainty'], 'o-', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Trial')
    axes[0, 1].set_ylabel('Rule Uncertainty')
    axes[0, 1].set_title(f'Rule Uncertainty Evolution (Child {child_id})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Evidence precision
    axes[1, 0].plot(child_data['trial_id'], child_data['posterior_evidence_precision'], 'o-', alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Trial')
    axes[1, 0].set_ylabel('Evidence Precision')
    axes[1, 0].set_title(f'Evidence Precision Evolution (Child {child_id})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Control allocation
    axes[1, 1].plot(child_data['trial_id'], child_data['predicted_control_allocation'], 'o-', alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Trial')
    axes[1, 1].set_ylabel('Control Allocation')
    axes[1, 1].set_title(f'Control Allocation (Child {child_id})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/01_belief_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/figures/01_belief_evolution.png")
    
    # 2. EVC components
    print("2. Creating EVC components plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # EVC score distribution
    axes[0, 0].hist(results['evc_score'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(results['evc_score'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel('EVC Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of EVC Scores')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Control allocation distribution
    axes[0, 1].hist(results['predicted_control_allocation'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(results['predicted_control_allocation'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].set_xlabel('Control Allocation')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Control Allocation')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # EVC vs Control
    axes[1, 0].scatter(results['evc_score'], results['predicted_control_allocation'], alpha=0.3, s=20)
    axes[1, 0].set_xlabel('EVC Score')
    axes[1, 0].set_ylabel('Control Allocation')
    axes[1, 0].set_title('EVC Score vs Control Allocation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uncertainty reduction
    axes[1, 1].hist(results['expected_uncertainty_reduction'], bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].axvline(results['expected_uncertainty_reduction'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('Expected Uncertainty Reduction')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Uncertainty Reduction')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/02_evc_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/figures/02_evc_components.png")
    
    # 3. Uncertainty effects
    print("3. Creating uncertainty effects plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Uncertainty vs Control
    axes[0].scatter(results['posterior_rule_uncertainty'], results['predicted_control_allocation'], alpha=0.3, s=20)
    
    # Add trend line
    z = np.polyfit(results['posterior_rule_uncertainty'], results['predicted_control_allocation'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results['posterior_rule_uncertainty'].min(), results['posterior_rule_uncertainty'].max(), 100)
    axes[0].plot(x_trend, p(x_trend), 'r-', lw=2, label='Trend')
    
    axes[0].set_xlabel('Rule Uncertainty')
    axes[0].set_ylabel('Control Allocation')
    axes[0].set_title('Rule Uncertainty vs Control Allocation')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Evidence clarity vs Control
    axes[1].scatter(results['evidence_clarity'], results['predicted_control_allocation'], alpha=0.3, s=20, color='orange')
    
    # Add trend line
    z = np.polyfit(results['evidence_clarity'], results['predicted_control_allocation'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results['evidence_clarity'].min(), results['evidence_clarity'].max(), 100)
    axes[1].plot(x_trend, p(x_trend), 'r-', lw=2, label='Trend')
    
    axes[1].set_xlabel('Evidence Clarity')
    axes[1].set_ylabel('Control Allocation')
    axes[1].set_title('Evidence Clarity vs Control Allocation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/03_uncertainty_effects.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/figures/03_uncertainty_effects.png")
    
    # 4. Per-child summary
    print("4. Creating per-child summary plot...")
    summary = results.groupby('child_id').agg({
        'predicted_control_allocation': 'mean',
        'posterior_rule_uncertainty': 'mean',
        'evc_score': 'mean',
        'predicted_accuracy': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Mean control by child
    axes[0, 0].bar(summary['child_id'].astype(str), summary['predicted_control_allocation'], alpha=0.7)
    axes[0, 0].set_xlabel('Child ID')
    axes[0, 0].set_ylabel('Mean Control Allocation')
    axes[0, 0].set_title('Mean Control Allocation by Child')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Mean uncertainty by child
    axes[0, 1].bar(summary['child_id'].astype(str), summary['posterior_rule_uncertainty'], alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Child ID')
    axes[0, 1].set_ylabel('Mean Rule Uncertainty')
    axes[0, 1].set_title('Mean Rule Uncertainty by Child')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Mean EVC by child
    axes[1, 0].bar(summary['child_id'].astype(str), summary['evc_score'], alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Child ID')
    axes[1, 0].set_ylabel('Mean EVC Score')
    axes[1, 0].set_title('Mean EVC Score by Child')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Mean accuracy by child
    axes[1, 1].bar(summary['child_id'].astype(str), summary['predicted_accuracy'], alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Child ID')
    axes[1, 1].set_ylabel('Mean Predicted Accuracy')
    axes[1, 1].set_title('Mean Predicted Accuracy by Child')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/04_per_child_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: results/figures/04_per_child_summary.png")
    
    print("\n" + "=" * 70)
    print("✓ VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nAll figures saved to: results/figures/")
    print("\nGenerated plots:")
    print("  1. 01_belief_evolution.png - Belief dynamics over trials")
    print("  2. 02_evc_components.png - EVC score components")
    print("  3. 03_uncertainty_effects.png - Uncertainty-control relationships")
    print("  4. 04_per_child_summary.png - Per-child statistics")


if __name__ == "__main__":
    main()

