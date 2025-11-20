"""
Selective Prediction and Accuracy-Coverage Trade-off Analysis

Analyzes the accuracy-coverage trade-off by varying confidence thresholds.
Computes Area Under Risk-Coverage Curve (AURC) as quality metric.

References:
- Geifman & El-Yaniv (2017): Selective Prediction
- Cordella et al. (2023): Risk-Coverage curves for medical AI

Usage:
    python selective_prediction.py \
        --results_dirs ../results/baseline_sgd ../results/ensemble_sgd \
        --method_names "Baseline" "Ensemble" \
        --output_dir outputs/selective_prediction
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score


def compute_accuracy_at_coverage(predictions, labels, uncertainties, coverage):
    """
    Compute accuracy when model abstains on (1-coverage) most uncertain predictions.
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        uncertainties: Array of uncertainty scores (higher = more uncertain)
        coverage: Fraction of predictions to keep (0 to 1)
        
    Returns:
        accuracy: Accuracy on selected predictions
        n_selected: Number of predictions kept
    """
    n_total = len(predictions)
    n_select = int(n_total * coverage)
    
    if n_select == 0:
        return 0.0, 0
    
    # Sort by uncertainty (ascending = most confident first)
    sorted_indices = np.argsort(uncertainties)
    selected_indices = sorted_indices[:n_select]
    
    selected_preds = predictions[selected_indices]
    selected_labels = labels[selected_indices]
    
    accuracy = accuracy_score(selected_labels, selected_preds)
    
    return accuracy, n_select


def compute_aurc(accuracy_curve, coverage_points):
    """
    Compute Area Under Risk-Coverage Curve.
    Lower is better (less risk at each coverage level).
    
    Risk = 1 - Accuracy
    """
    risk_curve = 1 - np.array(accuracy_curve)
    aurc = np.trapz(risk_curve, coverage_points)
    return aurc


def compute_accuracy_coverage_curve(predictions, labels, uncertainties, 
                                     coverage_points=None):
    """
    Compute accuracy at various coverage levels.
    
    Returns:
        coverage_points: Array of coverage levels
        accuracy_curve: Accuracy at each coverage level
    """
    if coverage_points is None:
        coverage_points = np.linspace(0.1, 1.0, 10)
    
    accuracy_curve = []
    n_selected_curve = []
    
    for coverage in coverage_points:
        acc, n_sel = compute_accuracy_at_coverage(predictions, labels, 
                                                    uncertainties, coverage)
        accuracy_curve.append(acc)
        n_selected_curve.append(n_sel)
    
    return coverage_points, accuracy_curve, n_selected_curve


def plot_accuracy_coverage_curves(results_by_method, output_path):
    """
    Plot accuracy-coverage curves for multiple methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Accuracy vs Coverage
    ax1 = axes[0]
    colors = sns.color_palette("husl", len(results_by_method))
    
    for idx, (method_name, data) in enumerate(results_by_method.items()):
        coverage = data['coverage_points']
        accuracy = data['accuracy_curve']
        aurc = data['aurc']
        
        ax1.plot(coverage, accuracy, marker='o', linewidth=2, 
                label=f"{method_name} (AURC={aurc:.4f})", color=colors[idx])
    
    ax1.set_xlabel('Coverage (Fraction of Predictions Accepted)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy-Coverage Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0.5, 1.0])
    
    # Right plot: Risk vs Coverage (for AURC visualization)
    ax2 = axes[1]
    
    for idx, (method_name, data) in enumerate(results_by_method.items()):
        coverage = data['coverage_points']
        risk = 1 - np.array(data['accuracy_curve'])
        
        ax2.plot(coverage, risk, marker='s', linewidth=2, 
                label=method_name, color=colors[idx])
        ax2.fill_between(coverage, 0, risk, alpha=0.2, color=colors[idx])
    
    ax2.set_xlabel('Coverage (Fraction of Predictions Accepted)', fontsize=12)
    ax2.set_ylabel('Risk (1 - Accuracy)', fontsize=12)
    ax2.set_title('Risk-Coverage Curve (Area = AURC)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 0.5])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy-coverage curves to {output_path}")
    plt.close()


def analyze_selective_prediction(args):
    """
    Main analysis function.
    """
    print("\n" + "="*60)
    print("SELECTIVE PREDICTION ANALYSIS")
    print("="*60)
    
    results_by_method = {}
    
    for results_dir, method_name in zip(args.results_dirs, args.method_names):
        print(f"\nAnalyzing: {method_name}")
        print("-" * 40)
        
        # Load results
        results_path = Path(results_dir) / 'test_results.json'
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        predictions = np.array(results['predictions'])
        labels = np.array(results['labels'])
        probabilities = np.array(results['probabilities'])
        
        # Compute uncertainty (entropy)
        # For binary classification: H = -[p*log(p) + (1-p)*log(1-p)]
        probs_class1 = probabilities[:, 1]
        uncertainties = -(probs_class1 * np.log(probs_class1 + 1e-10) + 
                         (1 - probs_class1) * np.log(1 - probs_class1 + 1e-10))
        
        # Compute accuracy-coverage curve
        coverage_points = np.linspace(0.1, 1.0, 19)  # 10%, 15%, ..., 100%
        coverage_points, accuracy_curve, n_selected = compute_accuracy_coverage_curve(
            predictions, labels, uncertainties, coverage_points
        )
        
        # Compute AURC
        aurc = compute_aurc(accuracy_curve, coverage_points)
        
        # Print results
        print(f"  AURC: {aurc:.4f} (lower is better)")
        print(f"\n  Coverage | Accuracy | N Selected | Improvement vs Full")
        print(f"  ---------|----------|------------|--------------------")
        
        full_accuracy = accuracy_score(labels, predictions)
        for cov, acc, n_sel in zip(coverage_points, accuracy_curve, n_selected):
            improvement = acc - full_accuracy
            print(f"  {cov:7.1%} | {acc:8.3f} | {n_sel:10d} | {improvement:+7.3f}")
        
        results_by_method[method_name] = {
            'coverage_points': coverage_points.tolist(),
            'accuracy_curve': accuracy_curve,
            'n_selected_curve': n_selected,
            'aurc': aurc,
            'full_accuracy': full_accuracy
        }
    
    # Plot comparison
    os.makedirs(args.output_dir, exist_ok=True)
    plot_accuracy_coverage_curves(
        results_by_method,
        Path(args.output_dir) / 'accuracy_coverage_curves.png'
    )
    
    # Print clinical interpretation
    print("\n" + "="*60)
    print("CLINICAL INTERPRETATION")
    print("="*60)
    
    # Find best method (lowest AURC)
    best_method = min(results_by_method.items(), key=lambda x: x[1]['aurc'])
    worst_method = max(results_by_method.items(), key=lambda x: x[1]['aurc'])
    
    print(f"\nBest method: {best_method[0]} (AURC={best_method[1]['aurc']:.4f})")
    print(f"Worst method: {worst_method[0]} (AURC={worst_method[1]['aurc']:.4f})")
    
    # Example: Accuracy at 80% coverage
    coverage_80_idx = np.argmin(np.abs(np.array(best_method[1]['coverage_points']) - 0.8))
    best_acc_80 = best_method[1]['accuracy_curve'][coverage_80_idx]
    worst_acc_80 = worst_method[1]['accuracy_curve'][coverage_80_idx]
    
    print(f"\nAt 80% coverage (radiologist reviews 20% most uncertain):")
    print(f"  {best_method[0]}: {best_acc_80:.3f} accuracy")
    print(f"  {worst_method[0]}: {worst_acc_80:.3f} accuracy")
    print(f"  Difference: {best_acc_80 - worst_acc_80:.3f}")
    
    # Compute misdiagnoses prevented
    test_size = len(labels)
    misdiag_best = (1 - best_acc_80) * test_size * 0.8
    misdiag_worst = (1 - worst_acc_80) * test_size * 0.8
    prevented = misdiag_worst - misdiag_best
    
    print(f"\n  For {test_size} patients with 80% automation:")
    print(f"    {best_method[0]}: {misdiag_best:.1f} misdiagnoses")
    print(f"    {worst_method[0]}: {misdiag_worst:.1f} misdiagnoses")
    print(f"    Prevented: {prevented:.1f} misdiagnoses")
    
    # Save summary
    summary = {
        'aurc_by_method': {name: data['aurc'] for name, data in results_by_method.items()},
        'accuracy_at_80_coverage': {
            name: data['accuracy_curve'][np.argmin(np.abs(np.array(data['coverage_points']) - 0.8))]
            for name, data in results_by_method.items()
        },
        'best_method': best_method[0],
        'worst_method': worst_method[0]
    }
    
    summary_path = Path(args.output_dir) / 'selective_prediction_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selective prediction analysis')
    parser.add_argument('--results_dirs', nargs='+', required=True,
                        help='Directories containing test_results.json for each method')
    parser.add_argument('--method_names', nargs='+', required=True,
                        help='Names for each method (same order as results_dirs)')
    parser.add_argument('--output_dir', type=str, default='outputs/selective_prediction',
                        help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    if len(args.results_dirs) != len(args.method_names):
        raise ValueError("Number of results_dirs must match number of method_names")
    
    analyze_selective_prediction(args)
