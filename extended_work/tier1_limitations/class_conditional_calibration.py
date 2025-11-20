"""
Class-Conditional Calibration Analysis

Analyzes calibration metrics (ECE, conformal set sizes) separately for each class
to detect potential bias and unfairness in uncertainty estimates.

References:
- Angelopoulos et al. (2021): Class-conditional conformal prediction
- Romano et al. (2020): Stratified conformal prediction

Usage:
    python class_conditional_calibration.py \
        --results_dir ../results/baseline_sgd \
        --calibration_file ../results/baseline_sgd/conformal_calibration.json \
        --output_dir outputs/class_conditional
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# Import from existing codebase
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from conformal_prediction import AdaptivePredictionSets
from utils.dataset import get_dataloaders


def compute_per_class_ece(predictions, confidences, labels, num_bins=10, class_idx=0):
    """
    Compute Expected Calibration Error for a specific class.
    
    Args:
        predictions: Array of predicted class indices
        confidences: Array of confidence scores
        labels: Array of true class indices  
        num_bins: Number of bins for calibration
        class_idx: Class index to compute ECE for
        
    Returns:
        ece: Expected Calibration Error for the specified class
        bin_data: Dictionary with binning information for plotting
    """
    # Filter to only examples of this class
    class_mask = (labels == class_idx)
    if class_mask.sum() == 0:
        return 0.0, None
        
    class_preds = predictions[class_mask]
    class_confs = confidences[class_mask]
    class_labels = labels[class_mask]
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_data = {'confidences': [], 'accuracies': [], 'counts': []}
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find examples in this bin
        in_bin = (class_confs > bin_lower) & (class_confs <= bin_upper)
        prop_in_bin = in_bin.sum() / len(class_confs)
        
        if prop_in_bin > 0:
            accuracy_in_bin = (class_preds[in_bin] == class_labels[in_bin]).mean()
            avg_confidence_in_bin = class_confs[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_data['confidences'].append(avg_confidence_in_bin)
            bin_data['accuracies'].append(accuracy_in_bin)
            bin_data['counts'].append(in_bin.sum())
        else:
            bin_data['confidences'].append((bin_lower + bin_upper) / 2)
            bin_data['accuracies'].append(0)
            bin_data['counts'].append(0)
    
    return ece, bin_data


def compute_per_class_conformal_metrics(predictions, labels, conformal_sets, class_idx=0):
    """
    Compute conformal prediction metrics for a specific class.
    
    Args:
        predictions: Array of predicted class indices
        labels: Array of true labels
        conformal_sets: List of prediction sets (each a list of class indices)
        class_idx: Class index to analyze
        
    Returns:
        metrics: Dictionary with coverage, avg_set_size, singleton_rate
    """
    # Filter to examples of this class
    class_mask = (labels == class_idx)
    if class_mask.sum() == 0:
        return None
        
    class_labels = labels[class_mask]
    class_sets = [conformal_sets[i] for i in np.where(class_mask)[0]]
    
    # Compute coverage
    coverage = np.mean([label in pred_set for label, pred_set in zip(class_labels, class_sets)])
    
    # Compute average set size
    avg_set_size = np.mean([len(pred_set) for pred_set in class_sets])
    
    # Compute singleton rate (confident predictions)
    singleton_rate = np.mean([len(pred_set) == 1 for pred_set in class_sets])
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'singleton_rate': singleton_rate,
        'n_samples': class_mask.sum()
    }


def plot_class_conditional_reliability(bin_data_by_class, class_names, output_path):
    """
    Plot reliability diagrams for each class side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (class_name, bin_data) in enumerate(zip(class_names, bin_data_by_class)):
        ax = axes[idx]
        
        if bin_data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f'{class_name} - No Data')
            continue
            
        confidences = bin_data['confidences']
        accuracies = bin_data['accuracies']
        counts = bin_data['counts']
        
        # Plot bars
        bars = ax.bar(range(len(confidences)), accuracies, alpha=0.7, 
                      color='steelblue', edgecolor='black')
        
        # Overlay confidence line
        ax.plot(range(len(confidences)), confidences, 'r--', linewidth=2, 
                label='Confidence', marker='o')
        
        # Perfect calibration line
        ax.plot([0, len(confidences)-1], [confidences[0], confidences[-1]], 
                'g-', linewidth=1, alpha=0.5, label='Perfect Calibration')
        
        # Add sample counts
        for i, count in enumerate(counts):
            if count > 0:
                ax.text(i, 0.02, f'n={count}', ha='center', fontsize=8)
        
        ax.set_xlabel('Confidence Bin')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{class_name} Calibration')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved class-conditional reliability diagram to {output_path}")
    plt.close()


def analyze_class_conditional_calibration(args):
    """
    Main analysis function.
    """
    # Load results
    results_path = Path(args.results_dir) / 'test_results.json'
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    confidences = np.array([max(probs) for probs in results['probabilities']])
    
    # Class names
    class_names = ['Normal', 'Pneumonia']
    num_classes = 2
    
    # Compute per-class ECE
    print("\n" + "="*60)
    print("CLASS-CONDITIONAL CALIBRATION ANALYSIS")
    print("="*60)
    
    ece_by_class = []
    bin_data_by_class = []
    
    for class_idx, class_name in enumerate(class_names):
        ece, bin_data = compute_per_class_ece(predictions, confidences, labels, 
                                                num_bins=10, class_idx=class_idx)
        ece_by_class.append(ece)
        bin_data_by_class.append(bin_data)
        
        n_samples = (labels == class_idx).sum()
        print(f"\n{class_name} (n={n_samples}):")
        print(f"  ECE: {ece:.4f}")
    
    # Compute calibration gap
    calibration_gap = max(ece_by_class) - min(ece_by_class)
    print(f"\nCalibration Gap: {calibration_gap:.4f}")
    
    if calibration_gap > 0.05:
        print("  ⚠️  WARNING: Large calibration gap detected!")
        print("  This suggests different uncertainty quality across classes.")
    else:
        print("  ✓ Calibration gap is acceptable (<0.05)")
    
    # Plot reliability diagrams
    os.makedirs(args.output_dir, exist_ok=True)
    plot_class_conditional_reliability(
        bin_data_by_class, 
        class_names, 
        Path(args.output_dir) / 'class_conditional_reliability.png'
    )
    
    # If conformal results available, analyze those too
    if args.calibration_file and os.path.exists(args.calibration_file):
        print("\n" + "="*60)
        print("CLASS-CONDITIONAL CONFORMAL METRICS")
        print("="*60)
        
        with open(args.calibration_file, 'r') as f:
            conformal_data = json.load(f)
        
        conformal_sets = conformal_data['prediction_sets']
        
        for class_idx, class_name in enumerate(class_names):
            metrics = compute_per_class_conformal_metrics(
                predictions, labels, conformal_sets, class_idx
            )
            
            if metrics:
                print(f"\n{class_name} (n={metrics['n_samples']}):")
                print(f"  Coverage: {metrics['coverage']:.3f}")
                print(f"  Avg Set Size: {metrics['avg_set_size']:.3f}")
                print(f"  Singleton Rate: {metrics['singleton_rate']:.3f}")
    
    # Save summary
    summary = {
        'ece_by_class': {name: float(ece) for name, ece in zip(class_names, ece_by_class)},
        'calibration_gap': float(calibration_gap),
        'n_samples_by_class': {name: int((labels == idx).sum()) 
                               for idx, name in enumerate(class_names)}
    }
    
    summary_path = Path(args.output_dir) / 'class_conditional_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Class-conditional calibration analysis')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing test_results.json')
    parser.add_argument('--calibration_file', type=str, default=None,
                        help='Path to conformal calibration JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs/class_conditional',
                        help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    analyze_class_conditional_calibration(args)
