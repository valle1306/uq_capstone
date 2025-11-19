"""
Analyze Conformal Calibration Set Size
Verifies that calibration set (1,044 samples) is appropriate for coverage guarantees

Questions:
1. Is 1,044 calibration samples sufficient for stable coverage?
2. How does coverage vary with different calibration set sizes?
3. What is the optimal train/cal/test split?
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from conformal_prediction import (
    load_model,
    get_softmax_scores,
    calibrate_conformal_simple,
    predict_with_conformal_simple,
    evaluate_conformal
)
from data_utils_classification import get_classification_loaders


def analyze_calibration_set_size(model, all_cal_probs, all_cal_labels, test_probs, test_labels, alpha=0.1, n_trials=10):
    """
    Analyze how calibration set size affects coverage
    
    Args:
        model: Trained model
        all_cal_probs: All available calibration probabilities
        all_cal_labels: All available calibration labels
        test_probs: Test probabilities
        test_labels: Test labels
        alpha: Miscoverage rate
        n_trials: Number of random trials per size
    
    Returns:
        results: Dict with coverage statistics for each cal set size
    """
    n_total = len(all_cal_labels)
    
    # Test different calibration set sizes
    cal_sizes = [50, 100, 200, 400, 600, 800, 1000, n_total]
    
    results = {}
    
    print(f"\nAnalyzing calibration set sizes (alpha={alpha}, {n_trials} trials each)...")
    print(f"{'Cal Size':<12} {'Mean Cov':<12} {'Std Cov':<12} {'Mean Size':<12} {'Coverage Range':<20}")
    print("-" * 70)
    
    for cal_size in cal_sizes:
        if cal_size > n_total:
            continue
        
        coverages = []
        set_sizes = []
        
        for trial in range(n_trials):
            # Randomly sample calibration set
            indices = np.random.choice(n_total, size=cal_size, replace=False)
            cal_probs_sample = all_cal_probs[indices]
            cal_labels_sample = all_cal_labels[indices]
            
            # Calibrate
            tau = calibrate_conformal_simple(cal_probs_sample, cal_labels_sample, alpha)
            
            # Predict on test set
            pred_sets = predict_with_conformal_simple(test_probs, tau)
            
            # Evaluate
            eval_results = evaluate_conformal(pred_sets, test_labels)
            
            coverages.append(eval_results['coverage'])
            set_sizes.append(eval_results['avg_set_size'])
        
        mean_cov = np.mean(coverages)
        std_cov = np.std(coverages)
        mean_size = np.mean(set_sizes)
        cov_range = f"[{np.min(coverages):.3f}, {np.max(coverages):.3f}]"
        
        results[cal_size] = {
            'mean_coverage': mean_cov,
            'std_coverage': std_cov,
            'mean_set_size': mean_size,
            'min_coverage': np.min(coverages),
            'max_coverage': np.max(coverages),
            'all_coverages': coverages,
            'all_set_sizes': set_sizes
        }
        
        print(f"{cal_size:<12} {mean_cov:<12.4f} {std_cov:<12.4f} {mean_size:<12.2f} {cov_range:<20}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze Conformal Calibration Set Size')
    parser.add_argument('--dataset', type=str, default='chest_xray')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--n_trials', type=int, default=10, help='Trials per calibration size')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Conformal Calibration Set Size Analysis")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Target coverage: {100 * (1 - args.alpha):.1f}%")
    print(f"Trials per size: {args.n_trials}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Calibration samples: {len(cal_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    print(f"\nLoading model...")
    model = load_model(args.model_path, args.arch, num_classes, device)
    
    # Get predictions
    print("Computing predictions...")
    cal_probs, cal_labels = get_softmax_scores(model, cal_loader, device)
    test_probs, test_labels = get_softmax_scores(model, test_loader, device)
    
    # Analyze calibration set size
    results = analyze_calibration_set_size(
        model, cal_probs, cal_labels, test_probs, test_labels,
        alpha=args.alpha, n_trials=args.n_trials
    )
    
    # Save results
    output_file = os.path.join(args.output_dir, 'calibration_size_analysis.json')
    results_dict = {
        'model_path': args.model_path,
        'dataset': args.dataset,
        'total_cal_samples': len(cal_labels),
        'test_samples': len(test_labels),
        'alpha': args.alpha,
        'target_coverage': 1 - args.alpha,
        'n_trials': args.n_trials,
        'results': {str(k): v for k, v in results.items()}
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Generate plot
    print("\nGenerating visualization...")
    cal_sizes = sorted([int(k) for k in results.keys()])
    mean_covs = [results[s]['mean_coverage'] for s in cal_sizes]
    std_covs = [results[s]['std_coverage'] for s in cal_sizes]
    target_cov = 1 - args.alpha
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(cal_sizes, mean_covs, yerr=std_covs, marker='o', capsize=5, label='Empirical Coverage')
    plt.axhline(y=target_cov, color='r', linestyle='--', label=f'Target Coverage ({100*target_cov:.0f}%)')
    plt.xlabel('Calibration Set Size')
    plt.ylabel('Coverage')
    plt.title(f'Conformal Coverage vs Calibration Set Size\n(Target: {100*target_cov:.0f}%, {args.n_trials} trials per size)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(args.output_dir, 'calibration_size_plot.png')
    plt.savefig(plot_file, dpi=150)
    print(f"Plot saved to {plot_file}")
    
    # Analysis summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    current_size = len(cal_labels)
    current_results = results[current_size]
    
    print(f"\nCurrent calibration set size: {current_size}")
    print(f"Target coverage: {100 * target_cov:.1f}%")
    print(f"Empirical coverage: {100 * current_results['mean_coverage']:.2f}% ± {100 * current_results['std_coverage']:.2f}%")
    print(f"Coverage range: [{100 * current_results['min_coverage']:.2f}%, {100 * current_results['max_coverage']:.2f}%]")
    print(f"Average set size: {current_results['mean_set_size']:.2f}")
    
    # Check if sufficient
    lower_bound = current_results['mean_coverage'] - 2 * current_results['std_coverage']
    upper_bound = current_results['mean_coverage'] + 2 * current_results['std_coverage']
    
    print(f"\n95% Confidence interval: [{100 * lower_bound:.2f}%, {100 * upper_bound:.2f}%]")
    
    if lower_bound <= target_cov <= upper_bound:
        print(f"✓ Calibration set size is SUFFICIENT")
        print(f"  Target coverage is within 95% CI of empirical coverage")
    else:
        if current_results['mean_coverage'] < target_cov:
            print(f"⚠ Calibration set might be TOO SMALL")
            print(f"  Empirical coverage is below target")
        else:
            print(f"✓ Calibration set is ADEQUATE (slight overcover age)")
            print(f"  Empirical coverage exceeds target (conservative)")
    
    # Recommendation
    print(f"\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    # Find minimum size with stable coverage
    stable_size = None
    for size in cal_sizes:
        if size >= 400 and results[size]['std_coverage'] < 0.02:  # std < 2%
            stable_size = size
            break
    
    if stable_size and stable_size < current_size:
        print(f"Minimum stable size: {stable_size} samples")
        print(f"Current size ({current_size}) provides good coverage guarantees")
    elif current_results['std_coverage'] < 0.02:
        print(f"Current size ({current_size}) provides stable coverage (std < 2%)")
    else:
        print(f"Consider increasing calibration set for more stable coverage")
        print(f"Current std: {100 * current_results['std_coverage']:.2f}%")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
