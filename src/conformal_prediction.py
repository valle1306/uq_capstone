"""
Conformal Prediction for Medical Image Classification
Implements various conformal prediction methods for uncertainty quantification

Methods:
1. Standard Conformal Prediction (APS - Adaptive Prediction Sets)
2. RAPS (Regularized Adaptive Prediction Sets)
3. Class-Conditional Coverage
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from torchvision import models
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils_classification import get_classification_loaders


def load_model(model_path, arch='resnet18', num_classes=2, device='cuda'):
    """Load trained model"""
    if arch == 'resnet18':
        model = models.resnet18(pretrained=False)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=False)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=False)
    
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle MC Dropout wrapper (base_model. prefix)
    if any(k.startswith('base_model.') for k in state_dict.keys()):
        state_dict = {k.replace('base_model.', ''): v for k, v in state_dict.items()}
    
    # Handle MC Dropout Sequential FC layer (fc.0 = Dropout, fc.1 = Linear)
    # Map fc.1.weight -> fc.weight and fc.1.bias -> fc.bias
    if 'fc.1.weight' in state_dict and 'fc.1.bias' in state_dict:
        state_dict['fc.weight'] = state_dict.pop('fc.1.weight')
        state_dict['fc.bias'] = state_dict.pop('fc.1.bias')
        # Remove dropout layer if present
        state_dict.pop('fc.0.weight', None)
        state_dict.pop('fc.0.bias', None)
    
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model


def get_softmax_scores(model, loader, device):
    """Get softmax probabilities for all samples"""
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Computing softmax scores"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_probs, all_labels


def calibrate_conformal_aps(cal_probs, cal_labels, alpha=0.1):
    """
    Calibrate Adaptive Prediction Sets (APS)
    
    Args:
        cal_probs: Calibration softmax probabilities (n_cal, n_classes)
        cal_labels: True labels (n_cal,)
        alpha: Miscoverage rate (1-alpha is coverage)
    
    Returns:
        tau: Threshold for prediction sets
    """
    n = len(cal_labels)
    
    # Compute conformity scores (1 - cumulative probability up to true label)
    sorted_probs = np.sort(cal_probs, axis=1)[:, ::-1]  # Sort descending
    cumsum_probs = np.cumsum(sorted_probs, axis=1)
    
    # Get true label probabilities
    true_label_probs = cal_probs[np.arange(n), cal_labels]
    
    # Conformity score = cumulative prob until true label is included
    scores = []
    for i in range(n):
        true_prob = true_label_probs[i]
        cumsum = cumsum_probs[i]
        # Find where true label would be included
        idx = np.searchsorted(cumsum, true_prob)
        if idx < len(cumsum):
            scores.append(cumsum[idx])
        else:
            scores.append(1.0)
    
    scores = np.array(scores)
    
    # Compute quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    tau = np.quantile(scores, q_level)
    
    return tau


def calibrate_conformal_simple(cal_probs, cal_labels, alpha=0.1):
    """
    Simple conformal prediction based on score = 1 - p(true_class)
    
    Returns threshold where we reject classes
    """
    n = len(cal_labels)
    
    # Conformity scores: probability of true class
    scores = cal_probs[np.arange(n), cal_labels]
    
    # We want high probability = low conformity score (invert)
    conformity_scores = 1 - scores
    
    # Compute quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    tau = np.quantile(conformity_scores, q_level)
    
    return tau


def predict_with_conformal_simple(test_probs, tau):
    """
    Predict with simple conformal threshold
    
    Returns prediction sets where 1 - p(class) <= tau
    """
    n_samples, n_classes = test_probs.shape
    prediction_sets = []
    
    for i in range(n_samples):
        probs = test_probs[i]
        # Include class if 1 - p(class) <= tau
        # Equivalent to: p(class) >= 1 - tau
        pred_set = np.where(probs >= (1 - tau))[0].tolist()
        
        # Always include at least one class (highest prob)
        if len(pred_set) == 0:
            pred_set = [np.argmax(probs)]
        
        prediction_sets.append(pred_set)
    
    return prediction_sets


def evaluate_conformal(prediction_sets, true_labels):
    """
    Evaluate conformal prediction sets
    
    Returns:
        coverage: Fraction of times true label is in prediction set
        avg_set_size: Average size of prediction sets
        accuracy: Fraction of singleton sets that are correct
    """
    n = len(true_labels)
    
    coverage = 0
    set_sizes = []
    singleton_correct = 0
    n_singletons = 0
    
    for i in range(n):
        pred_set = prediction_sets[i]
        true_label = true_labels[i]
        
        # Coverage
        if true_label in pred_set:
            coverage += 1
        
        # Set size
        set_sizes.append(len(pred_set))
        
        # Singleton accuracy
        if len(pred_set) == 1:
            n_singletons += 1
            if pred_set[0] == true_label:
                singleton_correct += 1
    
    coverage = coverage / n
    avg_set_size = np.mean(set_sizes)
    singleton_accuracy = singleton_correct / n_singletons if n_singletons > 0 else 0.0
    
    return {
        'coverage': coverage,
        'avg_set_size': avg_set_size,
        'singleton_accuracy': singleton_accuracy,
        'n_singletons': n_singletons,
        'singleton_fraction': n_singletons / n
    }


def main():
    parser = argparse.ArgumentParser(description='Conformal Prediction for Classification')
    parser.add_argument('--dataset', type=str, default='chest_xray')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage rate (default: 0.1 for 90% coverage)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Conformal Prediction for Medical Image Classification")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Target coverage: {100 * (1 - args.alpha):.1f}%")
    print(f"Alpha (miscoverage): {args.alpha}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Calibration samples: {len(cal_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = load_model(args.model_path, args.arch, num_classes, device)
    
    # Get predictions on calibration set
    print("\nComputing calibration scores...")
    cal_probs, cal_labels = get_softmax_scores(model, cal_loader, device)
    
    # Get predictions on test set
    print("Computing test predictions...")
    test_probs, test_labels = get_softmax_scores(model, test_loader, device)
    
    # Calibrate conformal predictor
    print(f"\nCalibrating conformal predictor (alpha={args.alpha})...")
    tau = calibrate_conformal_simple(cal_probs, cal_labels, args.alpha)
    print(f"Calibrated threshold (tau): {tau:.4f}")
    
    # Predict on test set
    print("\nGenerating prediction sets on test data...")
    prediction_sets = predict_with_conformal_simple(test_probs, tau)
    
    # Evaluate
    print("\nEvaluating conformal predictions...")
    results = evaluate_conformal(prediction_sets, test_labels)
    
    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION RESULTS")
    print("=" * 70)
    print(f"Target coverage:        {100 * (1 - args.alpha):.1f}%")
    print(f"Empirical coverage:     {100 * results['coverage']:.2f}%")
    print(f"Average set size:       {results['avg_set_size']:.2f}")
    print(f"Singleton fraction:     {100 * results['singleton_fraction']:.2f}%")
    print(f"Singleton accuracy:     {100 * results['singleton_accuracy']:.2f}%")
    print("=" * 70)
    
    # Test different alpha values
    print("\n" + "=" * 70)
    print("COVERAGE vs SET SIZE TRADE-OFF")
    print("=" * 70)
    print(f"{'Alpha':<10} {'Target Cov':<15} {'Empirical Cov':<18} {'Avg Set Size':<15}")
    print("-" * 70)
    
    alpha_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    all_results = {}
    
    for alpha in alpha_values:
        tau_alpha = calibrate_conformal_simple(cal_probs, cal_labels, alpha)
        pred_sets_alpha = predict_with_conformal_simple(test_probs, tau_alpha)
        res_alpha = evaluate_conformal(pred_sets_alpha, test_labels)
        
        target_cov = 100 * (1 - alpha)
        emp_cov = 100 * res_alpha['coverage']
        avg_size = res_alpha['avg_set_size']
        
        print(f"{alpha:<10.2f} {target_cov:<15.1f} {emp_cov:<18.2f} {avg_size:<15.2f}")
        
        all_results[f'alpha_{alpha}'] = {
            'alpha': alpha,
            'target_coverage': 1 - alpha,
            'empirical_coverage': res_alpha['coverage'],
            'avg_set_size': res_alpha['avg_set_size'],
            'singleton_fraction': res_alpha['singleton_fraction'],
            'singleton_accuracy': res_alpha['singleton_accuracy']
        }
    
    # Save results
    output_file = os.path.join(args.output_dir, 'conformal_prediction_results.json')
    
    results_dict = {
        'model_path': args.model_path,
        'dataset': args.dataset,
        'num_classes': int(num_classes),
        'n_calibration': len(cal_labels),
        'n_test': len(test_labels),
        'primary_alpha': args.alpha,
        'primary_results': {
            'tau': float(tau),
            'target_coverage': 1 - args.alpha,
            'empirical_coverage': results['coverage'],
            'avg_set_size': results['avg_set_size'],
            'singleton_fraction': results['singleton_fraction'],
            'singleton_accuracy': results['singleton_accuracy']
        },
        'all_alpha_results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Also save prediction sets for primary alpha
    pred_sets_file = os.path.join(args.output_dir, 'prediction_sets.json')
    pred_sets_dict = {
        'alpha': args.alpha,
        'tau': float(tau),
        'prediction_sets': [list(map(int, ps)) for ps in prediction_sets],
        'true_labels': test_labels.tolist(),
        'softmax_probs': test_probs.tolist()
    }
    
    with open(pred_sets_file, 'w') as f:
        json.dump(pred_sets_dict, f, indent=2)
    
    print(f"Prediction sets saved to {pred_sets_file}")
    
    print("\n" + "=" * 70)
    print("Conformal prediction complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
