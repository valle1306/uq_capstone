"""
Comprehensive Evaluation of UQ Methods for Medical Image Classification

Evaluates all uncertainty quantification methods:
1. Baseline (no UQ)
2. MC Dropout
3. Deep Ensemble  
4. Conformal Risk Control (multiple loss functions)

Computes metrics:
- Accuracy
- Expected Calibration Error (ECE)
- Brier Score
- Coverage (for conformal methods)
- Set size (for conformal methods)
- Uncertainty quality metrics
"""

import os
import argparse
import json
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from data_utils_classification import get_classification_loaders
from conformal_risk_control import (
    ConformalRiskControl,
    false_negative_rate_loss,
    expected_set_size_loss,
    composite_loss,
    f1_loss,
    precision_loss
)
from swag import load_swag_model


def load_model(model_path, num_classes, arch='resnet18', device='cuda', dropout_rate=None):
    """Load a trained model"""
    if arch == 'resnet18':
        base_model = models.resnet18(pretrained=False)
    elif arch == 'resnet34':
        base_model = models.resnet34(pretrained=False)
    elif arch == 'resnet50':
        base_model = models.resnet50(pretrained=False)
    
    if dropout_rate is not None:
        # MC Dropout model
        from train_classifier_mc_dropout import ResNetWithDropout
        model = ResNetWithDropout(base_model, num_classes, dropout_rate)
    else:
        # Standard model
        num_features = base_model.fc.in_features
        base_model.fc = nn.Linear(num_features, num_classes)
        model = base_model
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def compute_ece(probs, labels, n_bins=15):
    """
    Expected Calibration Error
    
    Measures calibration: how well predicted probabilities match actual frequencies
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    
    ece = 0.0
    for i in range(n_bins):
        bin_lower = i / n_bins
        bin_upper = (i + 1) / n_bins
        
        bin_mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if bin_mask.sum() > 0:
            bin_acc = accuracies[bin_mask].float().mean()
            bin_conf = confidences[bin_mask].mean()
            bin_size = bin_mask.sum().item() / len(labels)
            ece += bin_size * abs(bin_acc - bin_conf)
    
    return ece.item()


def compute_brier_score(probs, labels):
    """
    Brier Score
    
    Measures accuracy of probabilistic predictions
    Lower is better
    """
    num_classes = probs.shape[1]
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    brier = ((probs - one_hot) ** 2).sum(dim=1).mean()
    return brier.item()


def evaluate_baseline(model, test_loader, device):
    """Evaluate baseline model (no UQ)"""
    print("\n" + "=" * 70)
    print("Evaluating Baseline")
    print("=" * 70)
    
    model.eval()
    
    all_probs = []
    all_preds = []
    all_labels = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Baseline"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_probs.append(probs.cpu())
            all_preds.append(predicted.cpu())
            all_labels.append(labels)
    
    # Aggregate
    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    # Compute metrics
    accuracy = 100. * correct / total
    ece = compute_ece(all_probs, all_labels)
    brier = compute_brier_score(all_probs, all_labels)
    
    # Max confidence as "uncertainty" (higher = more certain)
    max_probs = all_probs.max(dim=1)[0].numpy()
    uncertainties = 1 - max_probs  # Convert to uncertainty
    
    results = {
        'method': 'Baseline',
        'accuracy': accuracy,
        'ece': ece,
        'brier_score': brier,
        'mean_uncertainty': float(np.mean(uncertainties)),
        'std_uncertainty': float(np.std(uncertainties))
    }
    
    print("\nBaseline Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    
    return results, uncertainties, (all_labels == all_preds).numpy()


def evaluate_mc_dropout(model, test_loader, device, n_samples=20):
    """Evaluate MC Dropout"""
    print("\n" + "=" * 70)
    print(f"Evaluating MC Dropout (T={n_samples})")
    print("=" * 70)
    
    model.eval()
    model.enable_dropout()  # Keep dropout active
    
    all_probs_mean = []
    all_uncertainties = []
    all_labels = []
    
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(test_loader, desc="MC Dropout"):
        inputs = inputs.to(device)
        
        # MC sampling
        probs_samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probs_samples.append(probs.cpu())
        
        # Average predictions
        probs_samples = torch.stack(probs_samples)  # [T, B, C]
        probs_mean = probs_samples.mean(dim=0)  # [B, C]
        
        # Uncertainty: variance of predictions
        probs_var = probs_samples.var(dim=0)  # [B, C]
        uncertainty = probs_var.mean(dim=1)  # [B]
        
        # Accuracy
        _, predicted = probs_mean.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_probs_mean.append(probs_mean)
        all_uncertainties.append(uncertainty)
        all_labels.append(labels)
    
    # Aggregate
    all_probs_mean = torch.cat(all_probs_mean)
    all_uncertainties = torch.cat(all_uncertainties).numpy()
    all_labels = torch.cat(all_labels)
    all_preds = all_probs_mean.max(1)[1]
    
    # Compute metrics
    accuracy = 100. * correct / total
    ece = compute_ece(all_probs_mean, all_labels)
    brier = compute_brier_score(all_probs_mean, all_labels)
    
    results = {
        'method': 'MC Dropout',
        'n_samples': n_samples,
        'accuracy': accuracy,
        'ece': ece,
        'brier_score': brier,
        'mean_uncertainty': float(np.mean(all_uncertainties)),
        'std_uncertainty': float(np.std(all_uncertainties))
    }
    
    print("\nMC Dropout Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Mean Uncertainty: {np.mean(all_uncertainties):.4f}")
    
    return results, all_uncertainties, (all_labels == all_preds).numpy()


def evaluate_ensemble(model_paths, num_classes, test_loader, device, arch='resnet18'):
    """Evaluate Deep Ensemble"""
    print("\n" + "=" * 70)
    print(f"Evaluating Deep Ensemble (M={len(model_paths)})")
    print("=" * 70)
    
    # Load all ensemble members
    models = []
    for path in model_paths:
        model = load_model(path, num_classes, arch, device)
        models.append(model)
    
    all_probs_mean = []
    all_uncertainties = []
    all_labels = []
    
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(test_loader, desc="Ensemble"):
        inputs = inputs.to(device)
        
        # Get predictions from all models
        probs_samples = []
        for model in models:
            with torch.no_grad():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probs_samples.append(probs.cpu())
        
        # Average predictions
        probs_samples = torch.stack(probs_samples)  # [M, B, C]
        probs_mean = probs_samples.mean(dim=0)  # [B, C]
        
        # Uncertainty: variance across ensemble
        probs_var = probs_samples.var(dim=0)  # [B, C]
        uncertainty = probs_var.mean(dim=1)  # [B]
        
        # Accuracy
        _, predicted = probs_mean.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_probs_mean.append(probs_mean)
        all_uncertainties.append(uncertainty)
        all_labels.append(labels)
    
    # Aggregate
    all_probs_mean = torch.cat(all_probs_mean)
    all_uncertainties = torch.cat(all_uncertainties).numpy()
    all_labels = torch.cat(all_labels)
    all_preds = all_probs_mean.max(1)[1]
    
    # Compute metrics
    accuracy = 100. * correct / total
    ece = compute_ece(all_probs_mean, all_labels)
    brier = compute_brier_score(all_probs_mean, all_labels)
    
    results = {
        'method': 'Deep Ensemble',
        'n_models': len(models),
        'accuracy': accuracy,
        'ece': ece,
        'brier_score': brier,
        'mean_uncertainty': float(np.mean(all_uncertainties)),
        'std_uncertainty': float(np.std(all_uncertainties))
    }
    
    print("\nEnsemble Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Mean Uncertainty: {np.mean(all_uncertainties):.4f}")
    
    return results, all_uncertainties, (all_labels == all_preds).numpy()


def evaluate_swag(swag_checkpoint_path, base_arch, num_classes, test_loader, device, n_samples=30, scale=0.5):
    """Evaluate SWAG by sampling from its posterior and aggregating predictions"""
    print("\n" + "=" * 70)
    print(f"Evaluating SWAG (T={n_samples})")
    print("=" * 70)

    # Build base model architecture and make sure final layer matches num_classes
    if base_arch == 'resnet18':
        base_model = models.resnet18(pretrained=False)
    elif base_arch == 'resnet34':
        base_model = models.resnet34(pretrained=False)
    elif base_arch == 'resnet50':
        base_model = models.resnet50(pretrained=False)

    # Ensure final classification layer matches the number of classes used during training
    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, num_classes)

    # Wrap with SWAG loader
    swag = load_swag_model(swag_checkpoint_path, base_model)

    all_probs_mean = []
    all_uncertainties = []
    all_labels = []

    correct = 0
    total = 0

    for inputs, labels in tqdm(test_loader, desc='SWAG'):
        inputs = inputs.to(device)

        # Sample predictions
        preds = []
        for _ in range(n_samples):
            sampled = swag.sample(scale=scale)
            sampled = sampled.to(device)
            sampled.eval()
            with torch.no_grad():
                outputs = sampled(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds.append(probs.cpu())

        preds = torch.stack(preds)  # [T, B, C]
        probs_mean = preds.mean(dim=0)
        probs_var = preds.var(dim=0)
        uncertainty = probs_var.mean(dim=1)

        _, predicted = probs_mean.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_probs_mean.append(probs_mean)
        all_uncertainties.append(uncertainty)
        all_labels.append(labels)

    # Aggregate
    all_probs_mean = torch.cat(all_probs_mean)
    all_uncertainties = torch.cat(all_uncertainties).numpy()
    all_labels = torch.cat(all_labels)
    all_preds = all_probs_mean.max(1)[1]

    accuracy = 100. * correct / total
    ece = compute_ece(all_probs_mean, all_labels)
    brier = compute_brier_score(all_probs_mean, all_labels)

    results = {
        'method': 'SWAG',
        'n_samples': n_samples,
        'accuracy': accuracy,
        'ece': ece,
        'brier_score': brier,
        'mean_uncertainty': float(all_uncertainties.mean()),
        'std_uncertainty': float(all_uncertainties.std())
    }

    print("\nSWAG Results:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  ECE: {ece:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Mean Uncertainty: {np.mean(all_uncertainties):.4f}")

    return results, all_uncertainties, (all_labels == all_preds).numpy()


def evaluate_conformal_risk_control(model, cal_loader, test_loader, device):
    """Evaluate Conformal Risk Control with different loss functions"""
    print("\n" + "=" * 70)
    print("Evaluating Conformal Risk Control")
    print("=" * 70)
    
    results_list = []
    
    # Test different loss functions
    loss_configs = [
        ('FNR Control (alpha=0.05)', false_negative_rate_loss, 0.05),
        ('FNR Control (alpha=0.10)', false_negative_rate_loss, 0.10),
        ('Set Size Control (alpha=2.0)', expected_set_size_loss, 2.0),
        ('Composite (FNR+Size)', lambda y, s, p: composite_loss(y, s, p, 0.5, 0.5), 0.15),
        ('F1 Score Control', f1_loss, 0.10),
    ]
    
    for name, loss_fn, alpha in loss_configs:
        print(f"\n{'-' * 70}")
        print(f"Testing: {name}")
        print(f"{'-' * 70}")
        
        # Create CRC instance
        crc = ConformalRiskControl(
            loss_fn=loss_fn,
            alpha=alpha,
            delta=0.1
        )
        
        # Calibrate
        crc.calibrate(model, cal_loader, device)
        
        # Evaluate
        metrics = crc.evaluate_risk(model, test_loader, device)
        
        results = {
            'method': f'CRC - {name}',
            'target_risk': alpha,
            'empirical_risk': metrics['empirical_risk'],
            'coverage': metrics['coverage'],
            'avg_set_size': metrics['avg_set_size'],
            'std_set_size': metrics['std_set_size'],
            'threshold': metrics['threshold']
        }
        
        results_list.append(results)
        
        # Print CRC results for this loss configuration
        print(f"\n{name} Results:")
        print(f"  Target Risk: {alpha:.4f}")
        print(f"  Empirical Risk: {metrics['empirical_risk']:.4f}")
        print(f"  Coverage: {metrics['coverage']:.4f}")
        print(f"  Avg Set Size: {metrics['avg_set_size']:.2f}")
        print(f"  Threshold: {metrics['threshold']:.4f}")
    
    return results_list


def main():
    parser = argparse.ArgumentParser(description='Evaluate All UQ Methods')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='chest_xray',
                        choices=['chest_xray', 'oct_retinal', 'brain_tumor'])
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model arguments
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--baseline_path', type=str, required=True,
                        help='Path to baseline model checkpoint')
    parser.add_argument('--mc_dropout_path', type=str, default=None,
                        help='Path to MC Dropout model checkpoint')
    parser.add_argument('--ensemble_dir', type=str, default=None,
                        help='Directory containing ensemble member checkpoints')
    parser.add_argument('--n_ensemble', type=int, default=5,
                        help='Number of ensemble members')
    
    # UQ arguments
    parser.add_argument('--mc_samples', type=int, default=20,
                        help='Number of MC Dropout samples')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                        help='Dropout rate for MC Dropout')
    parser.add_argument('--swag_path', type=str, default=None,
                        help='Path to SWAG checkpoint (swag_model.pth)')
    parser.add_argument('--swag_samples', type=int, default=30,
                        help='Number of SWAG posterior samples to draw for prediction')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='runs/classification/evaluation')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Comprehensive UQ Evaluation")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Data loaded (Test samples: {len(test_loader.dataset)})")
    
    # Evaluate all methods
    all_results = []
    
    # 1. Baseline
    baseline_model = load_model(args.baseline_path, num_classes, args.arch, args.device)
    baseline_results, baseline_unc, baseline_correct = evaluate_baseline(
        baseline_model, test_loader, args.device
    )
    all_results.append(baseline_results)
    
    # 2. MC Dropout
    if args.mc_dropout_path:
        mc_model = load_model(
            args.mc_dropout_path, num_classes, args.arch, args.device, args.dropout_rate
        )
        mc_results, mc_unc, mc_correct = evaluate_mc_dropout(
            mc_model, test_loader, args.device, args.mc_samples
        )
        all_results.append(mc_results)
    
    # 3. Deep Ensemble
    if args.ensemble_dir:
        ensemble_dir = Path(args.ensemble_dir)
        model_paths = [
            ensemble_dir / f'member_{i}' / 'best_model.pth'
            for i in range(args.n_ensemble)
        ]
        # Verify all paths exist
        model_paths = [p for p in model_paths if p.exists()]
        
        if len(model_paths) > 0:
            ensemble_results, ensemble_unc, ensemble_correct = evaluate_ensemble(
                model_paths, num_classes, test_loader, args.device, args.arch
            )
            all_results.append(ensemble_results)
    
    # 4. SWAG
    if args.swag_path:
        swag_results, swag_unc, swag_correct = evaluate_swag(
            args.swag_path, args.arch, num_classes, test_loader, args.device, args.swag_samples
        )
        all_results.append(swag_results)

    # 4. Conformal Risk Control
    crc_results = evaluate_conformal_risk_control(
        baseline_model, cal_loader, test_loader, args.device
    )
    all_results.extend(crc_results)
    
    # Save all results
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"Results saved to: {output_dir / 'all_results.json'}")
    
    # Print summary table
    print("\nSummary Table:")
    print("=" * 70)
    print(f"{'Method':<30} {'Accuracy':<12} {'ECE':<12}")
    print("-" * 70)
    for result in all_results:
        if 'accuracy' in result:
            print(f"{result['method']:<30} {result['accuracy']:>10.2f}% {result['ece']:>10.4f}")
        elif 'coverage' in result:
            print(f"{result['method']:<30} {'N/A':>12} {'N/A':>12}")
    print("=" * 70)


if __name__ == '__main__':
    main()
