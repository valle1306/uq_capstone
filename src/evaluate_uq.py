"""
Evaluate and compare all uncertainty quantification methods
Computes metrics for: Baseline, Temperature Scaling, MC Dropout, Deep Ensemble, Conformal Prediction, SWAG
"""
import os
import argparse
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model_utils import UNet
from src.data_utils import BraTSSegmentationDataset
from src.uq_methods import (
    TemperatureScaling, MCDropoutSegmentation, DeepEnsemble, 
    ConformalPrediction, compute_uncertainty_metrics, 
    compute_calibration_metrics, compute_dice
)
from src.swag import SWAG, load_swag_model


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate UQ methods')
    p.add_argument('--data_root', required=True)
    p.add_argument('--baseline_model', required=True, help='Path to baseline model')
    p.add_argument('--mc_dropout_model', required=True, help='Path to MC dropout model')
    p.add_argument('--ensemble_dir', required=True, help='Directory with ensemble models')
    p.add_argument('--swag_model', default=None, help='Path to SWAG model checkpoint')
    p.add_argument('--num_ensemble', type=int, default=5, help='Number of ensemble members')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--in_ch', type=int, default=1)
    p.add_argument('--save_dir', default='runs/evaluation', help='Directory to save results')
    p.add_argument('--mc_samples', type=int, default=20, help='Number of MC dropout samples')
    p.add_argument('--swag_samples', type=int, default=30, help='Number of SWAG samples')
    p.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for MC Dropout')
    return p.parse_args()


def load_model(model_path, in_ch, device, dropout_p=0.0):
    """Load a trained model."""
    model = UNet(in_channels=in_ch, num_classes=1, dropout_rate=dropout_p).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def evaluate_baseline(model, test_loader, device):
    """Evaluate baseline model (no uncertainty)."""
    print("\n" + "="*60)
    print("EVALUATING BASELINE")
    print("="*60)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Baseline"):
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            all_targets.append(y)
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    # Compute metrics
    pred_binary = (preds > 0.5).float()
    dice = compute_dice(pred_binary, targets)
    
    # Calibration
    calib_metrics = compute_calibration_metrics(preds, targets)
    
    results = {
        'method': 'Baseline',
        'dice': dice,
        'ece': calib_metrics['ece'],
        'has_uncertainty': False
    }
    
    print(f"Dice: {dice:.4f}")
    print(f"ECE: {calib_metrics['ece']:.4f}")
    
    return results, preds, targets


def evaluate_temperature_scaling(model, val_loader, test_loader, device):
    """Evaluate with temperature scaling."""
    print("\n" + "="*60)
    print("EVALUATING TEMPERATURE SCALING")
    print("="*60)
    
    # Fit temperature
    temp_scaler = TemperatureScaling().to(device)
    temp_scaler.fit(model, val_loader, device, max_iters=50)
    
    # Evaluate on test set
    all_preds = []
    all_targets = []
    
    model.eval()
    temp_scaler.eval()
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Temp Scaling"):
            x = x.to(device)
            logits = model(x)
            scaled_logits = temp_scaler(logits)
            probs = torch.sigmoid(scaled_logits)
            all_preds.append(probs.cpu())
            all_targets.append(y)
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    pred_binary = (preds > 0.5).float()
    dice = compute_dice(pred_binary, targets)
    calib_metrics = compute_calibration_metrics(preds, targets)
    
    results = {
        'method': 'Temperature Scaling',
        'dice': dice,
        'ece': calib_metrics['ece'],
        'temperature': temp_scaler.temperature.item(),
        'has_uncertainty': False  # Only calibration, not epistemic uncertainty
    }
    
    print(f"Temperature: {temp_scaler.temperature.item():.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"ECE: {calib_metrics['ece']:.4f}")
    
    return results, preds, targets


def evaluate_mc_dropout(model, test_loader, device, mc_samples=20, dropout_rate=0.2):
    """Evaluate MC Dropout."""
    print("\n" + "="*60)
    print("EVALUATING MC DROPOUT")
    print("="*60)
    print(f"MC samples: {mc_samples}")
    
    mc_dropout = MCDropoutSegmentation(model, dropout_rate=dropout_rate, num_samples=mc_samples)
    
    all_mean_preds = []
    all_std_preds = []
    all_targets = []
    
    for x, y in tqdm(test_loader, desc="MC Dropout"):
        mean_pred, std_pred, _ = mc_dropout.predict_with_uncertainty(x, device)
        all_mean_preds.append(mean_pred)
        all_std_preds.append(std_pred)
        all_targets.append(y)
    
    mean_preds = torch.cat(all_mean_preds)
    std_preds = torch.cat(all_std_preds)
    targets = torch.cat(all_targets)
    
    # Compute metrics
    uq_metrics = compute_uncertainty_metrics(mean_preds, std_preds, targets)
    calib_metrics = compute_calibration_metrics(mean_preds, targets)
    
    results = {
        'method': 'MC Dropout',
        'dice': uq_metrics['dice_score'],
        'ece': calib_metrics['ece'],
        'mean_uncertainty': uq_metrics['mean_uncertainty'],
        'uncertainty_on_errors': uq_metrics['uncertainty_on_errors'],
        'uncertainty_on_correct': uq_metrics['uncertainty_on_correct'],
        'has_uncertainty': True
    }
    
    print(f"Dice: {uq_metrics['dice_score']:.4f}")
    print(f"ECE: {calib_metrics['ece']:.4f}")
    print(f"Mean uncertainty: {uq_metrics['mean_uncertainty']:.4f}")
    print(f"Uncertainty on errors: {uq_metrics['uncertainty_on_errors']:.4f}")
    
    return results, mean_preds, std_preds, targets


def evaluate_ensemble(ensemble_dir, num_ensemble, test_loader, device, in_ch):
    """Evaluate Deep Ensemble."""
    print("\n" + "="*60)
    print("EVALUATING DEEP ENSEMBLE")
    print("="*60)
    print(f"Ensemble size: {num_ensemble}")
    
    # Load all ensemble members
    models = []
    for i in range(num_ensemble):
        model_path = os.path.join(ensemble_dir, f'member_{i}', 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found!")
            continue
        model = load_model(model_path, in_ch, device)
        models.append(model)
    
    print(f"Loaded {len(models)} ensemble members")
    
    if len(models) == 0:
        raise ValueError("No ensemble models found!")
    
    ensemble = DeepEnsemble(models)
    
    all_mean_preds = []
    all_std_preds = []
    all_targets = []
    
    for x, y in tqdm(test_loader, desc="Ensemble"):
        mean_pred, std_pred, _ = ensemble.predict_with_uncertainty(x, device)
        all_mean_preds.append(mean_pred)
        all_std_preds.append(std_pred)
        all_targets.append(y)
    
    mean_preds = torch.cat(all_mean_preds)
    std_preds = torch.cat(all_std_preds)
    targets = torch.cat(all_targets)
    
    # Compute metrics
    uq_metrics = compute_uncertainty_metrics(mean_preds, std_preds, targets)
    calib_metrics = compute_calibration_metrics(mean_preds, targets)
    
    results = {
        'method': 'Deep Ensemble',
        'ensemble_size': len(models),
        'dice': uq_metrics['dice_score'],
        'ece': calib_metrics['ece'],
        'mean_uncertainty': uq_metrics['mean_uncertainty'],
        'uncertainty_on_errors': uq_metrics['uncertainty_on_errors'],
        'uncertainty_on_correct': uq_metrics['uncertainty_on_correct'],
        'has_uncertainty': True
    }
    
    print(f"Dice: {uq_metrics['dice_score']:.4f}")
    print(f"ECE: {calib_metrics['ece']:.4f}")
    print(f"Mean uncertainty: {uq_metrics['mean_uncertainty']:.4f}")
    print(f"Uncertainty on errors: {uq_metrics['uncertainty_on_errors']:.4f}")
    
    return results, mean_preds, std_preds, targets


def evaluate_swag(swag_model, test_loader, device, n_samples=30):
    """Evaluate SWAG (Stochastic Weight Averaging-Gaussian)."""
    print("\n" + "="*60)
    print("EVALUATING SWAG")
    print("="*60)
    print(f"Sampling {n_samples} models from posterior...")
    
    all_preds = []
    all_stds = []
    all_targets = []
    
    for x, y in tqdm(test_loader, desc="SWAG"):
        x = x.to(device)
        mean_pred, uncertainty = swag_model.predict_with_uncertainty(x, n_samples=n_samples, scale=0.5)
        all_preds.append(mean_pred.cpu())
        all_stds.append(uncertainty.cpu())
        all_targets.append(y)
    
    mean_preds = torch.cat(all_preds)
    std_preds = torch.cat(all_stds)
    targets = torch.cat(all_targets)
    
    # Compute metrics
    pred_binary = (mean_preds > 0.5).float()
    
    # UQ metrics
    uq_metrics = compute_uncertainty_metrics(pred_binary, targets, std_preds)
    
    # Calibration metrics
    calib_metrics = compute_calibration_metrics(mean_preds.squeeze(), targets.squeeze())
    
    results = {
        'method': 'SWAG',
        'dice': uq_metrics['dice_score'],
        'ece': calib_metrics['ece'],
        'mean_uncertainty': uq_metrics['mean_uncertainty'],
        'uncertainty_on_errors': uq_metrics['uncertainty_on_errors'],
        'uncertainty_on_correct': uq_metrics['uncertainty_on_correct'],
        'n_samples': n_samples,
        'has_uncertainty': True
    }
    
    print(f"Dice: {uq_metrics['dice_score']:.4f}")
    print(f"ECE: {calib_metrics['ece']:.4f}")
    print(f"Mean uncertainty: {uq_metrics['mean_uncertainty']:.4f}")
    print(f"Uncertainty on errors: {uq_metrics['uncertainty_on_errors']:.4f}")
    
    return results, mean_preds, std_preds, targets


def evaluate_conformal(model, val_loader, test_loader, device, alpha=0.1):
    """Evaluate Conformal Prediction."""
    print("\n" + "="*60)
    print("EVALUATING CONFORMAL PREDICTION")
    print("="*60)
    print(f"Target coverage: {1-alpha:.1%}")
    
    # Calibrate
    conformal = ConformalPrediction(alpha=alpha)
    conformal.calibrate(model, val_loader, device)
    
    # Evaluate coverage on test set
    all_pred_sets = []
    all_targets = []
    
    for x, y in tqdm(test_loader, desc="Conformal"):
        pred_sets, _ = conformal.predict_sets(model, x, device)
        all_pred_sets.append(pred_sets)
        all_targets.append(y)
    
    pred_sets = torch.cat(all_pred_sets)
    targets = torch.cat(all_targets)
    
    # Compute coverage
    coverage = (pred_sets == targets).float().mean().item()
    
    # Compute prediction set size
    set_size = pred_sets.sum() / pred_sets.numel()
    
    results = {
        'method': 'Conformal Prediction',
        'target_coverage': 1 - alpha,
        'actual_coverage': coverage,
        'threshold': conformal.threshold,
        'avg_set_size': set_size,
        'has_uncertainty': False  # Provides sets, not uncertainty estimates
    }
    
    print(f"Threshold: {conformal.threshold:.4f}")
    print(f"Target coverage: {1-alpha:.1%}")
    print(f"Actual coverage: {coverage:.1%}")
    print(f"Avg set size: {set_size:.2%}")
    
    return results


def create_comparison_plots(all_results, save_dir):
    """Create comparison plots for all methods."""
    print("\nCreating comparison plots...")
    
    # Extract metrics
    methods = [r['method'] for r in all_results if 'dice' in r]
    dices = [r['dice'] for r in all_results if 'dice' in r]
    eces = [r['ece'] for r in all_results if 'ece' in r]
    
    # Plot Dice scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(methods, dices, color='skyblue', edgecolor='navy')
    ax1.set_ylabel('Dice Score')
    ax1.set_title('Segmentation Performance')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    ax2.bar(methods, eces, color='salmon', edgecolor='darkred')
    ax2.set_ylabel('Expected Calibration Error')
    ax2.set_title('Calibration Performance')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison.png'), dpi=150)
    print(f"Saved: {os.path.join(save_dir, 'comparison.png')}")


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    
    print("="*60)
    print("UNCERTAINTY QUANTIFICATION EVALUATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Save dir: {args.save_dir}")
    
    # Load data
    val_ds = BraTSSegmentationDataset(
        npz_dir=os.path.join(args.data_root, 'val'),
        augment=False
    )
    test_ds = BraTSSegmentationDataset(
        npz_dir=os.path.join(args.data_root, 'test'),
        augment=False
    )
    
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    all_results = []
    
    # Load baseline model
    baseline_model = load_model(args.baseline_model, args.in_ch, device)
    
    # 1. Baseline
    results, _, _ = evaluate_baseline(baseline_model, test_loader, device)
    all_results.append(results)
    
    # 2. Temperature Scaling
    results, _, _ = evaluate_temperature_scaling(baseline_model, val_loader, test_loader, device)
    all_results.append(results)
    
    # 3. MC Dropout
    mc_model = load_model(args.mc_dropout_model, args.in_ch, device, dropout_p=args.dropout_rate)
    results, _, _, _ = evaluate_mc_dropout(mc_model, test_loader, device, args.mc_samples, args.dropout_rate)
    all_results.append(results)
    
    # 4. Deep Ensemble
    results, _, _, _ = evaluate_ensemble(args.ensemble_dir, args.num_ensemble, test_loader, device, args.in_ch)
    all_results.append(results)
    
    # 5. SWAG (if available)
    if args.swag_model and os.path.exists(args.swag_model):
        print("\n" + "="*60)
        print("Loading SWAG model...")
        print("="*60)
        base_model_for_swag = UNet(in_channels=args.in_ch, num_classes=1, dropout_rate=0.0)
        swag_model = load_swag_model(args.swag_model, base_model_for_swag)
        swag_model.to(device)
        results, _, _, _ = evaluate_swag(swag_model, test_loader, device, args.swag_samples)
        all_results.append(results)
    else:
        print("\n⚠️  SWAG model not provided or not found. Skipping SWAG evaluation.")
    
    # 6. Conformal Prediction
    results = evaluate_conformal(baseline_model, val_loader, test_loader, device)
    all_results.append(results)
    
    # Save results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['method']}:")
        for key, value in result.items():
            if key != 'method':
                print(f"  {key}: {value}")
    
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create plots
    create_comparison_plots(all_results, args.save_dir)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {args.save_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
