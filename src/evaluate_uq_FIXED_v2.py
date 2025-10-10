"""
Evaluate and compare all uncertainty quantification methods
FIXED VERSION v2 - Includes SWAG evaluation
"""
import os
import sys
import argparse
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import BraTSSegmentationDataset
from src.model_utils import UNet, DiceLoss, DiceBCELoss
from src.swag import SWAG


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate UQ methods')
    p.add_argument('--data_dir', default='/scratch/hpl14/uq_capstone/data/brats')
    p.add_argument('--baseline_model', default='runs/baseline/best_model.pth')
    p.add_argument('--mc_dropout_model', default='runs/mc_dropout/best_model.pth')
    p.add_argument('--ensemble_dir', default='runs/ensemble')
    p.add_argument('--swag_model', default='runs/swag/swag_model.pth')
    p.add_argument('--num_ensemble', type=int, default=5)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='cuda')
    p.add_argument('--in_ch', type=int, default=1)
    p.add_argument('--save_dir', default='runs/evaluation')
    p.add_argument('--mc_samples', type=int, default=20)
    p.add_argument('--swag_samples', type=int, default=30)
    p.add_argument('--dropout_rate', type=float, default=0.2)
    return p.parse_args()


def compute_dice(pred, target, smooth=1e-6):
    """Compute Dice coefficient"""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def compute_ece(probs, targets, n_bins=15):
    """Compute Expected Calibration Error"""
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (targets[in_bin] == (probs[in_bin] > 0.5)).float().mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece.item()


def load_model(model_path, in_ch, device, dropout_rate=0.0):
    """Load a trained model"""
    model = UNet(in_channels=in_ch, num_classes=1, dropout_rate=dropout_rate).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model
def load_swag_model(swag_path, in_ch, device):
    """Load SWAG model"""
    base_model = UNet(in_channels=in_ch, num_classes=1, dropout_rate=0.0)
    swag_model = SWAG(base_model, max_num_models=20, max_var=1.0)
    
    checkpoint = torch.load(swag_path, map_location=device)
    
    # Load SWAG statistics directly
    swag_model.n_models = checkpoint["n_models"]
    swag_model.mean = checkpoint["mean"].to(device)
    swag_model.sq_mean = checkpoint["sq_mean"].to(device)
    swag_model.cov_mat_sqrt = [d.to(device) for d in checkpoint["cov_mat_sqrt"]]
    swag_model.max_num_models = checkpoint["max_num_models"]
    
    swag_model.to(device)
    
    print(f"✓ Loaded SWAG model with {swag_model.n_models} collected snapshots")
    
    return swag_model

def evaluate_baseline(model, test_loader, device):
    """Evaluate baseline model (no uncertainty)"""
    print("\n" + "="*60)
    print("EVALUATING BASELINE")
    print("="*60)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Baseline"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())
    
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    
    pred_binary = (preds > 0.5).float()
    dice = compute_dice(pred_binary, targets)
    ece = compute_ece(preds.flatten(), targets.flatten())
    
    results = {
        'method': 'Baseline',
        'dice': float(dice),
        'ece': float(ece),
        'has_uncertainty': False
    }
    
    print(f"Dice: {dice:.4f}")
    print(f"ECE: {ece:.4f}")
    
    return results


def evaluate_mc_dropout(model, test_loader, device, n_samples=20):
    """Evaluate MC Dropout"""
    print("\n" + "="*60)
    print(f"EVALUATING MC DROPOUT (samples={n_samples})")
    print("="*60)
    
    model.train()  # Enable dropout
    
    all_preds_mean = []
    all_preds_std = []
    all_targets = []
    
    for batch in tqdm(test_loader, desc="MC Dropout"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Multiple forward passes with dropout
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = model(images)
                probs = torch.sigmoid(logits)
                samples.append(probs.cpu())
        
        samples = torch.stack(samples)
        pred_mean = samples.mean(dim=0)
        pred_std = samples.std(dim=0)
        
        all_preds_mean.append(pred_mean)
        all_preds_std.append(pred_std)
        all_targets.append(masks.cpu())
    
    preds_mean = torch.cat(all_preds_mean)
    preds_std = torch.cat(all_preds_std)
    targets = torch.cat(all_targets)
    
    pred_binary = (preds_mean > 0.5).float()
    dice = compute_dice(pred_binary, targets)
    ece = compute_ece(preds_mean.flatten(), targets.flatten())
    avg_uncertainty = preds_std.mean()
    
    results = {
        'method': 'MC_Dropout',
        'dice': float(dice),
        'ece': float(ece),
        'avg_uncertainty': float(avg_uncertainty),
        'has_uncertainty': True
    }
    
    print(f"Dice: {dice:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Avg Uncertainty: {avg_uncertainty:.4f}")
    
    return results


def evaluate_ensemble(ensemble_dir, n_members, test_loader, device, in_ch):
    """Evaluate Deep Ensemble"""
    print("\n" + "="*60)
    print(f"EVALUATING DEEP ENSEMBLE (members={n_members})")
    print("="*60)
    
    # Load all ensemble members
    models = []
    for i in range(n_members):
        model_path = os.path.join(ensemble_dir, f'member_{i}', 'best_model.pth')
        if os.path.exists(model_path):
            model = load_model(model_path, in_ch, device)
            models.append(model)
            print(f"✓ Loaded member {i}")
        else:
            print(f"⚠️  Member {i} not found at {model_path}")
    
    if len(models) == 0:
        print("❌ No ensemble members found!")
        return {'method': 'Deep_Ensemble', 'error': 'No models found'}
    
    print(f"Using {len(models)} ensemble members")
    
    all_preds_mean = []
    all_preds_std = []
    all_targets = []
    
    for batch in tqdm(test_loader, desc="Deep Ensemble"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Get predictions from all members
        member_preds = []
        for model in models:
            with torch.no_grad():
                logits = model(images)
                probs = torch.sigmoid(logits)
                member_preds.append(probs.cpu())
        
        member_preds = torch.stack(member_preds)
        pred_mean = member_preds.mean(dim=0)
        pred_std = member_preds.std(dim=0)
        
        all_preds_mean.append(pred_mean)
        all_preds_std.append(pred_std)
        all_targets.append(masks.cpu())
    
    preds_mean = torch.cat(all_preds_mean)
    preds_std = torch.cat(all_preds_std)
    targets = torch.cat(all_targets)
    
    pred_binary = (preds_mean > 0.5).float()
    dice = compute_dice(pred_binary, targets)
    ece = compute_ece(preds_mean.flatten(), targets.flatten())
    avg_uncertainty = preds_std.mean()
    
    results = {
        'method': 'Deep_Ensemble',
        'dice': float(dice),
        'ece': float(ece),
        'avg_uncertainty': float(avg_uncertainty),
        'n_members': len(models),
        'has_uncertainty': True
    }
    
    print(f"Dice: {dice:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Avg Uncertainty: {avg_uncertainty:.4f}")
    
    return results


def evaluate_swag(swag_model, test_loader, device, n_samples=30):
    """Evaluate SWAG"""
    print("\n" + "="*60)
    print(f"EVALUATING SWAG (samples={n_samples})")
    print("="*60)
    
    all_preds_mean = []
    all_preds_std = []
    all_targets = []
    
    for batch in tqdm(test_loader, desc="SWAG"):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Get predictions with uncertainty from SWAG
        mean_pred, uncertainty = swag_model.predict_with_uncertainty(
            images, 
            n_samples=n_samples, 
            scale=0.5
        )
        
        all_preds_mean.append(mean_pred.cpu())
        all_preds_std.append(uncertainty.cpu())
        all_targets.append(masks.cpu())
    
    preds_mean = torch.cat(all_preds_mean)
    preds_std = torch.cat(all_preds_std)
    targets = torch.cat(all_targets)
    
    pred_binary = (preds_mean > 0.5).float()
    dice = compute_dice(pred_binary, targets)
    ece = compute_ece(preds_mean.flatten(), targets.flatten())
    avg_uncertainty = preds_std.mean()
    
    results = {
        'method': 'SWAG',
        'dice': float(dice),
        'ece': float(ece),
        'avg_uncertainty': float(avg_uncertainty),
        'n_samples': n_samples,
        'has_uncertainty': True
    }
    
    print(f"Dice: {dice:.4f}")
    print(f"ECE: {ece:.4f}")
    print(f"Avg Uncertainty: {avg_uncertainty:.4f}")
    
    return results


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("UNCERTAINTY QUANTIFICATION EVALUATION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Save dir: {args.save_dir}")
    
    # Load test data
    test_dataset = BraTSSegmentationDataset(
        npz_dir=os.path.join(args.data_dir, 'test'),
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=4)
    
    print(f"Test samples: {len(test_dataset)}")
    
    all_results = []
    
    # 1. Baseline
    if os.path.exists(args.baseline_model):
        print("\n📊 Evaluating Baseline...")
        baseline_model = load_model(args.baseline_model, args.in_ch, device)
        results = evaluate_baseline(baseline_model, test_loader, device)
        all_results.append(results)
    else:
        print(f"⚠️  Baseline model not found: {args.baseline_model}")
    
    # 2. MC Dropout
    if os.path.exists(args.mc_dropout_model):
        print("\n📊 Evaluating MC Dropout...")
        mc_model = load_model(args.mc_dropout_model, args.in_ch, device, dropout_rate=args.dropout_rate)
        results = evaluate_mc_dropout(mc_model, test_loader, device, args.mc_samples)
        all_results.append(results)
    else:
        print(f"⚠️  MC Dropout model not found: {args.mc_dropout_model}")
    
    # 3. Deep Ensemble
    if os.path.exists(args.ensemble_dir):
        print("\n📊 Evaluating Deep Ensemble...")
        results = evaluate_ensemble(args.ensemble_dir, args.num_ensemble, test_loader, device, args.in_ch)
        all_results.append(results)
    else:
        print(f"⚠️  Ensemble directory not found: {args.ensemble_dir}")
    
    # 4. SWAG
    if os.path.exists(args.swag_model):
        print("\n📊 Evaluating SWAG...")
        swag_model = load_swag_model(args.swag_model, args.in_ch, device)
        results = evaluate_swag(swag_model, test_loader, device, args.swag_samples)
        all_results.append(results)
    else:
        print(f"⚠️  SWAG model not found: {args.swag_model}")
    
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
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {args.save_dir}/results.json")
    print("="*60)


if __name__ == '__main__':
    main()
