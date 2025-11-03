#!/usr/bin/env python3
"""
Debug SWAG evaluation to identify the performance gap with Ensemble
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, '/scratch/hpl14/uq_capstone/src')

from data_utils_classification import get_classification_loaders
from swag import load_swag_model


def debug_swag():
    """Debug SWAG evaluation"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    _, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name='chest_xray',
        batch_size=32,
        num_workers=4
    )
    
    swag_path = '/scratch/hpl14/uq_capstone/runs/classification/swag_classification/swag_model.pth'
    baseline_path = '/scratch/hpl14/uq_capstone/runs/classification/baseline/best_model.pth'
    
    # Load baseline for comparison
    print(f"\nLoading baseline model from {baseline_path}")
    base_model_bl = models.resnet18(pretrained=False)
    num_features = base_model_bl.fc.in_features
    base_model_bl.fc = nn.Linear(num_features, num_classes)
    checkpoint_bl = torch.load(baseline_path, map_location='cpu')
    base_model_bl.load_state_dict(checkpoint_bl['model_state_dict'])
    base_model_bl = base_model_bl.to(device)
    base_model_bl.eval()
    
    # Test 1: Baseline
    print("\n" + "="*70)
    print("Test 1: Baseline Model (for reference)")
    print("="*70)
    
    all_preds_baseline = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = base_model_bl(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds_baseline.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds_baseline = np.concatenate(all_preds_baseline)
    all_labels = np.concatenate(all_labels)
    
    acc_baseline = np.mean(all_preds_baseline == all_labels) * 100
    print(f"Baseline Accuracy: {acc_baseline:.2f}%")
    
    # Load SWAG
    print(f"\n\nLoading SWAG model from {swag_path}")
    base_model = models.resnet18(pretrained=False)
    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, num_classes)
    swag = load_swag_model(swag_path, base_model)
    
    print(f"SWAG statistics:")
    print(f"  n_models: {swag.n_models}")
    print(f"  mean shape: {swag.mean.shape if swag.mean is not None else 'None'}")
    print(f"  sq_mean shape: {swag.sq_mean.shape if swag.sq_mean is not None else 'None'}")
    print(f"  cov_mat_sqrt entries: {len(swag.cov_mat_sqrt)}")
    if swag.mean is not None:
        print(f"  mean min/max: {swag.mean.min():.6f} / {swag.mean.max():.6f}")
        var = torch.clamp(swag.sq_mean - swag.mean ** 2, swag.var_clamp, swag.max_var)
        print(f"  var min/max: {var.min():.6e} / {var.max():.6e}")
    
    # Test 2: SWAG with mean weights (should match baseline)
    print("\n" + "="*70)
    print("Test 2: SWAG with Mean Weights (should match baseline)")
    print("="*70)
    
    swag.base_model.eval()
    swag.base_model = swag.base_model.to(device)
    
    all_preds_mean = []
    all_labels_mean = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = swag.base_model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds_mean.append(preds.cpu().numpy())
            all_labels_mean.append(labels.numpy())
    
    all_preds_mean = np.concatenate(all_preds_mean)
    all_labels_mean = np.concatenate(all_labels_mean)
    
    acc_swag_mean = np.mean(all_preds_mean == all_labels_mean) * 100
    print(f"SWAG Mean Model Accuracy: {acc_swag_mean:.2f}%")
    print(f"  (should be close to baseline {acc_baseline:.2f}%)")
    
    # Test 3: SWAG sampling with different scales
    print("\n" + "="*70)
    print("Test 3: SWAG Sampling with Different Scales")
    print("="*70)
    
    scales = [0.0, 0.1, 0.3, 0.5, 1.0]
    
    for scale in scales:
        print(f"\n  Scale: {scale}")
        
        all_preds_swag = []
        all_uncertainties = []
        all_labels_swag = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'    Scale {scale}'):
                inputs = inputs.to(device)
                
                # Try single forward pass first with scale 0
                if scale == 0.0:
                    # With scale=0, we should get mean model
                    sampled = swag.sample(scale=scale)
                    sampled = sampled.to(device)
                    sampled.eval()
                    outputs = sampled(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds_swag.append(preds.cpu().numpy())
                    all_labels_swag.append(labels.numpy())
                else:
                    # MC sampling
                    probs_samples = []
                    for _ in range(20):
                        sampled = swag.sample(scale=scale)
                        sampled = sampled.to(device)
                        sampled.eval()
                        outputs = sampled(inputs)
                        probs = torch.softmax(outputs, dim=1)
                        probs_samples.append(probs.cpu())
                    
                    probs_samples = torch.stack(probs_samples)
                    probs_mean = probs_samples.mean(dim=0)
                    probs_var = probs_samples.var(dim=0)
                    uncertainty = probs_var.mean(dim=1)
                    
                    all_preds_swag.append(torch.argmax(probs_mean, dim=1).numpy())
                    all_uncertainties.append(uncertainty.numpy())
                    all_labels_swag.append(labels.numpy())
        
        all_preds_swag = np.concatenate(all_preds_swag)
        all_labels_swag = np.concatenate(all_labels_swag)
        
        acc = np.mean(all_preds_swag == all_labels_swag) * 100
        print(f"    Accuracy: {acc:.2f}%")
        
        if all_uncertainties:
            all_uncertainties = np.concatenate(all_uncertainties)
            correct = (all_preds_swag == all_labels_swag).astype(int)
            correct_unc = all_uncertainties[correct == 1]
            incorrect_unc = all_uncertainties[correct == 0]
            print(f"    Mean Unc (correct): {np.mean(correct_unc):.6f}")
            print(f"    Mean Unc (incorrect): {np.mean(incorrect_unc):.6f}")
            print(f"    Separation: {np.mean(incorrect_unc) - np.mean(correct_unc):.6f}")
    
    # Test 4: Check if cov_mat_sqrt is properly loaded
    print("\n" + "="*70)
    print("Test 4: Covariance Matrix Status")
    print("="*70)
    print(f"Number of collected deviations: {len(swag.cov_mat_sqrt)}")
    
    if len(swag.cov_mat_sqrt) > 0:
        for i, dev in enumerate(swag.cov_mat_sqrt[:3]):  # Show first 3
            print(f"  Deviation {i}: shape {dev.shape}, min {dev.min():.6e}, max {dev.max():.6e}")


if __name__ == '__main__':
    debug_swag()
