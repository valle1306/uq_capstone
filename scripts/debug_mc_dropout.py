#!/usr/bin/env python3
"""
Debug MC Dropout evaluation to identify the performance drop issue
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
from train_classifier_mc_dropout import ResNetWithDropout


def debug_mc_dropout():
    """Debug MC Dropout evaluation"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    _, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name='chest_xray',
        batch_size=32,
        num_workers=4
    )
    
    model_path = '/scratch/hpl14/uq_capstone/runs/classification/mc_dropout/best_model.pth'
    
    # Load model
    print(f"Loading MC Dropout model from {model_path}")
    base_model = models.resnet18(pretrained=False)
    model = ResNetWithDropout(base_model, num_classes, dropout_rate=0.3)
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("\nCheckpoint keys:", checkpoint.keys())
    print("Model architecture:")
    print(model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Test 1: Single forward pass (eval mode, dropout disabled)
    print("\n" + "="*70)
    print("Test 1: Single forward pass (eval mode, no MC)")
    print("="*70)
    model.eval()
    
    all_preds_eval = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds_eval.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_preds_eval = np.concatenate(all_preds_eval)
    all_labels = np.concatenate(all_labels)
    
    acc_eval = np.mean(all_preds_eval == all_labels) * 100
    print(f"Accuracy (eval mode): {acc_eval:.2f}%")
    
    # Test 2: MC Dropout with dropout enabled
    print("\n" + "="*70)
    print("Test 2: MC Dropout (T=20 samples, dropout enabled)")
    print("="*70)
    model.enable_dropout()
    
    all_preds_mc = []
    all_labels_mc = []
    all_uncertainties = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            
            # MC sampling
            probs_samples = []
            for _ in range(20):
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                probs_samples.append(probs.cpu())
            
            probs_samples = torch.stack(probs_samples)  # [20, B, 2]
            probs_mean = probs_samples.mean(dim=0)      # [B, 2]
            probs_var = probs_samples.var(dim=0)        # [B, 2]
            uncertainty = probs_var.mean(dim=1)         # [B]
            
            all_preds_mc.append(torch.argmax(probs_mean, dim=1).numpy())
            all_uncertainties.append(uncertainty.numpy())
            all_labels_mc.append(labels.numpy())
    
    all_preds_mc = np.concatenate(all_preds_mc)
    all_uncertainties = np.concatenate(all_uncertainties)
    all_labels_mc = np.concatenate(all_labels_mc)
    
    acc_mc = np.mean(all_preds_mc == all_labels_mc) * 100
    print(f"Accuracy (MC Dropout, T=20): {acc_mc:.2f}%")
    print(f"Mean uncertainty: {np.mean(all_uncertainties):.6f}")
    print(f"Std uncertainty: {np.std(all_uncertainties):.6f}")
    
    # Uncertainty separation
    correct = (all_preds_mc == all_labels_mc).astype(int)
    correct_unc = all_uncertainties[correct == 1]
    incorrect_unc = all_uncertainties[correct == 0]
    print(f"Correct predictions - Mean unc: {np.mean(correct_unc):.6f}")
    print(f"Incorrect predictions - Mean unc: {np.mean(incorrect_unc):.6f}")
    print(f"Separation: {np.mean(incorrect_unc) - np.mean(correct_unc):.6f}")
    
    # Test 3: Check dropout is actually happening
    print("\n" + "="*70)
    print("Test 3: Verify dropout is active")
    print("="*70)
    model.enable_dropout()
    
    # Get two forward passes and check they differ
    inputs_batch, _ = next(iter(test_loader))
    inputs_batch = inputs_batch[:1].to(device)  # Single image
    
    with torch.no_grad():
        out1 = model(inputs_batch)
        out2 = model(inputs_batch)
    
    diff = (out1 - out2).abs().max().item()
    print(f"Difference between two passes (should be > 0 if dropout is active): {diff:.6f}")
    
    if diff < 1e-6:
        print("⚠️  WARNING: Outputs are identical! Dropout may not be active.")
    else:
        print("✓ Dropout is active - outputs differ as expected")


if __name__ == '__main__':
    debug_mc_dropout()
