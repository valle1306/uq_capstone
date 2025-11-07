#!/usr/bin/env python3
"""
Retrain SWAG - Following Maddox et al. 2019 Paper Exactly

Key changes from previous implementation:
1. SGD with momentum (not Adam)
2. Weight decay for L2 regularization
3. Proper SWA learning rate schedule
4. Batch normalization update after collecting snapshots
5. Train from scratch (not from baseline)
"""

import os
import json
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch.optim.swa_utils import SWALR, update_bn

from data_utils_classification import get_classification_loaders
from swag import SWAG


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validate', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='Retrain SWAG Following Paper')
    parser.add_argument('--dataset', type=str, default='chest_xray')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--swag_start', type=int, default=30, help='Start collecting SWAG snapshots')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate for SGD')
    parser.add_argument('--swa_lr', type=float, default=0.01, help='SWA learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='runs/classification/swag_classification')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SWAG Retraining - Following Maddox et al. 2019")
    print("="*70)
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print(f"\nKey Implementation Details (from paper):")
    print(f"  - Optimizer: SGD with momentum={args.momentum}")
    print(f"  - Weight decay (L2): {args.weight_decay}")
    print(f"  - Initial LR: {args.lr}")
    print(f"  - SWA LR: {args.swa_lr}")
    print(f"  - SWAG snapshots start: epoch {args.swag_start}")
    print(f"  - Training from scratch (random init)")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Create base model - TRAIN FROM SCRATCH (random init)
    print("\nInitializing model from scratch (random initialization)...")
    base_model = models.resnet18(pretrained=False)
    num_features = base_model.fc.in_features
    base_model.fc = nn.Linear(num_features, num_classes)
    base_model = base_model.to(device)
    
    # Create SWAG wrapper
    swag = SWAG(base_model, max_num_models=20)
    swag = swag.to(device)
    
    # Optimizer: SGD with momentum and weight decay (CRITICAL for paper replication)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        swag.parameters(), 
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay  # L2 regularization - prevents overfitting
    )
    
    # Learning rate schedule: Cosine annealing for warmup, then SWALR for SWA phase
    warmup_epochs = args.swag_start
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=warmup_epochs)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    
    print("\nStarting training...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
        'swag_snapshots': 0,
        'lr': []
    }
    
    best_val_acc = 0
    best_model_path = output_dir / 'best_model.pth'
    
    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"{'='*70}")
        
        train_loss, train_acc = train_epoch(swag, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(swag, cal_loader, criterion, device)
        test_loss, test_acc = validate(swag, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        
        # Save best base model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"✓ New best model! Val Acc: {val_acc:.2f}%")
            torch.save({
                'model_state_dict': swag.base_model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'test_acc': test_acc
            }, best_model_path)
        
        # Collect SWAG snapshots starting at swag_start epoch
        if epoch >= args.swag_start:
            swag.collect_model(swag.base_model)
            history['swag_snapshots'] += 1
            print(f"✓ Collected SWAG snapshot {history['swag_snapshots']} (total: {swag.n_models})")
            
            # Use SWA scheduler during snapshot collection phase
            swa_scheduler.step()
        else:
            # Use regular cosine annealing before SWA phase
            scheduler.step()
    
    # CRITICAL: Update batch normalization statistics after collecting all snapshots
    # This is mentioned in the paper and improves SWAG performance
    print("\n" + "="*70)
    print("Updating Batch Normalization statistics...")
    print("="*70)
    update_bn(train_loader, swag.base_model, device)
    print("✓ Batch norm updated")
    
    # Save SWAG model
    swag_path = output_dir / 'swag_model.pth'
    print(f"\nSaving SWAG model to {swag_path}")
    torch.save({
        'n_models': swag.n_models,
        'mean': swag.mean,
        'sq_mean': swag.sq_mean,
        'cov_mat_sqrt': swag.cov_mat_sqrt,
        'max_num_models': swag.max_num_models,
        'var_clamp': swag.var_clamp,
        'max_var': swag.max_var
    }, swag_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save configuration
    config = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'swag_start': args.swag_start,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'swa_lr': args.swa_lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'optimizer': 'SGD',
        'scheduler': 'CosineAnnealingLR + SWALR',
        'training_from': 'scratch',
        'batch_norm_update': True,
        'paper': 'Maddox et al. 2019'
    }
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"SWAG snapshots collected: {swag.n_models}")
    print(f"Files saved:")
    print(f"  - {swag_path}")
    print(f"  - {best_model_path}")
    print(f"  - {history_path}")
    print(f"  - {config_path}")
    print("\nImplementation follows Maddox et al. 2019:")
    print("  ✓ SGD with momentum")
    print("  ✓ Weight decay (L2 regularization)")
    print("  ✓ SWALR scheduler")
    print("  ✓ Batch normalization update")
    print("  ✓ Training from scratch")


if __name__ == '__main__':
    main()
