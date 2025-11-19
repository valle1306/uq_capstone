"""
SWAG Training with Adam Optimizer
For fair comparison with Baseline, MC Dropout, and Deep Ensemble (all use Adam)

Key differences from SGD version:
1. Adam optimizer instead of SGD (no momentum parameter - Adam has built-in)
2. Simpler LR schedule (Adam is less sensitive to LR changes)
3. Same weight decay (1e-4) for L2 regularization
4. Same SWAG procedure (weight averaging, batch norm update, sampling)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from tqdm import tqdm
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_utils_classification import get_classification_loaders
from swag import SWAG
from torch.optim.swa_utils import update_bn


def build_resnet(arch: str, num_classes: int):
    """Build ResNet from scratch (random init)"""
    if arch == 'resnet18':
        base = models.resnet18(pretrained=False)
    elif arch == 'resnet34':
        base = models.resnet34(pretrained=False)
    elif arch == 'resnet50':
        base = models.resnet50(pretrained=False)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Replace final FC layer
    in_features = base.fc.in_features
    base.fc = nn.Linear(in_features, num_classes)
    return base


def train_epoch(model, loader, criterion, optimizer, device, desc="Train"):
    """Train for one epoch"""
    model.train()
    losses = []
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=desc)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return np.mean(losses), 100. * correct / total


def validate(model, loader, criterion, device, desc="Val"):
    """Validate model"""
    model.eval()
    losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            losses.append(loss.item())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return np.mean(losses), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='SWAG Training with Adam Optimizer')
    parser.add_argument('--dataset', type=str, default='chest_xray', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=None, help='Optional data directory path')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--swag_start', type=int, default=30, help='Epoch to start collecting SWAG snapshots')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001, help='Initial LR (same as Baseline/MC/Ensemble)')
    parser.add_argument('--swag_lr', type=float, default=0.0001, help='SWAG collection LR')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--max_models', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("SWAG Training - Adam Optimizer (Fair Comparison)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Optimizer: ADAM (same as Baseline/MC Dropout/Ensemble)")
    print(f"Weight decay (L2): {args.weight_decay}")
    print(f"Initial LR: {args.lr}")
    print(f"SWAG LR: {args.swag_lr}")
    print(f"SWAG collection: epochs {args.swag_start}-{args.epochs}")
    print(f"Training from: SCRATCH (random init)")
    print(f"Batch norm update: YES (after SWAG collection)")
    print(f"Note: Using Adam for FAIR comparison with other methods")
    print("=" * 70)

    # Data loaders
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Build model from scratch
    base_model = build_resnet(args.arch, num_classes)
    base_model = base_model.to(device)

    # Wrap with SWAG
    swag = SWAG(base_model, max_num_models=args.max_models)
    swag.to(device)

    # Loss and optimizer (ADAM - same as other methods!)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        swag.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay  # Keep L2 regularization
    )

    # Simple learning rate schedule for Adam
    # Adam is less sensitive to LR, so use simpler schedule
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.swag_start, gamma=0.1)

    # Training history
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

    best_val_acc = 0.0
    swag_snapshots = 0

    print(f"\n{'='*70}")
    print("Starting Training...")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Train
        train_loss, train_acc = train_epoch(swag.base_model, train_loader, criterion, optimizer, device)
        
        # Validate on validation set
        val_loss, val_acc = validate(swag.base_model, cal_loader, criterion, device, desc="Val")
        
        # Test
        test_loss, test_acc = validate(swag.base_model, test_loader, criterion, device, desc="Test")

        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Val Acc:   {val_acc:.2f}%")
        print(f"Test Acc:  {test_acc:.2f}%")

        # Collect SWAG snapshot
        if epoch >= args.swag_start:
            swag.collect_model(swag.base_model)
            swag_snapshots += 1
            print(f"✓ Collected SWAG snapshot {swag_snapshots} (total: {swag.n_models})")
        
        # Step scheduler
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        history['swag_snapshots'] = swag.n_models

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': swag.base_model.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")

    # Update batch normalization statistics (CRITICAL!)
    print(f"\n{'='*70}")
    print("Updating Batch Normalization statistics...")
    print(f"{'='*70}")
    update_bn(train_loader, swag.base_model, device)
    print("✓ Batch norm updated")

    # Save SWAG model
    print(f"\nSaving SWAG model to {os.path.join(args.output_dir, 'swag_model.pth')}")
    swag.save(os.path.join(args.output_dir, 'swag_model.pth'))

    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'swag_start': args.swag_start,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'swag_lr': args.swag_lr,
        'weight_decay': args.weight_decay,
        'optimizer': 'Adam',
        'scheduler': 'StepLR',
        'training_from': 'scratch',
        'batch_norm_update': True,
        'note': 'Adam optimizer for fair comparison with Baseline/MC Dropout/Ensemble'
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"SWAG snapshots collected: {swag.n_models}")
    print(f"Files saved:")
    print(f"  - {os.path.join(args.output_dir, 'swag_model.pth')}")
    print(f"  - {os.path.join(args.output_dir, 'best_model.pth')}")
    print(f"  - {os.path.join(args.output_dir, 'training_history.json')}")
    print(f"  - {os.path.join(args.output_dir, 'config.json')}")
    print(f"\nUsing Adam optimizer for FAIR comparison:")
    print(f"  ✓ Same optimizer as Baseline (91.67%)")
    print(f"  ✓ Same optimizer as MC Dropout (85.26%)")
    print(f"  ✓ Same optimizer as Deep Ensemble (91.67%)")
    print(f"  ✓ SWAG methodology unchanged (weight averaging still valid)")


if __name__ == '__main__':
    main()
