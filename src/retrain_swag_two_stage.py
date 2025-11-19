"""
Two-Stage SWAG Training for Medical Image Classification

Strategy:
1. Load converged baseline model (91.67% accuracy)
2. Continue training with cyclic LR to force loss surface exploration
3. Collect SWAG snapshots during exploration phase
4. This prevents the "overfit before collection" problem

Author: Phan Nguyen Huong Le
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader
from torchvision import models
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils_classification import get_data_loaders
from swag import SWAG
from torch.optim.swa_utils import update_bn


def build_resnet(arch, num_classes, pretrained=True):
    """Build ResNet model"""
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(512, num_classes)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(2048, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser(description='Two-Stage SWAG Training')
    parser.add_argument('--dataset', type=str, default='chest_xray',
                        choices=['chest_xray', 'oct_retinal', 'brain_tumor'],
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--baseline_path', type=str, required=True,
                        help='Path to converged baseline model')
    parser.add_argument('--output_dir', type=str, 
                        default='runs/classification/swag_two_stage',
                        help='Output directory')
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for stage 2')
    parser.add_argument('--collection_start', type=int, default=20,
                        help='Epoch to start SWAG collection')
    parser.add_argument('--base_lr', type=float, default=0.0001,
                        help='Base learning rate for cyclic LR')
    parser.add_argument('--max_lr', type=float, default=0.001,
                        help='Max learning rate for cyclic LR')
    parser.add_argument('--cycle_length', type=int, default=5,
                        help='Cycle length for cyclic LR')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--swag_lr', type=float, default=0.0001,
                        help='Learning rate during SWAG collection')
    parser.add_argument('--num_snapshots', type=int, default=20,
                        help='Number of SWAG snapshots to collect')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Two-Stage SWAG Training")
    print("=" * 70)
    print(f"Strategy: Load baseline → Cyclic LR exploration → SWAG collection")
    print(f"This prevents 'overfit before collection' problem!")
    print()
    print(f"Stage 1: Load converged baseline from {args.baseline_path}")
    print(f"Stage 2: Train {args.epochs} epochs with cyclic LR")
    print(f"  - Epochs 1-{args.collection_start}: Exploration with cyclic LR")
    print(f"  - Epochs {args.collection_start+1}-{args.epochs}: SWAG collection")
    print(f"  - Cyclic LR: {args.base_lr} ↔ {args.max_lr} (cycle={args.cycle_length})")
    print(f"  - SWAG LR: {args.swag_lr}")
    print("=" * 70)
    print()
    
    # Load data
    print("Loading dataset...")
    train_loader, cal_loader, test_loader = get_data_loaders(
        args.dataset,
        args.data_path,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    dataset_sizes = {
        'train': len(train_loader.dataset),
        'cal': len(cal_loader.dataset),
        'test': len(test_loader.dataset)
    }
    
    print(f"{args.dataset.upper()} dataset loaded")
    print(f"Train: {dataset_sizes['train']}, Cal: {dataset_sizes['cal']}, Test: {dataset_sizes['test']}")
    print()
    
    # Build model
    num_classes = 2 if args.dataset == 'chest_xray' else 4
    base_model = build_resnet(args.arch, num_classes, pretrained=False)
    
    # Load baseline weights
    print("=" * 70)
    print("Stage 1: Loading Converged Baseline Model")
    print("=" * 70)
    
    if not os.path.exists(args.baseline_path):
        raise FileNotFoundError(f"Baseline model not found: {args.baseline_path}")
    
    checkpoint = torch.load(args.baseline_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['state_dict'])
        baseline_acc = checkpoint.get('test_acc', 'Unknown')
        print(f"✓ Loaded baseline from checkpoint (Test Acc: {baseline_acc})")
    else:
        base_model.load_state_dict(checkpoint)
        print(f"✓ Loaded baseline weights")
    
    base_model = base_model.to(device)
    
    # Evaluate baseline
    criterion = nn.CrossEntropyLoss()
    baseline_train_loss, baseline_train_acc = evaluate(base_model, train_loader, criterion, device)
    baseline_val_loss, baseline_val_acc = evaluate(base_model, cal_loader, criterion, device)
    baseline_test_loss, baseline_test_acc = evaluate(base_model, test_loader, criterion, device)
    
    print(f"Baseline performance:")
    print(f"  Train: {baseline_train_acc:.2f}%")
    print(f"  Val:   {baseline_val_acc:.2f}%")
    print(f"  Test:  {baseline_test_acc:.2f}%")
    print()
    
    # Initialize SWAG
    swag = SWAG(base_model, max_num_models=args.num_snapshots, var_clamp=1e-6)
    swag = swag.to(device)
    
    # Stage 2: Cyclic LR exploration + SWAG collection
    print("=" * 70)
    print("Stage 2: Cyclic LR Exploration + SWAG Collection")
    print("=" * 70)
    print()
    
    optimizer = optim.Adam(
        swag.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )
    
    # Cyclic LR for first phase (exploration)
    scheduler = CyclicLR(
        optimizer,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        step_size_up=args.cycle_length,
        mode='triangular',
        cycle_momentum=False
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': [],
        'swag_snapshots': 0,
        'baseline_test_acc': baseline_test_acc
    }
    
    best_val_acc = baseline_val_acc
    snapshot_count = 0
    
    print("Starting Stage 2 Training...")
    print()
    
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_loss, train_acc = train_epoch(swag, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(swag, cal_loader, criterion, device)
        test_loss, test_acc = evaluate(swag, test_loader, criterion, device)
        
        # Update LR (cyclic for first phase, fixed low LR during collection)
        if epoch < args.collection_start:
            scheduler.step()
        else:
            # Switch to fixed low LR during SWAG collection
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.swag_lr
        
        # Collect SWAG snapshot
        if epoch >= args.collection_start:
            swag.collect_model(swag)
            snapshot_count += 1
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Train Acc: {train_acc:.2f}%")
            print(f"Val Acc:   {val_acc:.2f}%")
            print(f"Test Acc:  {test_acc:.2f}%")
            print(f"✓ Collected SWAG snapshot {snapshot_count} (total: {snapshot_count})")
        else:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"Learning Rate: {current_lr:.6f}")
            print(f"Train Acc: {train_acc:.2f}%")
            print(f"Val Acc:   {val_acc:.2f}%")
            print(f"Test Acc:  {test_acc:.2f}%")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        history['swag_snapshots'] = snapshot_count
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'state_dict': swag.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
                'snapshot_count': snapshot_count
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_acc: {val_acc:.2f}%)")
    
    print()
    print("=" * 70)
    print("Updating Batch Normalization statistics...")
    print("=" * 70)
    update_bn(train_loader, swag, device=device)
    print("✓ Batch norm updated")
    print()
    
    # Save final SWAG model
    swag.save(os.path.join(args.output_dir, 'swag_model.pth'))
    print(f"Saving SWAG model to {args.output_dir}/swag_model.pth")
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print()
    print("=" * 70)
    print("Training Complete!")
    print(f"End time: {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}")
    print("=" * 70)
    print(f"Baseline test accuracy: {baseline_test_acc:.2f}%")
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Total SWAG snapshots collected: {snapshot_count}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print()
    print("Output files:")
    print(f"  - best_model.pth: Best model checkpoint")
    print(f"  - swag_model.pth: Final SWAG model")
    print(f"  - training_history.json: Training curves")
    print()


if __name__ == '__main__':
    main()
