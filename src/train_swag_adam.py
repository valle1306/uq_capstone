"""
SWAG Training with Adam Optimizer for Chest X-Ray Classification
Following medical imaging literature (Mehta et al. 2021, Adams & Elhabian 2023)

Adaptation of Maddox et al. (2019) SWAG with Adam optimizer instead of SGD.
While the original paper uses SGD, recent medical imaging work has successfully
applied SWAG with Adam for improved convergence on smaller datasets.

Key details:
- Adam optimizer with lr=0.0001 (medical imaging standard)
- Cosine annealing learning rate schedule
- Collection: Epochs 27-50 (last 46% of training, matching original paper)
- BN statistics update after collection
- Sample with scale=0.0/0.5 for evaluation

References:
- Original SWAG: Maddox et al. (2019) - https://arxiv.org/abs/1902.02476
- Medical imaging application: Mehta et al. (2021), Adams & Elhabian (2023)
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import models

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_utils_classification import get_classification_loaders
from swag import SWAG


def bn_update(loader, model, device):
    """
    Update batch normalization statistics using training set.
    Critical for SWAG! From utils.py in original repo.
    
    After sampling weights from SWAG posterior, BN statistics 
    (running mean/var) are stale. This recomputes them.
    """
    model.train()
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc='BN Update', leave=False):
            inputs = inputs.to(device)
            model(inputs)  # Forward pass updates BN running stats
    model.eval()


def train_epoch(model, loader, criterion, optimizer, device):
    """Standard training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc='Train', leave=False):
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

    return running_loss / total, 100. * correct / total


def eval_model(model, loader, criterion, device):
    """Evaluate model on dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval', leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, 100. * correct / total


def build_resnet(arch, num_classes, pretrained=False):
    """Build ResNet model."""
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif arch == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f'Unknown architecture: {arch}')
    
    # Modify final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model


def main():
    parser = argparse.ArgumentParser(description='SWAG Training with Adam Optimizer')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default='data/chest_xray',
                        help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained ImageNet weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Total training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (medical imaging standard)')
    parser.add_argument('--lr_final', type=float, default=0.00005,
                        help='Final learning rate for cosine annealing')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='Adam beta parameters')
    
    # SWAG arguments (matching original paper proportions)
    parser.add_argument('--swa_start', type=int, default=27,
                        help='Epoch to start SWAG collection (default: 27, last 46% for 50 epochs)')
    parser.add_argument('--max_num_models', type=int, default=20,
                        help='Maximum number of models to collect in SWAG')
    parser.add_argument('--swag_rank', type=int, default=20,
                        help='Rank of low-rank covariance approximation')
    
    # Evaluation arguments
    parser.add_argument('--num_samples', type=int, default=30,
                        help='Number of samples for SWAG prediction')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Scale for SWAG sampling (0.0=mean, 0.5=half std, 1.0=full std)')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, 
                        default='results/swag_adam',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*70)
    print("SWAG Training with Adam Optimizer")
    print("="*70)
    print(f"Collection period: Epochs {args.swa_start}-{args.epochs}")
    print(f"Collection length: {args.epochs - args.swa_start + 1} epochs ({100*(args.epochs - args.swa_start + 1)/args.epochs:.1f}% of training)")
    print(f"Max models: {args.max_num_models}")
    print(f"SWAG rank: {args.swag_rank}")
    print("="*70)
    
    # Load datasets
    print("\nLoading datasets...")
    train_loader, cal_loader, test_loader = get_classification_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Calibration samples: {len(cal_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create base model
    print(f"\nCreating {args.model} model...")
    base_model = build_resnet(args.model, num_classes=2, pretrained=args.pretrained)
    base_model = base_model.to(device)
    
    # Create SWAG model wrapper
    print("Initializing SWAG wrapper...")
    swag_model = SWAG(
        base_model,
        no_cov_mat=False,  # Use full covariance
        max_num_models=args.max_num_models,
        loading=False
    )
    swag_model = swag_model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer (medical imaging standard)
    print("\nUsing Adam optimizer (medical imaging standard)")
    optimizer = optim.Adam(
        swag_model.parameters(),
        lr=args.lr,
        betas=tuple(args.adam_betas),
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_final
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': [],
        'swag_collected': []
    }
    
    best_acc = 0.0
    swag_n_collected = 0
    
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check if we're in SWAG collection period
        in_swag_period = epoch >= args.swa_start
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f}", end='')
        if in_swag_period:
            print(f" | SWAG: Collecting ({swag_n_collected+1}/{args.max_num_models})")
        else:
            print(f" | Pre-SWAG (starts at epoch {args.swa_start+1})")
        
        # Train
        train_loss, train_acc = train_epoch(
            swag_model, train_loader, criterion, optimizer, device
        )
        
        # Collect model for SWAG
        if in_swag_period:
            swag_model.collect_model(swag_model)
            swag_n_collected += 1
            print(f"  → SWAG snapshot collected ({swag_n_collected} total)")
        
        # Evaluate base model
        test_loss, test_acc = eval_model(
            swag_model, test_loader, criterion, device
        )
        
        # Step scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        history['swag_collected'].append(swag_n_collected)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Save best base model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': swag_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'swag_n_collected': swag_n_collected
            }, save_dir / 'best_model.pt')
            print(f"  ✓ New best model saved! (Acc: {test_acc:.2f}%)")
    
    print("\n" + "="*70)
    print("Training complete! Now evaluating SWAG...")
    print("="*70)
    
    # Sample from SWAG and update BN statistics
    print(f"\nSampling from SWAG (scale={args.scale})...")
    swag_model.sample(scale=args.scale, cov=True)
    
    print("Updating batch normalization statistics...")
    bn_update(train_loader, swag_model, device)
    
    # Evaluate SWAG model
    print("Evaluating SWAG model...")
    swag_test_loss, swag_test_acc = eval_model(
        swag_model, test_loader, criterion, device
    )
    
    print(f"\nFinal Results:")
    print(f"  Base Model (last epoch): {test_acc:.2f}%")
    print(f"  Best Base Model: {best_acc:.2f}%")
    print(f"  SWAG Model (scale={args.scale}): {swag_test_acc:.2f}%")
    print(f"  SWAG snapshots collected: {swag_n_collected}")
    
    # Save final SWAG model
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': swag_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'swag_test_acc': swag_test_acc,
        'swag_test_loss': swag_test_loss,
        'base_test_acc': test_acc,
        'swag_n_collected': swag_n_collected,
        'scale': args.scale
    }, save_dir / 'swag_model.pt')
    
    # Save training history
    history['swag_test_acc'] = swag_test_acc
    history['swag_test_loss'] = swag_test_loss
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
