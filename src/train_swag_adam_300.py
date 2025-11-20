"""
SWAG Training with Adam Optimizer for Chest X-Ray Classification - 300 Epochs
Extends medical imaging standard to match SWAG paper training duration

Adaptation of Maddox et al. (2019) SWAG with Adam optimizer instead of SGD.
While the original paper uses SGD with 300 epochs, this extends the
Adam-based approach to the same duration for fair comparison.

Key details:
- Adam optimizer with lr=0.0001 (medical imaging standard)
- 300 epochs (matching original SWAG paper duration)
- Cosine annealing learning rate schedule
- Collection: Epochs 161-300 (last 46% of training, matching original paper)
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
    parser = argparse.ArgumentParser(description='SWAG Training with Adam - 300 Epochs')
    
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
    
    # Training arguments (300 epochs to match SWAG paper)
    parser.add_argument('--epochs', type=int, default=300,
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
    parser.add_argument('--swa_start', type=int, default=161,
                        help='Epoch to start SWAG collection (default: 161, last 46% for 300 epochs)')
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
                        default='results/swag_adam_300',
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
    print("SWAG Training with Adam Optimizer - 300 Epochs")
    print("="*70)
    print(f"Collection period: Epochs {args.swa_start}-{args.epochs}")
    print(f"Collection length: {args.epochs - args.swa_start + 1} epochs ({100*(args.epochs - args.swa_start + 1)/args.epochs:.1f}% of training)")
    print(f"Max models: {args.max_num_models}")
    print(f"Rank: {args.swag_rank}")
    print("="*70)
    
    print("\nLoading datasets...")
    train_loader, cal_loader, test_loader = get_classification_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Calibration samples: {len(cal_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=args.pretrained)
    elif args.model == 'resnet34':
        model = models.resnet34(pretrained=args.pretrained)
    else:  # resnet50
        model = models.resnet50(pretrained=args.pretrained)
    
    # Modify final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = model.to(device)
    
    # Initialize SWAG wrapper
    print("Initializing SWAG wrapper...")
    swag_model = SWAG(
        model,
        no_cov_mat=False,
        max_num_models=args.max_num_models,
        rank=args.swag_rank
    )
    swag_model = swag_model.to(device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer (following medical imaging literature)
    print("Using Adam optimizer (medical imaging standard)")
    optimizer = optim.Adam(
        model.parameters(),
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
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lr': []
    }
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Collect SWAG statistics (after swa_start)
        if epoch >= args.swa_start:
            swag_model.collect_model(model)
            print(f"  → SWAG: Collected model (total: {swag_model.n_models})")
        
        # Evaluate
        test_loss, test_acc = eval_model(
            model, test_loader, criterion, device
        )
        
        # Step scheduler
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, save_dir / 'best_model.pt')
            print(f"✓ New best model saved! (Acc: {test_acc:.2f}%)")
    
    print("\n" + "="*50)
    print("Training complete! Now evaluating SWAG...")
    print("="*50)
    
    # Save SWAG model
    print("\nSaving SWAG model...")
    torch.save({
        'swag_state_dict': swag_model.state_dict(),
        'n_models': swag_model.n_models
    }, save_dir / 'swag_model.pt')
    
    # SWAG evaluation
    print(f"\nEvaluating SWAG with {args.num_samples} samples (scale={args.scale})...")
    swag_model.sample(scale=args.scale, cov=True)
    bn_update(train_loader, swag_model, device)
    
    # Collect predictions for all samples
    all_preds = []
    all_labels = []
    
    for _ in tqdm(range(args.num_samples), desc='SWAG Sampling'):
        swag_model.sample(scale=args.scale, cov=True)
        bn_update(train_loader, swag_model, device)
        
        swag_model.eval()
        sample_preds = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = swag_model(inputs)
                probs = torch.softmax(outputs, dim=1)
                sample_preds.append(probs.cpu())
                
                if len(all_labels) < len(test_loader.dataset):
                    all_labels.extend(labels.numpy())
        
        all_preds.append(torch.cat(sample_preds, dim=0))
    
    # Ensemble predictions
    all_preds = torch.stack(all_preds)  # [num_samples, num_test, num_classes]
    mean_probs = all_preds.mean(dim=0)  # [num_test, num_classes]
    pred_classes = mean_probs.argmax(dim=1).numpy()
    all_labels = np.array(all_labels)
    
    swag_acc = 100.0 * (pred_classes == all_labels).mean()
    
    # Compute uncertainty (predictive entropy)
    predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1).mean().item()
    
    print("\n" + "="*50)
    print(f"Best SGD accuracy: {best_acc:.2f}%")
    print(f"SWAG accuracy: {swag_acc:.2f}%")
    print(f"Predictive entropy: {predictive_entropy:.4f}")
    print("="*50)
    
    # Save results
    results = {
        'best_sgd_acc': best_acc,
        'swag_acc': swag_acc,
        'predictive_entropy': predictive_entropy,
        'num_samples': args.num_samples,
        'scale': args.scale,
        'n_models_collected': swag_model.n_models
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")


if __name__ == '__main__':
    main()
