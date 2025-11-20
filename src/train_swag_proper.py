"""
SWAG Training for Chest X-Ray Classification
Adapted from wjmaddox/swa_gaussian experiments/train/run_swag.py

Key differences from original:
- Dataset: Chest X-ray (4172 samples) instead of CIFAR-10 (50K samples)
- Architecture: ResNet-18 pretrained on ImageNet
- Training: 50 epochs (scaled from 300 in original)
- Collection: Epochs 27-50 (scaled from 161-300, last 54% of training)

Critical implementation details from Maddox et al. (2019):
1. SGD optimizer with momentum=0.9 (not Adam!)
2. Learning rate schedule: cosine annealing from lr_init to swa_lr
3. BN statistics update after collection using training set
4. Sample with scale=0.0 (mean weights) for evaluation

Reference: 
- Paper: https://arxiv.org/abs/1902.02476
- Code: https://github.com/wjmaddox/swa_gaussian
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


def schedule(epoch, args):
    """
    Learning rate schedule from original SWAG paper.
    Cosine annealing from lr_init to swa_lr over training epochs.
    
    From run_swag.py:
        t = epoch / (swa_start if swa else epochs)
        lr_ratio = swa_lr / lr_init
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return lr_init * factor
    """
    t = epoch / args.swa_start  # Normalized time until SWA starts
    lr_ratio = args.swa_lr / args.lr_init
    
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    
    return args.lr_init * factor


def adjust_learning_rate(optimizer, lr):
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    
    # Replace final layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser(description='SWAG Training (Maddox et al. 2019)')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='chest_xray',
                        help='dataset name (default: chest_xray)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='path to dataset (default: None)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loading workers (default: 4)')
    
    # Model
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='model architecture (default: resnet18)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained ImageNet weights')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs (default: 50)')
    parser.add_argument('--lr_init', type=float, default=0.01,
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    
    # SWAG
    parser.add_argument('--swa_start', type=int, default=27,
                        help='epoch to start SWAG collection (default: 27, which is 54%% of 50 epochs)')
    parser.add_argument('--swa_lr', type=float, default=0.005,
                        help='SWAG learning rate (default: 0.005)')
    parser.add_argument('--swa_c_epochs', type=int, default=1,
                        help='SWAG collection frequency (default: 1, collect every epoch)')
    parser.add_argument('--max_num_models', type=int, default=20,
                        help='maximum number of SWAG models (default: 20)')
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='runs/classification/swag_proper',
                        help='output directory (default: runs/classification/swag_proper)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device (default: cuda)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save command
    with open(os.path.join(args.output_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv) + '\n')
    
    print('\n' + '='*80)
    print('SWAG Training for Chest X-Ray Classification')
    print('Based on: Maddox et al. "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (NeurIPS 2019)')
    print('='*80)
    print(f'Dataset: {args.dataset}')
    print(f'Architecture: {args.arch} (pretrained={args.pretrained})')
    print(f'Epochs: {args.epochs}, SWAG collection starts at epoch {args.swa_start}')
    print(f'Optimizer: SGD(lr={args.lr_init}, momentum={args.momentum}, wd={args.wd})')
    print(f'SWAG LR: {args.swa_lr}, Max models: {args.max_num_models}')
    print(f'Device: {device}')
    print('='*80 + '\n')
    
    # Data
    print('Loading data...')
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f'Train: {len(train_loader.dataset)}, Cal: {len(cal_loader.dataset)}, Test: {len(test_loader.dataset)}')
    print(f'Classes: {num_classes}\n')
    
    # Model
    print('Building model...')
    model = build_resnet(args.arch, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print(f'Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n')
    
    # SWAG wrapper
    print('Creating SWAG wrapper...')
    swag_model = SWAG(
        base_model=model,
        max_num_models=args.max_num_models,
        var_clamp=1e-30,
        max_var=100.0  # Prevent variance explosion
    )
    swag_model = swag_model.to(device)
    print(f'SWAG: max_models={args.max_num_models}\n')
    
    # Optimizer and loss
    # CRITICAL: Use SGD with momentum, not Adam!
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.wd
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print('Starting training...\n')
    history = {
        'epoch': [],
        'lr': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'swag_test_loss': [],
        'swag_test_acc': [],
        'swag_n_models': []
    }
    
    best_test_acc = 0.0
    best_swag_acc = 0.0
    
    for epoch in range(args.epochs):
        # Learning rate schedule
        lr = schedule(epoch, args)
        adjust_learning_rate(optimizer, lr)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate SGD model
        test_loss, test_acc = eval_model(model, test_loader, criterion, device)
        
        # SWAG collection
        if epoch >= args.swa_start and (epoch - args.swa_start) % args.swa_c_epochs == 0:
            print(f'  [Epoch {epoch+1}] Collecting SWAG model snapshot ({swag_model.n_models+1}/{args.max_num_models})')
            swag_model.collect_model(model)
            
            # Evaluate SWAG model
            swag_model.sample(scale=0.0)  # Sample mean weights
            bn_update(train_loader, swag_model.base_model, device)  # Update BN stats
            swag_test_loss, swag_test_acc = eval_model(swag_model.base_model, test_loader, criterion, device)
        else:
            swag_test_loss, swag_test_acc = None, None
        
        # Log
        history['epoch'].append(epoch + 1)
        history['lr'].append(lr)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['swag_test_loss'].append(swag_test_loss)
        history['swag_test_acc'].append(swag_test_acc)
        history['swag_n_models'].append(swag_model.n_models)
        
        # Print
        print(f'Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | '
              f'Train: {train_loss:.4f} / {train_acc:.2f}% | '
              f'Test: {test_loss:.4f} / {test_acc:.2f}%', end='')
        if swag_test_acc is not None:
            print(f' | SWAG: {swag_test_loss:.4f} / {swag_test_acc:.2f}%', end='')
        print()
        
        # Save best models
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss
            }, os.path.join(args.output_dir, 'best_sgd_model.pth'))
        
        if swag_test_acc is not None and swag_test_acc > best_swag_acc:
            best_swag_acc = swag_test_acc
            torch.save({
                'n_models': swag_model.n_models,
                'mean': swag_model.mean.cpu(),
                'sq_mean': swag_model.sq_mean.cpu(),
                'cov_mat_sqrt': [c.cpu() for c in swag_model.cov_mat_sqrt],
                'max_num_models': swag_model.max_num_models,
                'epoch': epoch + 1,
                'test_acc': swag_test_acc,
                'test_loss': swag_test_loss
            }, os.path.join(args.output_dir, 'best_swag_model.pth'))
    
    # Final evaluation with multiple SWAG samples
    print('\n' + '='*80)
    print('Final Evaluation')
    print('='*80)
    
    # SGD model
    final_test_loss, final_test_acc = eval_model(model, test_loader, criterion, device)
    print(f'SGD Model: Loss={final_test_loss:.4f}, Acc={final_test_acc:.2f}%')
    
    # SWAG model (mean weights)
    swag_model.sample(scale=0.0)
    bn_update(train_loader, swag_model.base_model, device)
    swag_mean_loss, swag_mean_acc = eval_model(swag_model.base_model, test_loader, criterion, device)
    print(f'SWAG Mean: Loss={swag_mean_loss:.4f}, Acc={swag_mean_acc:.2f}%')
    
    # SWAG ensemble (sample 30 models)
    print('\nSWAG Ensemble (30 samples):')
    ensemble_predictions = []
    for i in range(30):
        swag_model.sample(scale=0.5)  # Scale from paper
        bn_update(train_loader, swag_model.base_model, device)
        
        # Get predictions
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = swag_model.base_model(inputs)
                all_preds.append(torch.softmax(outputs, dim=1).cpu())
                all_labels.append(labels)
        
        all_preds = torch.cat(all_preds, dim=0)
        ensemble_predictions.append(all_preds)
    
    # Average predictions
    ensemble_probs = torch.stack(ensemble_predictions, dim=0).mean(dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    ensemble_preds = ensemble_probs.argmax(dim=1)
    ensemble_acc = (ensemble_preds == all_labels).float().mean().item() * 100
    
    print(f'SWAG Ensemble: Acc={ensemble_acc:.2f}%')
    
    # Save results
    results = {
        'sgd_test_acc': final_test_acc,
        'sgd_test_loss': final_test_loss,
        'swag_mean_acc': swag_mean_acc,
        'swag_mean_loss': swag_mean_loss,
        'swag_ensemble_acc': ensemble_acc,
        'swag_n_models': swag_model.n_models,
        'best_sgd_acc': best_test_acc,
        'best_swag_acc': best_swag_acc
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print('\n' + '='*80)
    print('Training Complete!')
    print(f'Results saved to: {args.output_dir}')
    print('='*80)


if __name__ == '__main__':
    main()
