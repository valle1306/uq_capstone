"""
SGD Baseline Training for Chest X-Ray Classification
Matches the training setup of SWAG for fair comparison (Maddox et al. 2019)

Key details:
- SGD optimizer with momentum=0.9 (same as SWAG)
- Same learning rate schedule: cosine annealing
- Same weight decay, batch size, epochs
- Only difference: no SWAG collection

This provides an apples-to-apples comparison with SWAG.
"""

import os
import sys
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


def schedule(epoch, lr_init, swa_lr, swa_start):
    """
    Learning rate schedule from SWAG paper.
    Cosine annealing from lr_init to swa_lr.
    """
    t = epoch / swa_start
    lr_ratio = swa_lr / lr_init
    
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    
    return lr_init * factor


def adjust_learning_rate(optimizer, lr):
    """Set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
    parser = argparse.ArgumentParser(description='SGD Baseline Training (for SWAG comparison)')
    
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
    
    # Training (match SWAG settings)
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs (default: 50)')
    parser.add_argument('--lr_init', type=float, default=0.01,
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--swa_start', type=int, default=27,
                        help='epoch for LR schedule reference (default: 27)')
    parser.add_argument('--swa_lr', type=float, default=0.005,
                        help='final learning rate (default: 0.005)')
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='runs/classification/baseline_sgd',
                        help='output directory (default: runs/classification/baseline_sgd)')
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
    print('SGD Baseline Training for Chest X-Ray Classification')
    print('Matches SWAG training setup for fair comparison')
    print('='*80)
    print(f'Dataset: {args.dataset}')
    print(f'Architecture: {args.arch} (pretrained={args.pretrained})')
    print(f'Epochs: {args.epochs}')
    print(f'Optimizer: SGD(lr={args.lr_init}, momentum={args.momentum}, wd={args.wd})')
    print(f'LR Schedule: Cosine annealing {args.lr_init} -> {args.swa_lr}')
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
    
    # Optimizer and loss (CRITICAL: Use SGD with momentum, not Adam!)
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
        'test_acc': []
    }
    
    best_test_acc = 0.0
    
    for epoch in range(args.epochs):
        # Learning rate schedule (same as SWAG)
        lr = schedule(epoch, args.lr_init, args.swa_lr, args.swa_start)
        adjust_learning_rate(optimizer, lr)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = eval_model(model, test_loader, criterion, device)
        
        # Log
        history['epoch'].append(epoch + 1)
        history['lr'].append(lr)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print
        print(f'Epoch {epoch+1:3d}/{args.epochs} | LR: {lr:.6f} | '
              f'Train: {train_loss:.4f} / {train_acc:.2f}% | '
              f'Test: {test_loss:.4f} / {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss
            }, os.path.join(args.output_dir, 'best_model.pth'))
    
    # Final evaluation
    print('\n' + '='*80)
    print('Final Evaluation')
    print('='*80)
    
    final_test_loss, final_test_acc = eval_model(model, test_loader, criterion, device)
    print(f'SGD Baseline: Loss={final_test_loss:.4f}, Acc={final_test_acc:.2f}%')
    print(f'Best Test Acc: {best_test_acc:.2f}%')
    
    # Save results
    results = {
        'final_test_acc': final_test_acc,
        'final_test_loss': final_test_loss,
        'best_test_acc': best_test_acc
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
