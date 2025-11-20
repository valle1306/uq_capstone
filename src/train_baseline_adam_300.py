"""
Adam Baseline Training for Chest X-Ray Classification - 300 Epochs
Extends medical imaging standard to match SWAG paper training duration

Key details:
- Adam optimizer with lr=0.0001 (medical imaging standard)
- 300 epochs (matching original SWAG paper duration)
- Cosine annealing learning rate schedule
- Same weight decay, batch size as SGD experiments
- Provides comparison for SWAG-Adam implementation

References:
- Mehta et al. (2021): Propagating uncertainty across cascaded medical imaging tasks
- Adams & Elhabian (2023): Benchmarking scalable epistemic UQ in organ segmentation
- Maddox et al. (2019): A Simple Baseline for Bayesian Uncertainty Estimation
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
    parser = argparse.ArgumentParser(description='Adam Baseline Training - 300 Epochs')
    
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
    
    # Training (300 epochs to match SWAG paper)
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs (default: 300)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate (default: 0.0001)')
    parser.add_argument('--lr_final', type=float, default=0.00005,
                        help='final learning rate (default: 0.00005)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--adam_betas', type=float, nargs=2, default=[0.9, 0.999],
                        help='Adam beta parameters (default: [0.9, 0.999])')
    
    # Misc
    parser.add_argument('--output_dir', type=str, default='results/baseline_adam_300',
                        help='output directory (default: results/baseline_adam_300)')
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
    print('Adam Baseline Training for Chest X-Ray Classification - 300 Epochs')
    print('Extends medical imaging standard to match SWAG paper duration')
    print('='*80)
    print(f'Dataset: {args.dataset}')
    print(f'Architecture: {args.arch} (pretrained={args.pretrained})')
    print(f'Epochs: {args.epochs}')
    print(f'Optimizer: Adam(lr={args.lr}, betas={args.adam_betas}, wd={args.weight_decay})')
    print(f'LR Schedule: Cosine annealing {args.lr} -> {args.lr_final}')
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
    
    # Optimizer and loss (Adam for medical imaging)
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
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss, test_acc = eval_model(model, test_loader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log
        history['epoch'].append(epoch + 1)
        history['lr'].append(current_lr)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print
        print(f'Epoch {epoch+1:3d}/{args.epochs} | LR: {current_lr:.6f} | '
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
    print(f'Adam Baseline: Loss={final_test_loss:.4f}, Acc={final_test_acc:.2f}%')
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
