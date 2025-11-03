"""
Train ResNet with SWAG for classification (chest_xray)

This script adapts the SWAG wrapper to the classification pipeline used
in this repo. It uses `get_classification_loaders` from
`src/data_utils_classification.py` and a ResNet backbone from torchvision.

Usage example:
    python -u src/train_swag_classification.py --dataset chest_xray --epochs 5 --swag_start 3 --batch_size 8 --device cpu --output_dir runs/classification/swag_classification

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

from data_utils_classification import get_classification_loaders
from swag import SWAG, SWAGScheduler


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

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}'})

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

    val_loss = running_loss / total
    val_acc = 100. * correct / total
    return val_loss, val_acc


def build_resnet(arch: str, num_classes: int, pretrained: bool = False):
    if arch == 'resnet18':
        base = models.resnet18(pretrained=pretrained)
    elif arch == 'resnet34':
        base = models.resnet34(pretrained=pretrained)
    elif arch == 'resnet50':
        base = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f'Unknown arch: {arch}')

    num_features = base.fc.in_features
    base.fc = nn.Linear(num_features, num_classes)
    return base


def main():
    parser = argparse.ArgumentParser(description='Train classification SWAG')
    parser.add_argument('--dataset', type=str, default='chest_xray')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='runs/classification/swag_classification')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--swag_start', type=int, default=15)
    parser.add_argument('--swag_lr', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--max_models', type=int, default=20)
    parser.add_argument('--collect_freq', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--swag_scale', type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    print('\n' + '='*60)
    print('SWAG (classification) training')
    print('='*60)
    print(f"Dataset: {args.dataset}, Arch: {args.arch}, Epochs: {args.epochs}")
    print('='*60 + '\n')

    # Data loaders
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Build model
    base_model = build_resnet(args.arch, num_classes, pretrained=False)
    base_model = base_model.to(device)

    # Wrap with SWAG
    swag_model = SWAG(base_model, max_num_models=args.max_models)
    swag_model.to(device)
    swag_scheduler = SWAGScheduler(swag_model, collect_start_epoch=args.swag_start, collect_freq=args.collect_freq)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=args.lr)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'swag_epochs': []}
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        if epoch == args.swag_start:
            print(f"Switching to SWAG collection at epoch {epoch}. LR {args.lr} -> {args.swag_lr}")
            for g in optimizer.param_groups:
                g['lr'] = args.swag_lr

        train_loss, train_acc = train_epoch(base_model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(base_model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Train: {train_loss:.4f} | {train_acc:.2f}%  Val: {val_loss:.4f} | {val_acc:.2f}%")

        # SWAG collection
        swag_scheduler.step(epoch, base_model)
        if epoch >= args.swag_start:
            history['swag_epochs'].append(epoch)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best base model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_base_model.pth'))
            print(f"Saved best base model (val_acc: {val_acc:.2f}%)")

    # After training, save SWAG checkpoint
    swag_checkpoint = {
        'n_models': swag_model.n_models,
        'mean': swag_model.mean.cpu() if isinstance(swag_model.mean, torch.Tensor) else swag_model.mean,
        'sq_mean': swag_model.sq_mean.cpu() if isinstance(swag_model.sq_mean, torch.Tensor) else swag_model.sq_mean,
        'cov_mat_sqrt': [c.cpu() for c in swag_model.cov_mat_sqrt],
        'max_num_models': swag_model.max_num_models,
        'config': {
            'arch': args.arch,
            'num_classes': num_classes,
            'swag_start': args.swag_start,
            'swag_lr': args.swag_lr
        }
    }
    torch.save(swag_checkpoint, os.path.join(args.output_dir, 'swag_model.pth'))

    # Save history and config
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    config = {
        'epochs': args.epochs,
        'swag_start': args.swag_start,
        'lr': args.lr,
        'swag_lr': args.swag_lr,
        'batch_size': args.batch_size,
        'max_models': args.max_models,
        'best_val_acc': best_val_acc
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print('\nSWAG classification training complete. Outputs saved to', args.output_dir)


if __name__ == '__main__':
    main()
