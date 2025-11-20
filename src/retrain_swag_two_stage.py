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

from data_utils_classification import get_classification_loaders
from swag import SWAG
from torch.optim.swa_utils import update_bn


"""
Two-stage SWAG retraining script.
- Stage 1: cyclic learning rate for exploration
- Stage 2: fixed low LR and snapshot collection

This script is intentionally a runnable skeleton using PyTorch.
It saves snapshot checkpoints during the collection phase so that
you can post-process them with `swa_gaussian` utilities or your own tool.

Usage example:
  python src/retrain_swag_two_stage.py --baseline_path runs/classification/baseline/best_model.pth \
    --epochs 50 --collection_start 21 --snap_interval 1 --save_dir runs/classification/swag_two_stage

Note: This script will not automatically compute SWAG posterior; it creates snapshots.
Use `swa_gaussian` or provided postprocessing code to convert snapshots to a SWAG model.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def get_dataloaders(data_dir, batch_size=64, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    # expects ImageFolder layout: train/val/test
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size), DataLoader(test_ds, batch_size=batch_size)


def adjust_learning_rate_cyclic(optimizer, base_lr, max_lr, epoch, cycle_length):
    # triangular cyclic lr between base_lr and max_lr
    cycle_pos = (epoch - 1) % cycle_length
    factor = cycle_pos / max(1, (cycle_length - 1))
    # triangular: rise then fall
    lr = base_lr + (max_lr - base_lr) * (1 - abs(2 * factor - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return running_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', help='Dataset directory with train/val/test subfolders')
    parser.add_argument('--baseline_path', default=None, help='Optional pretrained baseline to load')
    parser.add_argument('--arch', default='resnet18')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--collection_start', type=int, default=21)
    parser.add_argument('--snap_interval', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--max_lr', type=float, default=1e-3)
    parser.add_argument('--cycle_length', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', default='runs/classification/swag_two_stage')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # model
    if args.arch == 'resnet18':
        model = models.resnet18(pretrained=False)
        # Keep flexibility: do not hardcode num classes; detect from dataset
    else:
        model = models.resnet18(pretrained=False)

    model = model.to(device)

    # If baseline provided, attempt to load weights
    if args.baseline_path is not None and os.path.exists(args.baseline_path):
        try:
            print(f"Loading baseline from {args.baseline_path}")
            state = torch.load(args.baseline_path, map_location='cpu')
            try:
                model.load_state_dict(state)
            except Exception:
                if isinstance(state, dict) and 'state_dict' in state:
                    model.load_state_dict(state['state_dict'])
                else:
                    print("Baseline state incompatible; continuing without strict load")
        except Exception as e:
            print(f"Warning: failed to load baseline: {e}")

    # Update classifier head to match dataset if possible
    # Attempt to detect number of classes from data folder
    try:
        train_root = os.path.join(args.data_dir, 'train')
        if os.path.isdir(train_root):
            classes = [d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d))]
            n_classes = max(2, len(classes))
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, n_classes)
            model = model.to(device)
    except Exception:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)

    train_loader, val_loader, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        if epoch <= args.collection_start:
            # exploration phase: cyclic lr
            lr = adjust_learning_rate_cyclic(optimizer, args.base_lr, args.max_lr, epoch, args.cycle_length)
        else:
            # fixed lr during collection
            for pg in optimizer.param_groups:
                pg['lr'] = args.base_lr
            lr = args.base_lr

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{args.epochs} | lr={lr:.6f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        # Snapshot collection during collection phase
        if epoch >= args.collection_start and ((epoch - args.collection_start) % args.snap_interval == 0):
            snap_path = os.path.join(args.save_dir, f"snapshot_epoch_{epoch}.pth")
            torch.save(model.state_dict(), snap_path)
            print(f"Saved snapshot {snap_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    main()
    
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
