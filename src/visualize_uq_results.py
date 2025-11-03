"""
Load training histories and evaluation results, then plot comparison charts.

Produces:
 - runs/classification/visualizations/accuracy_loss_comparison.png
 - runs/classification/visualizations/uq_metrics_comparison.png

Usage:
    python src/visualize_uq_results.py --results_dir runs/classification

"""
import os
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np


def find_histories(root: Path):
    histories = {}
    for p in root.glob('**/history.json'):
        try:
            data = json.loads(p.read_text())
            # infer method name from path pieces
            parts = p.parts
            # e.g., runs/classification/baseline/history.json or runs/classification/ensemble/member_0/history.json
            if 'ensemble' in parts:
                method = 'Ensemble'
            elif 'swag' in parts or 'swag_classification' in parts:
                method = 'SWAG'
            elif 'mc_dropout' in parts:
                method = 'MC Dropout'
            elif 'baseline' in parts:
                method = 'Baseline'
            else:
                method = parts[-2]

            # If multiple histories for same method, keep as list
            histories.setdefault(method, []).append((p, data))
        except Exception:
            continue
    return histories


def plot_accuracy_loss(histories, out_dir: Path):
    plt.figure(figsize=(12, 6))

    # Plot validation accuracy for each method
    plt.subplot(1, 2, 1)
    for method, entries in histories.items():
        # pick the first history for method
        _, data = entries[0]
        if 'val_acc' in data:
            plt.plot(data['val_acc'], label=method)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy by Method')
    plt.legend()

    # Plot validation loss
    plt.subplot(1, 2, 2)
    for method, entries in histories.items():
        _, data = entries[0]
        if 'val_loss' in data:
            plt.plot(data['val_loss'], label=method)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss by Method')
    plt.legend()

    out_file = out_dir / 'accuracy_loss_comparison.png'
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_uq_metrics(results_path: Path, out_dir: Path):
    if not results_path.exists():
        return
    data = json.loads(results_path.read_text())

    methods = []
    accuracies = []
    eces = []
    briers = []
    for r in data:
        methods.append(r.get('method', 'unknown'))
        accuracies.append(r.get('accuracy', np.nan))
        eces.append(r.get('ece', np.nan))
        briers.append(r.get('brier_score', np.nan))

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.bar(x, accuracies, width)
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.title('Accuracy (%)')

    plt.subplot(1, 3, 2)
    plt.bar(x, eces, width, color='orange')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.title('ECE')

    plt.subplot(1, 3, 3)
    plt.bar(x, briers, width, color='green')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.title('Brier Score')

    plt.tight_layout()
    out_file = out_dir / 'uq_metrics_comparison.png'
    plt.savefig(out_file)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='runs/classification')
    args = parser.parse_args()

    root = Path(args.results_dir)
    out_dir = root / 'visualizations'
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = find_histories(root)
    plot_accuracy_loss(histories, out_dir)

    results_path = root / 'evaluation' / 'all_results.json'
    plot_uq_metrics(results_path, out_dir)

    print('Visualizations saved to', out_dir)


if __name__ == '__main__':
    main()
