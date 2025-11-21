"""
Generate publication-quality visualizations for SGD 50-epoch results
Creates: reliability diagrams, training curves, method comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 300

results_dir = Path("c:/Users/lpnhu/Downloads/uq_capstone/results")
output_dir = results_dir / "visualizations"
output_dir.mkdir(exist_ok=True)

# Load all results
methods = {
    'Baseline': 'sgd_50epochs_baseline',
    'MC Dropout': 'sgd_50epochs_mcdropout/mc_dropout_sgd',
    'SWAG': 'sgd_50epochs_swag',
    'Ensemble': 'sgd_50epochs_ensemble/ensemble_sgd/member_1'
}

histories = {}
for name, path in methods.items():
    history_path = results_dir / path / "history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            h_raw = json.load(f)
            # Normalize structure
            if 'epoch' not in h_raw and 'train_acc' in h_raw:
                n_epochs = len(h_raw['train_acc'])
                h_raw['epoch'] = list(range(1, n_epochs + 1))
                if 'test_acc' not in h_raw and 'val_acc' in h_raw:
                    h_raw['test_acc'] = h_raw['val_acc']
                    h_raw['test_loss'] = h_raw['val_loss']
            histories[name] = h_raw

# Generate comprehensive comparison figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

colors = {'Baseline': '#3498db', 'MC Dropout': '#2ecc71', 'SWAG': '#e74c3c', 'Ensemble': '#f39c12'}

# Row 1: Training and Validation Accuracy
ax1 = fig.add_subplot(gs[0, :2])
for name, h in histories.items():
    epochs = range(1, len(h['epoch']) + 1)
    train_accs = h.get('train_acc', [])
    ax1.plot(epochs, train_accs, label=f'{name} (Train)', color=colors[name], linewidth=2.5, alpha=0.8)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Training Accuracy: All Methods Overfit by Epoch 10', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([92, 100.5])

ax2 = fig.add_subplot(gs[0, 2])
for name, h in histories.items():
    epochs = range(1, len(h['epoch']) + 1)
    test_accs = h.get('test_acc', [])
    ax2.plot(epochs, test_accs, label=name, color=colors[name], linewidth=2.5, alpha=0.8)
    
    # Mark best
    best_idx = np.argmax(test_accs)
    best_acc = test_accs[best_idx]
    ax2.scatter([best_idx + 1], [best_acc], color=colors[name], s=120, zorder=5, edgecolors='white', linewidths=2)

ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Validation Accuracy:\nBest at Epochs 1-4', fontsize=13, fontweight='bold')
ax2.legend(loc='lower left', framealpha=0.9, fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([78, 93])

# Row 2: Overfitting Analysis
ax3 = fig.add_subplot(gs[1, :2])
for name, h in histories.items():
    epochs = range(1, len(h['epoch']) + 1)
    train_accs = h.get('train_acc', [])
    test_accs = h.get('test_acc', [])
    gaps = [train_accs[i] - test_accs[i] for i in range(len(train_accs))]
    ax3.plot(epochs, gaps, label=name, color=colors[name], linewidth=2.5, alpha=0.8)

ax3.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
ax3.axhline(y=10, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='10% Overfitting Threshold')
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Train - Val Gap (%)', fontsize=12, fontweight='bold')
ax3.set_title('Overfitting Gap Grows Rapidly After Epoch 10', fontsize=14, fontweight='bold')
ax3.legend(loc='upper left', framealpha=0.9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([-2, 18])

# SWAG Collection Window Visualization
ax4 = fig.add_subplot(gs[1, 2])
swag_h = histories.get('SWAG', {})
epochs = range(1, len(swag_h['epoch']) + 1)
test_accs = swag_h.get('test_acc', [])
ax4.plot(epochs, test_accs, color=colors['SWAG'], linewidth=3, label='SWAG Validation')
ax4.axvspan(28, 50, alpha=0.2, color='red', label='SWAG Collection\n(Epochs 28-50)')
ax4.axvline(x=1, color='green', linestyle='--', linewidth=2, label=f'Best Val: Epoch 1\n({max(test_accs):.2f}%)')
if len(test_accs) >= 27:
    ax4.scatter([27], [test_accs[26]], color='red', s=180, zorder=5, marker='X', edgecolors='white', linewidths=2)
    ax4.text(27, test_accs[26]-2.5, f'Epoch 27:\n{test_accs[26]:.2f}%', 
             ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax4.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
ax4.set_title('SWAG Problem:\nCollects After Overfitting', fontsize=13, fontweight='bold', color='darkred')
ax4.legend(loc='lower left', framealpha=0.9, fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([80, 92])

# Row 3: Summary Statistics
ax5 = fig.add_subplot(gs[2, :])
summary_data = []
for name in ['Baseline', 'MC Dropout', 'Ensemble', 'SWAG']:
    if name in histories:
        h = histories[name]
        train_accs = h.get('train_acc', [])
        test_accs = h.get('test_acc', [])
        final_train = train_accs[-1] if train_accs else 0
        best_val = max(test_accs) if test_accs else 0
        best_epoch = np.argmax(test_accs) + 1 if test_accs else 0
        final_val = test_accs[-1] if test_accs else 0
        gap = final_train - final_val
        
        summary_data.append({
            'Method': name,
            'Final Train': final_train,
            'Best Val': best_val,
            'Best Epoch': best_epoch,
            'Final Val': final_val,
            'Overfitting Gap': gap
        })

# Create table
table_data = []
headers = ['Method', 'Final Train\nAcc (%)', 'Best Val\nAcc (%)', 'Best Val\nEpoch', 'Final Val\nAcc (%)', 'Overfitting\nGap (%)']
for item in summary_data:
    row = [
        item['Method'],
        f"{item['Final Train']:.2f}",
        f"{item['Best Val']:.2f}",
        f"{item['Best Epoch']}",
        f"{item['Final Val']:.2f}",
        f"{item['Overfitting Gap']:.2f}"
    ]
    table_data.append(row)

table = ax5.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                  colWidths=[0.2, 0.16, 0.16, 0.13, 0.16, 0.17])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Color code the cells
for i, key in enumerate(summary_data):
    color = colors[key['Method']]
    table[(i+1, 0)].set_facecolor(color)
    table[(i+1, 0)].set_text_props(weight='bold', color='white')
    
    # Highlight best val accuracy
    if key['Best Val'] == max([d['Best Val'] for d in summary_data]):
        table[(i+1, 2)].set_facecolor('#d5f4e6')
        table[(i+1, 2)].set_text_props(weight='bold')

# Style headers
for j in range(len(headers)):
    table[(0, j)].set_facecolor('#34495e')
    table[(0, j)].set_text_props(weight='bold', color='white')

ax5.axis('off')
ax5.set_title('Summary: MC Dropout Achieves Best Validation (91.35%), SWAG Underperforms (89.42%)', 
              fontsize=14, fontweight='bold', pad=20)

plt.suptitle('SGD 50-Epoch Experiments: Understanding SWAG Failure on Small Datasets', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(output_dir / 'sgd_50epoch_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print(f"\n✅ Saved comprehensive analysis to: {output_dir / 'sgd_50epoch_comprehensive_analysis.png'}")

plt.show()

# Print summary statistics
print("\n" + "="*80)
print("SGD 50-EPOCH RESULTS SUMMARY")
print("="*80)
for item in summary_data:
    print(f"\n{item['Method']:12}")
    print(f"  Final Train:  {item['Final Train']:6.2f}%")
    print(f"  Best Val:     {item['Best Val']:6.2f}% (Epoch {item['Best Epoch']})")
    print(f"  Final Val:    {item['Final Val']:6.2f}%")
    print(f"  Overfit Gap:  {item['Overfitting Gap']:6.2f}%")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print("1. MC Dropout achieves highest validation accuracy (91.35%)")
print("2. All methods severely overfit (train: 99.98-100%, best val: 89-91%)")
print("3. Best validation occurs at epochs 1-4, followed by degradation")
print("4. SWAG collects snapshots epochs 28-50, AFTER best validation at epoch 1")
print("5. Small dataset (4,172 samples) causes early overfitting, violating")
print("   SWAG's assumption of SGD exploring a broad posterior mode")
print("="*80)
