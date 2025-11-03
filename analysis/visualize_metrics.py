#!/usr/bin/env python3
"""
Visualization of Comprehensive UQ Metrics
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load results
results_dir = Path('runs/classification/metrics')
results_file = results_dir / 'comprehensive_metrics.json'

with open(results_file, 'r') as f:
    results = json.load(f)

# Create figure with subplots
fig = plt.figure(figsize=(18, 14))

# ============ Plot 1: Accuracy Comparison ============
ax1 = plt.subplot(3, 3, 1)
methods = []
accuracies = []
for method, data in results.items():
    if method != 'Conformal Risk Control':
        if isinstance(data, list):
            continue
        methods.append(method)
        accuracies.append(data.get('accuracy', 0))

colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
bars = ax1.bar(methods, accuracies, color=colors[:len(methods)])
ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 100])
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============ Plot 2: Calibration (ECE) ============
ax2 = plt.subplot(3, 3, 2)
eces = []
for method, data in results.items():
    if method != 'Conformal Risk Control':
        if isinstance(data, list):
            continue
        methods_ece = method
        eces.append(data.get('ece', 0))

bars = ax2.bar(methods, eces, color=colors[:len(methods)])
ax2.set_ylabel('ECE (Expected Calibration Error)', fontsize=11, fontweight='bold')
ax2.set_title('Calibration: ECE Comparison', fontsize=12, fontweight='bold')
ax2.axhline(y=0.1, color='r', linestyle='--', label='Acceptable (0.1)', linewidth=2)
ax2.legend()
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=9)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============ Plot 3: Brier Score ============
ax3 = plt.subplot(3, 3, 3)
briers = []
for method, data in results.items():
    if method != 'Conformal Risk Control':
        if isinstance(data, list):
            continue
        brier_str = data.get('brier_score', '0')
        if isinstance(brier_str, str):
            brier = float(brier_str)
        else:
            brier = brier_str
        briers.append(brier)

bars = ax3.bar(methods, briers, color=colors[:len(methods)])
ax3.set_ylabel('Brier Score', fontsize=11, fontweight='bold')
ax3.set_title('Calibration: Brier Score', fontsize=12, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=9)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============ Plot 4: FPR/FNR ============
ax4 = plt.subplot(3, 3, 4)
fprs = []
fnrs = []
for method, data in results.items():
    if method != 'Conformal Risk Control':
        if isinstance(data, list):
            continue
        fprs.append(data.get('fpr', 0))
        fnrs.append(data.get('fnr', 0))

x_pos = np.arange(len(methods))
width = 0.35
ax4.bar(x_pos - width/2, fprs, width, label='FPR', color='#e74c3c', alpha=0.8)
ax4.bar(x_pos + width/2, fnrs, width, label='FNR', color='#3498db', alpha=0.8)
ax4.set_ylabel('Error Rate', fontsize=11, fontweight='bold')
ax4.set_title('Error Rates: FPR vs FNR', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(methods, rotation=45, ha='right')
ax4.legend()
ax4.set_ylim([0, 0.5])

# ============ Plot 5: ROC-AUC ============
ax5 = plt.subplot(3, 3, 5)
aucs = []
for method, data in results.items():
    if method != 'Conformal Risk Control':
        if isinstance(data, list):
            continue
        aucs.append(data.get('roc_auc', 0))

bars = ax5.bar(methods, aucs, color=colors[:len(methods)])
ax5.set_ylabel('ROC-AUC Score', fontsize=11, fontweight='bold')
ax5.set_title('Discrimination: ROC-AUC', fontsize=12, fontweight='bold')
ax5.set_ylim([0, 1.0])
ax5.axhline(y=0.5, color='r', linestyle='--', linewidth=2, alpha=0.5)
for bar in bars:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=9)
plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============ Plot 6: Mean Uncertainty ============
ax6 = plt.subplot(3, 3, 6)
uncs = []
methods_with_unc = []
for method, data in results.items():
    if method != 'Conformal Risk Control':
        if isinstance(data, list):
            continue
        if 'mean_uncertainty' in data:
            methods_with_unc.append(method)
            uncs.append(data.get('mean_uncertainty', 0))

if uncs:
    bars = ax6.bar(methods_with_unc, uncs, color=colors[:len(methods_with_unc)])
    ax6.set_ylabel('Mean Uncertainty', fontsize=11, fontweight='bold')
    ax6.set_title('Uncertainty Magnitude', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.6f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============ Plot 7: Uncertainty Separation ============
ax7 = plt.subplot(3, 3, 7)
seps = []
methods_with_sep = []
for method, data in results.items():
    if method != 'Conformal Risk Control':
        if isinstance(data, list):
            continue
        if 'unc_separation' in data:
            methods_with_sep.append(method)
            seps.append(data.get('unc_separation', 0))

if seps:
    bars = ax7.bar(methods_with_sep, seps, color=colors[:len(methods_with_sep)])
    ax7.set_ylabel('Uncertainty Separation', fontsize=11, fontweight='bold')
    ax7.set_title('Uncertainty Quality (Higher=Better)', fontsize=12, fontweight='bold')
    ax7.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.6f}', ha='center', va='bottom', fontsize=9)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ============ Plot 8: Conformal Risk Control ============
ax8 = plt.subplot(3, 3, 8)
if 'Conformal Risk Control' in results:
    crc_results = results['Conformal Risk Control']
    crc_names = []
    empirical_risks = []
    coverage = []
    
    for res in crc_results:
        crc_names.append(res['method'].replace('CRC - ', ''))
        empirical_risks.append(res.get('empirical_risk', 0))
        coverage.append(res.get('coverage', 0))
    
    x_pos = np.arange(len(crc_names))
    width = 0.35
    ax8.bar(x_pos - width/2, empirical_risks, width, label='Empirical Risk', color='#e74c3c', alpha=0.8)
    ax8.bar(x_pos + width/2, coverage, width, label='Coverage', color='#2ecc71', alpha=0.8)
    ax8.set_ylabel('Rate', fontsize=11, fontweight='bold')
    ax8.set_title('Conformal Risk Control Performance', fontsize=12, fontweight='bold')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(crc_names, rotation=45, ha='right', fontsize=9)
    ax8.legend()
    ax8.set_ylim([0, 1.1])

# ============ Plot 9: Summary Table ============
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Create summary data
summary_data = []
for method, data in results.items():
    if method != 'Conformal Risk Control':
        if isinstance(data, list):
            continue
        summary_data.append({
            'Method': method,
            'Acc %': f"{data.get('accuracy', 0):.1f}",
            'ECE': f"{data.get('ece', 0):.4f}",
            'FNR': f"{data.get('fnr', 0):.4f}"
        })

summary_df = pd.DataFrame(summary_data)
table = ax9.table(cellText=summary_df.values, colLabels=summary_df.columns,
                  cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.suptitle('Comprehensive UQ Metrics Evaluation - 50 Epoch Classification Models', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
output_path = results_dir / 'comprehensive_metrics_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

# Also save individual method comparison plot
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy vs Calibration
ax = axes[0, 0]
for method, data in results.items():
    if method != 'Conformal Risk Control' and not isinstance(data, list):
        ax.scatter(data.get('ece', 0), data.get('accuracy', 0), s=200, alpha=0.7)
        ax.annotate(method, (data.get('ece', 0), data.get('accuracy', 0)), 
                   fontsize=10, fontweight='bold', ha='center', va='center')
ax.set_xlabel('ECE (Calibration Error)', fontsize=11, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax.set_title('Accuracy vs Calibration Tradeoff', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# FNR vs FPR
ax = axes[0, 1]
for method, data in results.items():
    if method != 'Conformal Risk Control' and not isinstance(data, list):
        ax.scatter(data.get('fpr', 0), data.get('fnr', 0), s=200, alpha=0.7)
        ax.annotate(method, (data.get('fpr', 0), data.get('fnr', 0)), 
                   fontsize=10, fontweight='bold', ha='center', va='center')
ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax.set_ylabel('False Negative Rate', fontsize=11, fontweight='bold')
ax.set_title('Error Rate Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Brier Score vs ROC-AUC
ax = axes[1, 0]
for method, data in results.items():
    if method != 'Conformal Risk Control' and not isinstance(data, list):
        brier_str = data.get('brier_score', '0')
        brier = float(brier_str) if isinstance(brier_str, str) else brier_str
        ax.scatter(brier, data.get('roc_auc', 0), s=200, alpha=0.7)
        ax.annotate(method, (brier, data.get('roc_auc', 0)), 
                   fontsize=10, fontweight='bold', ha='center', va='center')
ax.set_xlabel('Brier Score (Lower Better)', fontsize=11, fontweight='bold')
ax.set_ylabel('ROC-AUC Score', fontsize=11, fontweight='bold')
ax.set_title('Calibration vs Discrimination', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Uncertainty metrics
ax = axes[1, 1]
methods_unc = []
uncs_mean = []
seps = []
for method, data in results.items():
    if method != 'Conformal Risk Control' and not isinstance(data, list):
        if 'mean_uncertainty' in data and 'unc_separation' in data:
            methods_unc.append(method)
            uncs_mean.append(data.get('mean_uncertainty', 0))
            seps.append(data.get('unc_separation', 0))

if methods_unc:
    ax.scatter(uncs_mean, seps, s=200, alpha=0.7)
    for i, method in enumerate(methods_unc):
        ax.annotate(method, (uncs_mean[i], seps[i]), 
                   fontsize=10, fontweight='bold', ha='center', va='center')
ax.set_xlabel('Mean Uncertainty', fontsize=11, fontweight='bold')
ax.set_ylabel('Uncertainty Separation', fontsize=11, fontweight='bold')
ax.set_title('Uncertainty Quality Metrics', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('Method Comparisons', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save second figure
output_path2 = results_dir / 'method_comparisons.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved to: {output_path2}")

print("\n✓ All visualizations generated successfully!")
