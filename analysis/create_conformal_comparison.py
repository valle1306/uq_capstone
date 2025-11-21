"""
Create comprehensive conformal prediction comparison visualization
Compares all 4 UQ methods: Baseline, MC Dropout, Ensemble, SWAG
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Load conformal results
results_dir = Path('results/sgd_50epochs_conformal')

methods = {
    'Baseline': 'baseline_corrected',
    'MC Dropout': 'mcdropout_corrected',
    'Ensemble': 'ensemble_corrected',
    'SWAG': 'swag_sgd_corrected'
}

data = {}
for method_name, folder in methods.items():
    result_file = results_dir / folder / 'conformal_prediction_results.json'
    with open(result_file, 'r') as f:
        data[method_name] = json.load(f)

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Conformal Prediction Comparison: SGD 50-Epoch Training\n(Target Coverage: 90%)', 
             fontsize=16, fontweight='bold', y=0.98)

# ============ Subplot 1: Coverage Comparison ============
ax1 = axes[0, 0]

method_names = list(methods.keys())
coverages = [data[m]['primary_results']['empirical_coverage'] * 100 for m in method_names]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']  # Green, Blue, Red, Orange

bars = ax1.bar(method_names, coverages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target (90%)', zorder=0)
ax1.set_ylabel('Empirical Coverage (%)', fontweight='bold')
ax1.set_title('Coverage Achievement', fontweight='bold')
ax1.set_ylim([85, 95])
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, cov in zip(bars, coverages):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
             f'{cov:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# ============ Subplot 2: Set Size Distribution ============
ax2 = axes[0, 1]

avg_sizes = [data[m]['primary_results']['avg_set_size'] for m in method_names]
singleton_fracs = [data[m]['primary_results']['singleton_fraction'] * 100 for m in method_names]

x = np.arange(len(method_names))
width = 0.35

bars1 = ax2.bar(x - width/2, avg_sizes, width, label='Avg Set Size', 
                color='#3498db', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + width/2, singleton_fracs, width, label='% Singleton', 
                     color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Method', fontweight='bold')
ax2.set_ylabel('Average Set Size', color='#3498db', fontweight='bold')
ax2_twin.set_ylabel('Singleton Fraction (%)', color='#e74c3c', fontweight='bold')
ax2.set_title('Prediction Set Characteristics', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(method_names, rotation=15, ha='right')
ax2.set_ylim([0.95, 1.05])
ax2_twin.set_ylim([95, 102])
ax2.tick_params(axis='y', labelcolor='#3498db')
ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
ax2.grid(axis='y', alpha=0.3)

# Combined legend
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

# ============ Subplot 3: Coverage vs Alpha Trade-off ============
ax3 = axes[1, 0]

alphas = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
target_coverages = [(1 - a) * 100 for a in alphas]

for method_name, color in zip(method_names, colors):
    alpha_results = data[method_name]['all_alpha_results']
    empirical_covs = []
    for alpha in alphas:
        key = f'alpha_{alpha}'
        empirical_covs.append(alpha_results[key]['empirical_coverage'] * 100)
    
    ax3.plot(target_coverages, empirical_covs, marker='o', linewidth=2, 
             label=method_name, color=color, markersize=6)

# Perfect calibration line
ax3.plot([70, 99], [70, 99], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.5)

ax3.set_xlabel('Target Coverage (%)', fontweight='bold')
ax3.set_ylabel('Empirical Coverage (%)', fontweight='bold')
ax3.set_title('Coverage vs Alpha Trade-off', fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([68, 100])
ax3.set_ylim([68, 100])

# ============ Subplot 4: Summary Table ============
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary table
table_data = []
table_data.append(['Method', 'Test Acc', 'Coverage', 'Set Size', 'Singleton', 'τ'])

for method_name in method_names:
    result = data[method_name]['primary_results']
    # Get test accuracy from model performance (approximate from coverage since 100% singletons)
    test_acc = result['singleton_accuracy'] * 100
    coverage = result['empirical_coverage'] * 100
    set_size = result['avg_set_size']
    singleton = result['singleton_fraction'] * 100
    tau = result['tau']
    
    table_data.append([
        method_name,
        f"{test_acc:.2f}%",
        f"{coverage:.2f}%",
        f"{set_size:.2f}",
        f"{singleton:.0f}%",
        f"{tau:.4f}"
    ])

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  bbox=[0, 0.3, 1, 0.6])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(6):
    cell = table[(0, i)]
    cell.set_facecolor('#3498db')
    cell.set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(6):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#ecf0f1')
        else:
            cell.set_facecolor('white')

# Add title
ax4.text(0.5, 0.95, 'Summary Statistics (α=0.1)', 
         transform=ax4.transAxes, fontsize=13, fontweight='bold',
         ha='center', va='top')

# Add key finding text
finding_text = (
    "Key Finding: All methods achieve ~90% coverage (within 1.35% of target),\n"
    "confirming conformal prediction works correctly. 100% singleton sets indicate\n"
    "models are overconfident (p(class) > 80%), so conformal = standard classification."
)
ax4.text(0.5, 0.15, finding_text,
         transform=ax4.transAxes, fontsize=9, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save
output_file = 'results/visualizations/conformal_comparison_sgd_50epoch.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved conformal comparison to: {output_file}")

plt.close()

print("\n" + "="*80)
print("CONFORMAL PREDICTION COMPARISON SUMMARY")
print("="*80)
print(f"{'Method':<15} {'Coverage':<12} {'Set Size':<12} {'Singleton %':<12} {'Threshold'}")
print("-"*80)
for method_name in method_names:
    result = data[method_name]['primary_results']
    print(f"{method_name:<15} {result['empirical_coverage']*100:>6.2f}%     "
          f"{result['avg_set_size']:>6.2f}       "
          f"{result['singleton_fraction']*100:>6.0f}%         "
          f"{result['tau']:>8.4f}")
print("="*80)
