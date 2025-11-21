"""
Create Publication-Quality Workflow Flowchart

Minimal, professional visualization following Maddox et al. style.
No verbose text, clean lines, publication-ready.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set publication style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
color_data = '#E8F4F8'  # Light blue
color_train = '#FFE6E6'  # Light red
color_eval = '#E6F3E6'  # Light green
color_conformal = '#FFF4E6'  # Light orange
color_viz = '#F0E6FF'  # Light purple
color_thesis = '#FFE6F0'  # Light pink

def add_box(ax, x, y, width, height, text, color, fontsize=10, bold=False):
    """Add a rounded rectangle box with text"""
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color,
                          linewidth=2)
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            weight=weight, wrap=True)

def add_arrow(ax, x1, y1, x2, y2, label='', style='->', width=2):
    """Add an arrow between boxes"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color='black',
                           linewidth=width, mutation_scale=20)
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y, label, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='none', alpha=0.8))

# Title
ax.text(5, 11.5, 'Uncertainty Quantification Workflow', 
        ha='center', fontsize=18, weight='bold')
ax.text(5, 11.0, 'Chest X-Ray Pneumonia Detection with Conformal Prediction',
        ha='center', fontsize=14, style='italic')

# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================
add_box(ax, 0.5, 9.5, 2.0, 0.8, 'Raw Data\n5,840 Chest X-Rays\n(1,583 Normal, 4,273 Pneumonia)',
        color_data, fontsize=10, bold=True)

add_arrow(ax, 1.5, 9.5, 1.5, 9.0, '')

add_box(ax, 0.5, 8.2, 2.0, 0.7, 'Data Split\nTrain: 4,172 (71.4%)\nCal: 1,044 (17.9%)\nTest: 624 (10.7%)',
        color_data, fontsize=9)

add_arrow(ax, 1.5, 8.2, 1.5, 7.7, '')

add_box(ax, 0.5, 6.9, 2.0, 0.7, 'Preprocessing\n• Resize: 224×224\n• ImageNet normalization\n• Augmentation (train only)',
        color_data, fontsize=9)

# ============================================================================
# PHASE 2: TRAINING (4 PARALLEL PATHS)
# ============================================================================
add_arrow(ax, 1.5, 6.9, 1.5, 6.4, '')
add_box(ax, 0.2, 5.6, 2.6, 0.7, 'TRAINING: 4 UQ Methods × 2 Durations',
        color_train, fontsize=11, bold=True)

# Baseline path
add_arrow(ax, 0.5, 5.6, 0.5, 5.0, '')
add_box(ax, 0.1, 4.0, 0.8, 0.9, 'Baseline\nResNet-50\nSGD\n50/300 ep',
        color_train, fontsize=8)

# MC Dropout path
add_arrow(ax, 1.0, 5.6, 1.0, 5.0, '')
add_box(ax, 0.95, 4.0, 0.8, 0.9, 'MC Dropout\nDropout p=0.3\nSGD\n50/300 ep',
        color_train, fontsize=8)

# Ensemble path
add_arrow(ax, 1.5, 5.6, 1.5, 5.0, '')
add_box(ax, 1.4, 4.0, 0.8, 0.9, 'Ensemble\n5 members\nSGD\n50/300 ep',
        color_train, fontsize=8)

# SWAG path
add_arrow(ax, 2.2, 5.6, 2.2, 5.0, '')
add_box(ax, 1.85, 4.0, 0.8, 0.9, 'SWAG\nCollect 27-50\nor 161-300\nSGD',
        color_train, fontsize=8)

# ============================================================================
# PHASE 3: INITIAL EVALUATION
# ============================================================================
# Converge arrows
add_arrow(ax, 0.5, 4.0, 1.5, 3.5, '')
add_arrow(ax, 1.0, 4.0, 1.5, 3.5, '')
add_arrow(ax, 1.8, 4.0, 1.5, 3.5, '')
add_arrow(ax, 2.2, 4.0, 1.5, 3.5, '')

add_box(ax, 0.5, 2.7, 2.0, 0.7, 'Initial Evaluation\n• Training/validation curves\n• Test accuracy\n• Save checkpoints',
        color_eval, fontsize=9)

# ============================================================================
# PHASE 4: CONFORMAL PREDICTION
# ============================================================================
add_arrow(ax, 2.5, 3.0, 3.5, 3.0, '')

add_box(ax, 3.6, 2.7, 2.0, 0.7, 'Conformal Calibration\n• Adaptive Prediction Sets (APS)\n• α = 0.1 (90% coverage)\n• n_cal = 1,044 samples',
        color_conformal, fontsize=9)

add_arrow(ax, 4.6, 2.7, 4.6, 2.2, '')

add_box(ax, 3.6, 1.4, 2.0, 0.7, 'Conformal Evaluation\n• Coverage rate\n• Set size distribution\n• Singleton percentage',
        color_conformal, fontsize=9)

# ============================================================================
# PHASE 5: COMPREHENSIVE ANALYSIS
# ============================================================================
add_arrow(ax, 5.6, 3.0, 6.5, 3.0, '')

add_box(ax, 6.6, 2.0, 2.5, 1.8, 'Analysis & Metrics\n\n• Accuracy & F1 score\n• ECE & Brier score\n• Coverage guarantees\n• Training dynamics\n• Overfitting analysis\n• Method comparison',
        color_eval, fontsize=9)

# ============================================================================
# PHASE 6: VISUALIZATION
# ============================================================================
add_arrow(ax, 7.3, 2.0, 7.3, 1.5, '')

add_box(ax, 6.6, 0.2, 2.5, 1.2, 'Visualizations\n\n• Figure 4: Training dynamics\n• Figure 5: Conformal comparison\n• Calibration plots\n• Set size distributions',
        color_viz, fontsize=9)

# ============================================================================
# PHASE 7: THESIS WRITING
# ============================================================================
add_arrow(ax, 9.1, 2.9, 9.5, 2.9, '')

add_box(ax, 0.3, 0.2, 1.6, 1.2, 'DELIVERABLES\n\n• Thesis sections\n  4.2, 4.6, 4.7\n• 2 figures\n• Complete analysis\n• Production-ready',
        color_thesis, fontsize=9, bold=True)

# ============================================================================
# KEY FINDINGS (Right side)
# ============================================================================
add_box(ax, 3.2, 5.8, 6.5, 3.6, '', 'white')  # Background box

ax.text(6.5, 9.0, 'KEY FINDINGS', ha='center', fontsize=14, weight='bold')

findings = [
    "1. MC Dropout achieves best validation accuracy (91.35%)",
    "   → Dropout regularization prevents early overfitting",
    "",
    "2. SWAG underperforms on small datasets (89.42%)",
    "   → Best validation at epoch 1, before collection window",
    "   → Requires ≥10-15K samples per class",
    "",
    "3. All methods severely overfit by epoch 10",
    "   → Training accuracy: 99.98-100%",
    "   → Validation accuracy: 89-91%",
    "",
    "4. Conformal prediction validated (89-91% coverage)",
    "   → All methods meet 90% target",
    "   → 100% singleton sets (model overconfidence)",
    "",
    "5. Deep Ensemble best calibration (ECE = 0.027)",
    "   → Initialization diversity reduces overconfidence"
]

y_pos = 8.5
for finding in findings:
    ax.text(3.5, y_pos, finding, fontsize=9, va='top', family='monospace')
    y_pos -= 0.22

# ============================================================================
# EXPERIMENTAL CONFIGURATIONS (Bottom right)
# ============================================================================
add_box(ax, 3.2, 0.2, 6.5, 1.6, '', 'white')

ax.text(6.5, 1.6, 'EXPERIMENTAL CONFIGURATIONS', ha='center', fontsize=12, weight='bold')

configs = [
    "SGD 50-Epoch (COMPLETE ✅):  Baseline, MC Dropout, Ensemble, SWAG",
    "  • Learning rate: 0.01 → 0.005 (cosine)",
    "  • Momentum: 0.9, Weight decay: 1e-4",
    "  • SWAG collection: Epochs 27-50 (final 46%)",
    "",
    "SGD 300-Epoch (IN PROGRESS 🔄):  Baseline ✅, SWAG ✅, MC Dropout 🔄, Ensemble 🔄",
    "  • Learning rate: 0.01 → 0.005 (cosine)",
    "  • SWAG collection: Epochs 161-300 (final 46%)",
    "  • Tests: Does extended training prevent rapid overfitting?"
]

y_pos = 1.35
for config in configs:
    ax.text(3.4, y_pos, config, fontsize=8, va='top', family='monospace')
    y_pos -= 0.16

# ============================================================================
# LEGEND
# ============================================================================
legend_y = 11.8
legend_items = [
    ('Data Preparation', color_data),
    ('Training', color_train),
    ('Evaluation', color_eval),
    ('Conformal Prediction', color_conformal),
    ('Visualization', color_viz),
    ('Thesis', color_thesis)
]

legend_x = 0.5
for label, color in legend_items:
    box = mpatches.Rectangle((legend_x, legend_y), 0.25, 0.15,
                             facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(box)
    ax.text(legend_x + 0.32, legend_y + 0.075, label, fontsize=8, va='center')
    legend_x += 1.6

# Add timestamp
ax.text(9.5, 0.05, f'Generated: November 20, 2025', 
        ha='right', fontsize=8, style='italic', color='gray')

plt.tight_layout()
plt.savefig('results/visualizations/workflow_flowchart.png', dpi=300, bbox_inches='tight')
print("✅ Workflow flowchart saved to results/visualizations/workflow_flowchart.png")
plt.show()
