"""
Publication-Quality Workflow Flowchart

Clean, minimal visualization following academic paper standards.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Color scheme - subtle, professional
color_data = '#E8F5E9'
color_train = '#E3F2FD'
color_eval = '#FFF9C4'
color_conformal = '#FFE0B2'

def add_box(ax, x, y, w, h, text, color, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, weight='normal')

def add_arrow(ax, x1, y1, x2, y2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                           color='black', linewidth=1.5, mutation_scale=15)
    ax.add_patch(arrow)

# Title
ax.text(5, 5.7, 'Experimental Pipeline', ha='center', fontsize=14, weight='bold')

# Row 1: Data
add_box(ax, 0.5, 4.8, 1.5, 0.6, 'Data\n(n=5,840)', color_data)
add_arrow(ax, 2.0, 5.1, 2.8, 5.1)
add_box(ax, 2.8, 4.8, 1.5, 0.6, 'Split\n(Train/Cal/Test)', color_data)
add_arrow(ax, 4.3, 5.1, 5.1, 5.1)
add_box(ax, 5.1, 4.8, 1.5, 0.6, 'Preprocess\n(224×224)', color_data)

# Row 2: Training methods
add_arrow(ax, 3.3, 4.8, 3.3, 4.2)
ax.text(5, 4.0, 'Training (SGD, 50/300 epochs)', ha='center', fontsize=11, weight='bold')

y_train = 3.2
x_positions = [0.8, 2.5, 4.2, 5.9]
methods = ['Baseline', 'MC Dropout', 'Ensemble', 'SWAG']
for i, (x, method) in enumerate(zip(x_positions, methods)):
    add_box(ax, x, y_train, 1.2, 0.5, method, color_train, fontsize=9)

# Convergence to evaluation
for x in x_positions:
    add_arrow(ax, x + 0.6, y_train, 3.3, 2.5)

# Row 3: Evaluation
add_box(ax, 2.3, 2.0, 2.0, 0.5, 'Test Evaluation', color_eval)

# Row 4: Conformal
add_arrow(ax, 3.3, 2.0, 3.3, 1.5)
add_box(ax, 2.3, 0.9, 2.0, 0.5, 'Conformal Prediction\n(α=0.1)', color_conformal, fontsize=9)

# Row 5: Results
add_arrow(ax, 3.3, 0.9, 3.3, 0.4)
add_box(ax, 2.3, 0.05, 2.0, 0.3, 'Coverage Analysis', '#F5F5F5', fontsize=9)

# Add minimal legend
legend_y = 0.1
legend_items = [
    ('Data', color_data),
    ('Training', color_train),
    ('Evaluation', color_eval),
    ('Conformal', color_conformal)
]

legend_x = 7.0
for i, (label, color) in enumerate(legend_items):
    box = Rectangle((legend_x, legend_y + i*0.25), 0.2, 0.15,
                    facecolor=color, edgecolor='black', linewidth=0.8)
    ax.add_patch(box)
    ax.text(legend_x + 0.25, legend_y + i*0.25 + 0.075, label, 
            fontsize=8, va='center')

plt.tight_layout()
plt.savefig('results/visualizations/workflow_flowchart.png', dpi=300, bbox_inches='tight')
print("✅ Flowchart saved")
plt.close()
