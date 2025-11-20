# UQ Analysis - Quick Start Guide

## 🎯 Overview

All UQ analysis scripts are now ready! This guide shows you how to run them on Amarel HPC to generate comprehensive analysis and visualizations.

## 📊 What Gets Generated

### 1. Metrics Analysis (`analyze_uq_metrics.py`)
- **Calibration Metrics**: ECE, MCE, Brier Score
- **Reliability Diagrams**: Figure 3 style from SWAG paper (confidence vs accuracy)
- **Uncertainty Quality**: Correlation with errors, AUROC for error detection
- **Method Comparison**: Radar charts, metrics tables
- **Outputs**:
  - `metrics_summary.csv` - All metrics in table format
  - `metrics_summary.md` - Markdown version
  - `reliability_diagrams.png` - 4-panel calibration plots
  - `uncertainty_error_correlation.png` - Scatter plots
  - `roc_curves_error_detection.png` - ROC curves
  - `method_comparison_radar.png` - Radar chart

### 2. Visualizations (`visualize_uq.py`)
- **4-Panel Maps**: Image | Ground Truth | Prediction | Uncertainty
- **Method Comparisons**: All 4 methods on same samples
- **Uncertainty Distributions**: Histograms for each method
- **Performance Heatmap**: All metrics across methods
- **Outputs**:
  - `figures/baseline/sample_*_4panel.png`
  - `figures/mc_dropout/sample_*_4panel.png`
  - `figures/ensemble/sample_*_4panel.png`
  - `figures/swag/sample_*_4panel.png`
  - `method_comparison_sample_*.png`
  - `uncertainty_distributions.png`
  - `performance_heatmap.png`

### 3. Comprehensive Report (`generate_uq_report.py`)
- **UQ_ANALYSIS_REPORT.md** with:
  - Executive summary with key results table
  - Method overviews (Baseline, MC Dropout, Ensemble, SWAG)
  - Performance analysis (Dice scores, rankings)
  - Calibration analysis (ECE, reliability)
  - Uncertainty quality (correlation, AUROC)
  - Key findings and highlights
  - Recommendations for each use case
  - Conclusions and future directions
  - Complete references and appendices

## 🚀 Running on Amarel

### Step 1: Upload Analysis Scripts

```bash
# From local machine (PowerShell)
cd C:\Users\lpnhu\Downloads\uq_capstone

# Upload the analysis folder
scp -r analysis hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/
```

### Step 2: SSH to Amarel

```bash
ssh hpl14@amarel.rutgers.edu
cd /scratch/hpl14/uq_capstone
```

### Step 3: Activate Environment

```bash
conda activate uq_capstone
```

### Step 4: Run Analysis Scripts

```bash
# Create output directory
mkdir -p runs/uq_analysis/figures

# Run metrics analysis (generates calibration plots, correlation analysis)
python analysis/analyze_uq_metrics.py

# Run visualizations (generates uncertainty maps, method comparisons)
python analysis/visualize_uq.py

# Generate comprehensive report
python analysis/generate_uq_report.py
```

**Expected Runtime**:
- `analyze_uq_metrics.py`: ~2-3 minutes
- `visualize_uq.py`: ~5-10 minutes (depends on model loading)
- `generate_uq_report.py`: <1 minute

### Step 5: Download Results

```bash
# From local machine (PowerShell)
cd C:\Users\lpnhu\Downloads\uq_capstone

# Download all analysis results
scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/uq_analysis ./runs/
```

## 📁 Output Structure

After running all scripts, you'll have:

```
runs/uq_analysis/
├── metrics_summary.csv              # Comprehensive metrics table
├── metrics_summary.md               # Markdown version
├── reliability_diagrams.png         # Calibration plots (4 methods)
├── uncertainty_error_correlation.png # Uncertainty vs error
├── roc_curves_error_detection.png   # ROC curves
├── method_comparison_radar.png      # Radar chart
├── uncertainty_distributions.png    # Histograms
├── performance_heatmap.png          # Metrics heatmap
├── UQ_ANALYSIS_REPORT.md           # Comprehensive report
└── figures/                         # Sample visualizations
    ├── baseline/
    │   ├── sample_0_4panel.png
    │   ├── sample_20_4panel.png
    │   └── sample_40_4panel.png
    ├── mc_dropout/
    │   └── ... (same structure)
    ├── ensemble/
    │   └── ... (same structure)
    ├── swag/
    │   └── ... (same structure)
    ├── method_comparison_sample_10.png
    ├── method_comparison_sample_30.png
    └── method_comparison_sample_50.png
```

## 🎨 Key Visualizations Explained

### Reliability Diagrams (Figure 3 from SWAG Paper)
- **X-axis**: Confidence (predicted probability)
- **Y-axis**: Accuracy (actual correctness)
- **Diagonal line**: Perfect calibration
- **Points below diagonal**: Overconfident
- **Points above diagonal**: Underconfident

### Uncertainty Maps (4-Panel)
1. **Input Image**: Original MRI slice (FLAIR)
2. **Ground Truth**: Tumor mask overlay
3. **Prediction**: Model's segmentation
4. **Uncertainty**: Heatmap (red = high uncertainty)

### Method Comparison
- Shows all 4 methods on same sample
- Easy to see which method captures uncertainty better
- Identifies where methods disagree

## 🔍 What to Look For

### Good Uncertainty Estimates Should:

1. **Correlate with errors**
   - High uncertainty where predictions are wrong
   - Low uncertainty where predictions are correct
   - Check `uncertainty_error_correlation.png`

2. **Detect mistakes**
   - AUROC > 0.7 means uncertainty helps identify errors
   - Check `roc_curves_error_detection.png`

3. **Show calibration**
   - Points follow diagonal in reliability diagrams
   - Low ECE (<0.05 is excellent)
   - Check `reliability_diagrams.png`

4. **Be interpretable**
   - Higher values in ambiguous regions (tumor boundaries)
   - Lower values in clear regions (background, tumor core)
   - Check sample 4-panel visualizations

## 📊 Expected Results

Based on our evaluation (Job 47441209):

| Method | Dice | ECE | Uncertainty | Correlation Expected |
|--------|------|-----|-------------|---------------------|
| Deep Ensemble | 0.7550 | 0.9589 | 0.0158 | Highest |
| SWAG | 0.7419 | 0.9656 | 0.0026 | Good |
| MC Dropout | 0.7403 | 0.9663 | 0.0011 | Moderate |
| Baseline | 0.7401 | 0.9673 | N/A | N/A |

### Key Insights You'll Discover:

1. **Deep Ensemble** has best overall performance
2. **SWAG** is best efficiency-performance tradeoff
3. **MC Dropout** is simplest to implement
4. **Baseline** performs surprisingly well without uncertainty

## 🐛 Troubleshooting

### If Model Checkpoints Are Missing:

The scripts will use **synthetic predictions** for visualization. This is fine for:
- Understanding the output structure
- Testing the pipeline
- Seeing example visualizations

To use **real predictions**, ensure model checkpoints exist:
```bash
ls runs/baseline/best_model.pth
ls runs/mc_dropout/best_model.pth
ls runs/ensemble/model_0_best.pth
ls runs/swag/swag_model.pth
```

### If `results.json` Is Missing:

Re-run evaluation:
```bash
sbatch scripts/evaluate_uq.sbatch
```

### If Imports Fail:

Ensure environment is activated:
```bash
conda activate uq_capstone
```

### If Out of Memory:

Reduce batch size in visualization script:
```python
# In visualize_uq.py, line ~30
n_samples_per_method = 3  # Reduce from 5 to 3
```

## 🎯 Next Steps After Analysis

1. **Review the Report**
   - Read `UQ_ANALYSIS_REPORT.md` for comprehensive insights
   - Check executive summary for key findings

2. **Examine Visualizations**
   - Look at reliability diagrams for calibration quality
   - Check 4-panel maps to see uncertainty in action
   - Review method comparisons to understand differences

3. **Iterate Based on Findings**
   - If calibration is poor: Apply temperature scaling
   - If uncertainty is low: Tune hyperparameters (dropout rate, max_var)
   - If correlation is weak: Try ensemble methods

4. **Prepare for Presentation**
   - All figures are publication-ready (300 DPI)
   - Report includes complete methodology
   - Metrics tables ready for papers/slides

## 📚 Understanding the Paper's Figure 3

The reliability diagrams we generate are modeled after **Figure 3** from the SWAG paper:

- **Multiple datasets**: WideResNet28x10, ResNet-152, DenseNet-161
- **Confidence vs Accuracy**: Shows calibration quality
- **Diagonal = Perfect**: Points should follow this line
- **Our version**: Same concept, applied to medical imaging

**Key difference**: Paper uses classification (ImageNet), we use segmentation (BraTS). But the calibration principle is the same!

## 🏆 Final Checklist

Before running analysis:
- ✅ Evaluation completed (`results.json` exists)
- ✅ Model checkpoints saved (optional but recommended)
- ✅ Environment activated (`conda activate uq_capstone`)
- ✅ Output directory created (`runs/uq_analysis/`)

After running analysis:
- ✅ All visualizations generated
- ✅ Metrics tables created
- ✅ Report generated
- ✅ Results downloaded to local machine

**You're ready to run the complete UQ analysis pipeline!** 🚀

---

## 📞 Support

If you encounter issues:
1. Check the terminal output for error messages
2. Verify file paths in scripts match your setup
3. Ensure all dependencies are installed
4. Check that evaluation results (`results.json`) exist

For questions, open a GitHub issue at: https://github.com/valle1306/uq_capstone

---

**Last Updated**: 2025-10-10  
**Scripts Version**: 1.0  
**Status**: ✅ Ready to run
