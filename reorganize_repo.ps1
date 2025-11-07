# Reorganize Repository Structure
# This script organizes the repo into a cleaner structure

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Reorganizing Repository" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Create new organized directory structure
Write-Host "`nCreating organized directory structure..." -ForegroundColor Yellow

# Core directories
New-Item -ItemType Directory -Force -Path "docs\archive" | Out-Null
New-Item -ItemType Directory -Force -Path "docs\guides" | Out-Null
New-Item -ItemType Directory -Force -Path "docs\status" | Out-Null
New-Item -ItemType Directory -Force -Path "results\final" | Out-Null
New-Item -ItemType Directory -Force -Path "results\figures" | Out-Null
New-Item -ItemType Directory -Force -Path "results\metrics" | Out-Null

Write-Host "[1/5] Organizing documentation..." -ForegroundColor Green

# Move status/tracking docs to docs/status/
$statusDocs = @(
    "RETRAINING_STATUS.md",
    "TRAINING_STATUS.md",
    "FINAL_STATUS.md",
    "DATASET_READY.md",
    "IMPLEMENTATION_COMPLETE.md",
    "AMAREL_READY_50EPOCHS.md"
)

foreach ($doc in $statusDocs) {
    if (Test-Path $doc) {
        Move-Item $doc "docs\status\" -Force
        Write-Host "  Moved $doc -> docs\status\" -ForegroundColor Gray
    }
}

# Move guide docs to docs/guides/
$guideDocs = @(
    "QUICK_START_RETRAIN.md",
    "README_RETRAIN.md",
    "RETRAINING_COMMANDS.md",
    "TROUBLESHOOTING_RETRAINING.md",
    "EXECUTION_CHECKLIST.md",
    "POST_TRAINING_WORKFLOW.md",
    "IMPORT_FIX_GUIDE.md",
    "NEXT_STEPS.md",
    "UPLOAD_COMMANDS.md"
)

foreach ($doc in $guideDocs) {
    if (Test-Path $doc) {
        Move-Item $doc "docs\guides\" -Force
        Write-Host "  Moved $doc -> docs\guides\" -ForegroundColor Gray
    }
}

Write-Host "`n[2/5] Organizing scripts..." -ForegroundColor Green

# Keep scripts organized (already in scripts/)
Write-Host "  Scripts already organized in scripts/" -ForegroundColor Gray

Write-Host "`n[3/5] Organizing results and metrics..." -ForegroundColor Green

# Copy metrics to results/ for easy access
if (Test-Path "runs\classification\metrics\comprehensive_metrics.json") {
    Copy-Item "runs\classification\metrics\comprehensive_metrics.json" "results\metrics\" -Force
    Write-Host "  Copied comprehensive_metrics.json -> results\metrics\" -ForegroundColor Gray
}

if (Test-Path "runs\classification\metrics\metrics_summary.csv") {
    Copy-Item "runs\classification\metrics\metrics_summary.csv" "results\metrics\" -Force
    Write-Host "  Copied metrics_summary.csv -> results\metrics\" -ForegroundColor Gray
}

# Copy visualizations to results/figures
if (Test-Path "runs\classification\metrics\*.png") {
    Copy-Item "runs\classification\metrics\*.png" "results\figures\" -Force
    Write-Host "  Copied visualization PNGs -> results\figures\" -ForegroundColor Gray
}

Write-Host "`n[4/5] Creating final results summary..." -ForegroundColor Green

# Create a final results document if metrics exist
if (Test-Path "runs\classification\metrics\metrics_summary.csv") {
    $summary = @"
# Final UQ Evaluation Results

**Date:** $(Get-Date -Format "MMMM dd, yyyy")

## Quick Summary

This directory contains the final evaluation results from our Uncertainty Quantification study on Chest X-Ray classification.

### Files:
- ``metrics/comprehensive_metrics.json`` - Complete metrics for all methods
- ``metrics/metrics_summary.csv`` - Summary table (accuracy, ECE, Brier, FNR, etc.)
- ``figures/comprehensive_metrics_visualization.png`` - Main visualization dashboard
- ``figures/method_comparisons.png`` - Method comparison scatter plots

### Methods Evaluated:
1. **Baseline** - Standard ResNet-18
2. **MC Dropout** - Monte Carlo Dropout (T=15 samples)
3. **Deep Ensemble** - 5 independent models
4. **SWAG** - Stochastic Weight Averaging Gaussian (T=30 samples)
5. **Conformal Risk Control** - Post-hoc calibration methods

### Key Findings:
- All methods trained for 50 epochs
- MC Dropout and SWAG retrained from baseline checkpoint
- Comprehensive metrics include: accuracy, calibration (ECE), Brier score, FNR/FPR, ROC-AUC
- Uncertainty quantification metrics: mean uncertainty, uncertainty separation

See ``../runs/classification/`` for full model checkpoints and training logs.
See ``../docs/`` for detailed documentation and guides.
"@
    
    Set-Content -Path "results\final\README.md" -Value $summary
    Write-Host "  Created results\final\README.md" -ForegroundColor Gray
}

Write-Host "`n[5/5] Creating updated main README..." -ForegroundColor Green

$mainReadme = @"
# Uncertainty Quantification for Medical Image Classification

**Capstone Project - Rutgers University**  
**Date:** November 2025

## 🎯 Project Overview

This project implements and compares multiple uncertainty quantification (UQ) methods for chest X-ray classification using deep learning.

## 📊 Results

Final evaluation results are available in ``results/``:
- **Metrics:** See ``results/metrics/`` for JSON and CSV summaries
- **Visualizations:** See ``results/figures/`` for all plots
- **Analysis:** See ``results/final/`` for complete analysis

### Quick Results Summary

| Method | Accuracy | ECE | Notes |
|--------|----------|-----|-------|
| Baseline | 91.67% | 0.0498 | Standard ResNet-18 |
| MC Dropout | 85.26% | 0.1171 | T=15 samples, dropout=0.2 |
| Deep Ensemble | 91.67% | 0.0285 | M=5 models |
| SWAG | 83.33% | 0.1518 | T=30 samples, scale=0.5 |
| CRC Methods | - | - | Post-hoc calibration |

## 🏗️ Repository Structure

\`\`\`
uq_capstone/
├── src/                    # Source code (training, evaluation, UQ methods)
├── scripts/                # SLURM scripts for HPC (Amarel cluster)
├── analysis/               # Analysis and visualization scripts
├── results/                # Final results, metrics, and figures
│   ├── metrics/           # JSON and CSV metric summaries
│   ├── figures/           # Visualization plots
│   └── final/             # Final analysis documents
├── runs/                   # Training runs and checkpoints
│   └── classification/    # Classification model runs
│       ├── baseline/
│       ├── mc_dropout/
│       ├── ensemble/
│       └── swag_classification/
├── docs/                   # Documentation
│   ├── guides/            # Setup and workflow guides
│   ├── status/            # Status tracking documents
│   └── *.md               # Various documentation files
├── data/                   # Dataset (not tracked in git)
├── papers/                 # Reference papers
└── presentation/           # Final presentation materials

\`\`\`

## 🚀 Quick Start

### View Results
1. Check ``results/figures/`` for visualizations
2. Review ``results/metrics/metrics_summary.csv`` for quick summary
3. Read ``results/final/README.md`` for detailed findings

### Run Locally (if you have the dataset)
\`\`\`powershell
# Install dependencies
pip install -r requirements.txt

# Run comprehensive evaluation
python src/comprehensive_metrics.py

# Generate visualizations
python analysis/visualize_metrics.py
\`\`\`

### Train on HPC (Amarel)
See ``docs/guides/`` for complete guides:
- ``QUICK_START_RETRAIN.md`` - Quick retraining guide
- ``RETRAINING_COMMANDS.md`` - Copy-paste commands
- ``AMAREL_SETUP_GUIDE.md`` - Full HPC setup

## 📚 Documentation

- **Setup Guides:** ``docs/guides/QUICK_START_*.md``
- **Status Reports:** ``docs/status/``
- **Implementation Details:** ``docs/IMPLEMENTATION_SUMMARY.md``
- **Troubleshooting:** ``docs/guides/TROUBLESHOOTING_RETRAINING.md``

## 🔬 Methods Implemented

1. **Baseline Classifier**
   - Standard ResNet-18 with cross-entropy loss
   - No uncertainty quantification

2. **MC Dropout**
   - Dropout-based Bayesian approximation
   - 15 forward passes with dropout enabled
   - dropout_rate = 0.2

3. **Deep Ensemble**
   - 5 independently trained models
   - Variance across predictions for uncertainty

4. **SWAG (Stochastic Weight Averaging Gaussian)**
   - Bayesian posterior approximation
   - 30 samples from weight distribution
   - Snapshots collected from epoch 30-50

5. **Conformal Risk Control**
   - Post-hoc calibration methods
   - Guarantees coverage under distribution shift

## 📈 Key Findings

1. **Accuracy:** Baseline and Ensemble achieve ~92%, MC Dropout ~85%, SWAG ~83%
2. **Calibration:** Ensemble best calibrated (ECE=0.0285), SWAG least (ECE=0.1518)
3. **Uncertainty:** MC Dropout and SWAG provide meaningful uncertainty estimates
4. **Overfitting:** MC Dropout and SWAG show signs of validation overfitting (99.62% val vs 85% test)

## 🛠️ Technologies

- **Framework:** PyTorch 2.0+
- **Model:** ResNet-18 (torchvision)
- **Dataset:** Chest X-Ray Images (Kaggle)
- **HPC:** Rutgers Amarel cluster (SLURM)
- **Analysis:** NumPy, Pandas, Matplotlib, Seaborn

## 👥 Contributors

- **Student:** [Your Name]
- **Advisor:** [Advisor Name]
- **Institution:** Rutgers University

## 📝 License

This project is for academic purposes only.

---

**Last Updated:** November 6, 2025
"@

Set-Content -Path "README.md" -Value $mainReadme
Write-Host "  Updated README.md" -ForegroundColor Gray

Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "Reorganization Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Cyan

Write-Host "`nNew structure:" -ForegroundColor Yellow
Write-Host "  docs/" -ForegroundColor Cyan
Write-Host "    ├── guides/          (setup and workflow guides)" -ForegroundColor Gray
Write-Host "    └── status/          (status tracking documents)" -ForegroundColor Gray
Write-Host "  results/" -ForegroundColor Cyan
Write-Host "    ├── metrics/         (JSON and CSV summaries)" -ForegroundColor Gray
Write-Host "    ├── figures/         (visualization plots)" -ForegroundColor Gray
Write-Host "    └── final/           (final analysis)" -ForegroundColor Gray

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Review the changes: git status" -ForegroundColor White
Write-Host "  2. Stage changes: git add ." -ForegroundColor White
Write-Host "  3. Commit: git commit -m Reorganize repository structure" -ForegroundColor White
Write-Host "  4. Push: git push origin main" -ForegroundColor White
