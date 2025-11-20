# Medical Image Classification with UQ - Setup Guide

This guide explains how to set up and run medical image classification experiments with uncertainty quantification methods, including the new **Conformal Risk Control**.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Dataset Options](#dataset-options)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Running Experiments](#running-experiments)
6. [Understanding Results](#understanding-results)

---

## Overview

### What Changed from Segmentation?

**Previous:** BraTS brain tumor **segmentation** (pixel-wise predictions)
**Now:** Medical image **classification** (image-level predictions)

### Why Classification?

1. **Simpler interpretation** of uncertainty
2. **Better suited** for conformal methods
3. **Easier validation** of UQ quality
4. **More datasets available**

### Methods Implemented

| Method | Type | Uncertainty Source |
|--------|------|-------------------|
| **Baseline** | Point estimate | None (confidence only) |
| **MC Dropout** | Bayesian approx. | Model parameter uncertainty |
| **Deep Ensemble** | Ensemble | Model disagreement |
| **Conformal Risk Control** | Distribution-free | Calibration-based sets |

---

## Dataset Options

### 1. Chest X-Ray Pneumonia ⭐ RECOMMENDED

**Best for initial experiments**

- **Source:** [Kaggle - Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes:** 2 (NORMAL, PNEUMONIA)
- **Size:** ~5,863 images
- **Medical relevance:** Pneumonia detection
- **Download:**
  ```bash
  # On Kaggle, download and extract to:
  data/chest_xray/
  ```

**Directory structure:**
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 2. OCT Retinal Images

**For multi-class experiments**

- **Source:** [Kaggle - OCT Dataset](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- **Classes:** 4 (CNV, DME, DRUSEN, NORMAL)
- **Size:** ~84,495 images
- **Medical relevance:** Diabetic retinopathy detection

### 3. Brain Tumor MRI

**For brain imaging continuity**

- **Source:** [Kaggle - Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** 4 (glioma, meningioma, notumor, pituitary)
- **Size:** ~7,023 images
- **Medical relevance:** Brain tumor classification

---

## Installation

### On Amarel HPC

```bash
# 1. SSH to Amarel
ssh YOUR_USERNAME@amarel.rutgers.edu

# 2. Navigate to project
cd /scratch/$USER/uq_capstone

# 3. Activate environment (should already exist from segmentation project)
conda activate uq_capstone

# 4. Verify packages
python -c "import torch, torchvision, sklearn; print('✓ All packages ready')"
```

### Local Testing (Optional)

```bash
# Create new environment
conda create -n uq_classification python=3.8
conda activate uq_classification

# Install packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn tqdm pillow
```

---

## Data Preparation

### Step 1: Download Dataset

**Option A: Direct download on Amarel**
```bash
# SSH to Amarel
cd /scratch/$USER/uq_capstone/data

# Download from Kaggle (requires Kaggle API)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d chest_xray/
```

**Option B: Download locally, then upload**
```powershell
# On your local machine (Windows PowerShell)
# 1. Download from Kaggle website manually
# 2. Upload to Amarel
scp -r chest_xray YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/data/
```

### Step 2: Verify Dataset

```bash
# On Amarel
cd /scratch/$USER/uq_capstone

# Test data loading
python -c "
from src.data_utils_classification import get_classification_loaders
try:
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name='chest_xray',
        batch_size=16
    )
    print(f'✓ Dataset loaded successfully!')
    print(f'  Classes: {num_classes}')
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Test batches: {len(test_loader)}')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

---

## Running Experiments

### Quick Start: Run All Experiments

```bash
# Submit all training jobs + evaluation
bash scripts/run_all_classification_experiments.sh
```

This will:
1. Train baseline model (~12 hours)
2. Train MC Dropout model (~12 hours)
3. Train 5 ensemble members (~24 hours)
4. Run comprehensive evaluation (~8 hours)

**Total time:** ~30-40 hours (jobs run in parallel)

### Individual Experiments

#### 1. Train Baseline

```bash
sbatch scripts/train_classifier_baseline.sbatch
```

**What it does:**
- Trains standard ResNet-18
- No uncertainty quantification
- Serves as baseline for comparison

**Output:** `runs/classification/baseline/best_model.pth`

#### 2. Train MC Dropout

```bash
sbatch scripts/train_classifier_mc_dropout.sbatch
```

**What it does:**
- Trains ResNet-18 with dropout layers
- Enables MC sampling for uncertainty
- Dropout rate: 0.3

**Output:** `runs/classification/mc_dropout/best_model.pth`

#### 3. Train Deep Ensemble

```bash
sbatch scripts/train_classifier_ensemble.sbatch
```

**What it does:**
- Trains 5 independent models
- Different random seeds
- Ensemble predictions for uncertainty

**Output:** `runs/classification/ensemble/member_{0-4}/best_model.pth`

#### 4. Run Evaluation

```bash
sbatch scripts/evaluate_classification.sbatch
```

**What it does:**
- Evaluates all trained models
- Tests Conformal Risk Control with 5 different loss functions
- Computes calibration metrics

**Output:** `runs/classification/evaluation/all_results.json`

---

## Understanding Results

### Metrics Explained

#### 1. Standard Classification Metrics

- **Accuracy:** Overall correctness (%)
- **ECE (Expected Calibration Error):** How well probabilities match frequencies (lower is better)
- **Brier Score:** Probabilistic prediction quality (lower is better)

#### 2. Uncertainty Quality Metrics

- **Mean Uncertainty:** Average uncertainty across test set
- **Uncertainty-Error Correlation:** Do uncertain predictions correlate with errors?

#### 3. Conformal Risk Control Metrics

- **Target Risk (α):** Desired risk level (user-specified)
- **Empirical Risk:** Actual risk on test set (should be ≤ α)
- **Coverage:** % of times true label in prediction set
- **Avg Set Size:** Average number of labels in prediction sets

### Expected Results

Based on CIFAR-10 benchmarks and medical imaging literature:

| Method | Expected Accuracy | Notes |
|--------|------------------|-------|
| **Baseline** | 90-95% | Standard ResNet-18 |
| **MC Dropout** | 90-95% | Similar to baseline, adds uncertainty |
| **Ensemble** | 92-96% | Typically 1-3% better |
| **CRC** | N/A | Focuses on risk control, not accuracy |

### Conformal Risk Control Variants

The evaluation tests 5 different risk control objectives:

1. **FNR Control (α=0.05):** Guarantee false negative rate ≤ 5%
   - *Use case:* Medical diagnosis where missing disease is critical
   
2. **FNR Control (α=0.10):** Guarantee false negative rate ≤ 10%
   - *Use case:* Less strict version, allows smaller prediction sets
   
3. **Set Size Control (α=2.0):** Keep prediction sets small (avg ≤ 2 labels)
   - *Use case:* Limit computational cost or user burden
   
4. **Composite (FNR+Size):** Balance coverage and efficiency
   - *Use case:* Trade-off between safety and practicality
   
5. **F1 Score Control:** Optimize precision-recall balance
   - *Use case:* When both false positives and negatives matter

---

## Monitoring Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Detailed info
scontrol show job JOB_ID
```

### View Real-Time Logs

```bash
# Training logs
tail -f runs/classification/baseline/train_*.out

# Evaluation logs
tail -f runs/classification/evaluation/eval_*.out
```

### Check GPU Usage

```bash
# On compute node (during job)
nvidia-smi
watch -n 1 nvidia-smi  # Update every second
```

---

## Troubleshooting

### Dataset Not Found

**Error:** `FileNotFoundError: Dataset not found at data/chest_xray`

**Solution:**
```bash
# Verify directory structure
ls -R data/chest_xray/

# Should see: train/, test/, val/ directories
# If not, re-download dataset
```

### Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Edit sbatch script, reduce batch size
--batch_size 16  # Instead of 32
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
# Reinstall environment
conda activate uq_capstone
pip install -r requirements.txt
```

---

## Next Steps

After experiments complete:

1. **Review results:**
   ```bash
   cat runs/classification/evaluation/all_results.json
   ```

2. **Download results locally:**
   ```powershell
   # On your Windows machine
   scp -r YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/runs/classification/ ./
   ```

3. **Analyze and visualize:**
   - Compare accuracy across methods
   - Examine calibration curves
   - Evaluate risk control guarantees

4. **Try other datasets:**
   - Modify sbatch scripts to use `oct_retinal` or `brain_tumor`
   - Rerun experiments

---

## Questions for Your Professor

After reviewing results, consider discussing:

1. **Which UQ method is best for medical diagnosis?**
   - Ensemble had best accuracy in segmentation
   - Does this hold for classification?
   
2. **How to choose conformal risk objectives?**
   - When to prioritize FNR vs set size?
   - Clinical decision-making implications

3. **Comparison with previous results:**
   - 74% Dice in segmentation vs ~95% accuracy in classification
   - Why such different numbers? (Different tasks!)

---

## Additional Resources

- **Conformal Risk Control Paper:** `/papers/Conformal Risk Control.pdf`
- **Original Segmentation Docs:** `/docs/START_HERE.md`
- **Code Documentation:** Comments in all Python files

---

## Summary

You've now set up a complete medical image classification pipeline with:

✅ Multiple medical datasets
✅ 3 UQ methods (Baseline, MC Dropout, Ensemble)
✅ **New:** Conformal Risk Control with 5 loss functions
✅ Automated training and evaluation on Amarel
✅ Comprehensive metrics and analysis

Good luck with your experiments! 🚀
