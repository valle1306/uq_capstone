# Quick Start: Medical Image Classification + Conformal Risk Control

**Fast track to running classification experiments on Amarel**

## Prerequisites ✅

- [x] Access to Amarel HPC
- [x] Conda environment `uq_capstone` activated
- [x] Dataset downloaded (Chest X-Ray recommended)

## 3-Step Quick Start

### Step 1: Upload Code to Amarel

```powershell
# On your local Windows machine
cd C:\Users\lpnhu\Downloads\uq_capstone

# Upload new classification files
scp src/data_utils_classification.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/conformal_risk_control.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/train_classifier_baseline.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/train_classifier_mc_dropout.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/train_classifier_ensemble_member.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/evaluate_uq_classification.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/

# Upload batch scripts
scp scripts/train_classifier_*.sbatch YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/scripts/
scp scripts/evaluate_classification.sbatch YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/scripts/
scp scripts/run_all_classification_experiments.sh YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/scripts/
```

### Step 2: Download Dataset

**Option A: Using Kaggle API (on Amarel)**
```bash
ssh YOUR_USERNAME@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone/data

# Download Chest X-Ray dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d chest_xray/
rm chest-xray-pneumonia.zip
```

**Option B: Manual Download (recommended if Kaggle API not setup)**
1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download" (requires Kaggle account)
3. Extract on your local machine
4. Upload to Amarel:
   ```powershell
   scp -r chest_xray YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/data/
   ```

### Step 3: Run All Experiments

```bash
# SSH to Amarel
ssh YOUR_USERNAME@amarel.rutgers.edu

# Navigate to project
cd /scratch/$USER/uq_capstone

# Make script executable
chmod +x scripts/run_all_classification_experiments.sh

# Run everything!
bash scripts/run_all_classification_experiments.sh
```

**This will:**
- Train baseline classifier (~12 hours)
- Train MC Dropout (~12 hours)
- Train 5 ensemble members (~24 hours)
- Run comprehensive evaluation with CRC (~8 hours)

**Total:** 30-40 hours (jobs run in parallel)

## Monitor Progress

```bash
# Check job status
squeue -u $USER

# View training logs
tail -f runs/classification/baseline/train_*.out
tail -f runs/classification/mc_dropout/train_*.out
tail -f runs/classification/ensemble/train_all_*.out

# View evaluation logs
tail -f runs/classification/evaluation/eval_*.out
```

## Get Results

```bash
# On Amarel, view results
cat runs/classification/evaluation/all_results.json

# Download to local machine (Windows PowerShell)
scp -r YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/runs/classification/ ./results_classification/
```

## What to Expect

### Training Output

Each training script will show:
```
Epoch 1/50
  Train Loss: 0.4521 | Train Acc: 82.34%
  Val Loss:   0.3821 | Val Acc:   85.67%
  🎉 New best accuracy: 85.67%
```

### Evaluation Output

```
========================================
Evaluating Baseline
========================================
✓ Baseline Results:
  Accuracy: 94.23%
  ECE: 0.0342
  Brier Score: 0.1245

========================================
Evaluating Conformal Risk Control
========================================
Testing: FNR Control (α=0.05)
✓ FNR Control (α=0.05) Results:
  Target Risk: 0.0500
  Empirical Risk: 0.0421
  Coverage: 0.9579
  Avg Set Size: 1.23
```

## File Structure After Experiments

```
uq_capstone/
├── data/
│   └── chest_xray/          # Dataset
│       ├── train/
│       ├── test/
│       └── val/
│
├── runs/
│   └── classification/
│       ├── baseline/
│       │   ├── best_model.pth
│       │   ├── history.json
│       │   └── train_*.out
│       ├── mc_dropout/
│       │   ├── best_model.pth
│       │   └── ...
│       ├── ensemble/
│       │   ├── member_0/
│       │   ├── member_1/
│       │   ├── member_2/
│       │   ├── member_3/
│       │   └── member_4/
│       └── evaluation/
│           ├── all_results.json  ← MAIN RESULTS
│           └── eval_*.out
│
├── src/
│   ├── data_utils_classification.py     # Dataset loaders
│   ├── conformal_risk_control.py        # CRC implementation
│   ├── train_classifier_baseline.py     # Training scripts
│   ├── train_classifier_mc_dropout.py
│   ├── train_classifier_ensemble_member.py
│   └── evaluate_uq_classification.py    # Evaluation
│
└── scripts/
    ├── train_classifier_baseline.sbatch
    ├── train_classifier_mc_dropout.sbatch
    ├── train_classifier_ensemble.sbatch
    ├── evaluate_classification.sbatch
    └── run_all_classification_experiments.sh
```

## Troubleshooting

### Dataset Not Found
```bash
# Check if dataset exists
ls -R data/chest_xray/

# Should show train/, test/, val/ directories
# If not, re-download dataset
```

### Jobs Pending Too Long
```bash
# Check queue
squeue

# If too busy, specify different partition
sbatch --partition=main scripts/train_classifier_baseline.sbatch
```

### Out of Memory
```bash
# Edit sbatch script, reduce batch size
nano scripts/train_classifier_baseline.sbatch
# Change: --batch_size 32 to --batch_size 16
```

## Next Steps

1. **Review Results:**
   - Compare accuracy across methods
   - Check conformal risk control guarantees
   - Examine calibration metrics

2. **Try Other Datasets:**
   - OCT Retinal: `--dataset oct_retinal`
   - Brain Tumor: `--dataset brain_tumor`

3. **Tune Hyperparameters:**
   - Adjust learning rate, dropout rate
   - Try different architectures (resnet34, resnet50)

4. **Customize CRC:**
   - Modify loss functions in `conformal_risk_control.py`
   - Add domain-specific risk metrics

## Key Differences from Segmentation

| Aspect | Segmentation (Previous) | Classification (New) |
|--------|------------------------|---------------------|
| **Task** | Pixel-wise tumor masks | Image-level labels |
| **Output** | Dice score (~0.74) | Accuracy (~95%) |
| **Uncertainty** | Per-pixel variance | Per-image confidence |
| **Dataset** | BraTS (brain tumors) | Chest X-Ray (pneumonia) |
| **New Method** | - | ✨ Conformal Risk Control |

## Conformal Risk Control Explained

**Standard Conformal Prediction:**
- Goal: "True label in prediction set 90% of time"
- Metric: Coverage

**Conformal Risk Control (NEW):**
- Goal: "False negative rate ≤ 5%" or "Set size ≤ 2"
- Metric: Any risk functional
- More flexible for medical decisions!

### 5 Loss Functions Tested:

1. **FNR (α=0.05):** Miss disease ≤ 5% of time
2. **FNR (α=0.10):** Miss disease ≤ 10% of time
3. **Set Size (α=2.0):** Predict ≤ 2 labels on average
4. **Composite:** Balance FNR and set size
5. **F1 Score:** Optimize precision-recall trade-off

## Questions to Consider

After experiments, discuss with your professor:

1. **Why did ensemble perform well in segmentation?**
   - Is it still best for classification?

2. **Which risk objective is most clinically relevant?**
   - FNR control for critical diagnoses?
   - Set size control for practical deployment?

3. **How do results compare across datasets?**
   - Pneumonia vs Retinopathy vs Brain Tumor?

4. **What's the right balance?**
   - High coverage vs small prediction sets?

---

## Summary Checklist

- [ ] Code uploaded to Amarel
- [ ] Dataset downloaded and verified
- [ ] All experiments submitted
- [ ] Jobs running (check with `squeue -u $USER`)
- [ ] Logs being generated
- [ ] Waiting for results (~1-2 days)

**You're all set!** 🎉

---

*For detailed explanations, see: `docs/CLASSIFICATION_SETUP_GUIDE.md`*
