# Classification Experiments - Deployment Checklist

## Pre-Deployment ✅

- [x] Create dataset loaders for medical classification
- [x] Implement Conformal Risk Control
- [x] Create training scripts (Baseline, MC Dropout, Ensemble)
- [x] Create evaluation script with CRC
- [x] Create SLURM batch scripts
- [x] Write comprehensive documentation
- [x] Update main README
- [ ] Upload code to Amarel
- [ ] Download dataset

## Deployment Steps

### Step 1: Upload Code to Amarel

```bash
# From local machine (Windows PowerShell)
cd C:\Users\lpnhu\Downloads\uq_capstone

# Upload source files
scp src/data_utils_classification.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/conformal_risk_control.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/train_classifier_baseline.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/train_classifier_mc_dropout.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/train_classifier_ensemble_member.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/
scp src/evaluate_uq_classification.py YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/src/

# Upload batch scripts
scp scripts/train_classifier_baseline.sbatch YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/scripts/
scp scripts/train_classifier_mc_dropout.sbatch YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/scripts/
scp scripts/train_classifier_ensemble.sbatch YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/scripts/
scp scripts/evaluate_classification.sbatch YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/scripts/
scp scripts/run_all_classification_experiments.sh YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/scripts/

# Upload documentation
scp docs/CLASSIFICATION_QUICK_START.md YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/docs/
scp docs/CLASSIFICATION_SETUP_GUIDE.md YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/docs/
scp docs/CLASSIFICATION_IMPLEMENTATION_SUMMARY.md YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/docs/
```

- [ ] Source files uploaded
- [ ] Batch scripts uploaded
- [ ] Documentation uploaded

### Step 2: Download Dataset

#### Option A: Direct Download on Amarel (Recommended)
```bash
ssh YOUR_USERNAME@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone/data

# Install Kaggle CLI if needed
pip install kaggle

# Configure Kaggle API (need API key from kaggle.com)
mkdir -p ~/.kaggle
# Upload your kaggle.json to ~/.kaggle/

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d chest_xray/
rm chest-xray-pneumonia.zip
```

#### Option B: Download Locally Then Upload
1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download" (creates chest-xray-pneumonia.zip)
3. Extract locally
4. Upload:
   ```powershell
   scp -r chest_xray YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/data/
   ```

- [ ] Dataset downloaded
- [ ] Dataset verified (check directory structure)

### Step 3: Verify Setup

```bash
ssh YOUR_USERNAME@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone

# Activate environment
conda activate uq_capstone

# Test imports
python -c "
from src.data_utils_classification import get_classification_loaders
from src.conformal_risk_control import ConformalRiskControl
print('✓ All imports successful')
"

# Test data loading
python -c "
from src.data_utils_classification import get_classification_loaders
train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
    dataset_name='chest_xray',
    data_dir='data/chest_xray',
    batch_size=16
)
print(f'✓ Dataset loaded: {num_classes} classes')
print(f'  Train batches: {len(train_loader)}')
print(f'  Cal batches: {len(cal_loader)}')
print(f'  Test batches: {len(test_loader)}')
"
```

- [ ] Imports working
- [ ] Data loading working
- [ ] No errors

### Step 4: Launch Experiments

```bash
cd /scratch/$USER/uq_capstone

# Make script executable
chmod +x scripts/run_all_classification_experiments.sh

# Launch all jobs
bash scripts/run_all_classification_experiments.sh
```

This will submit 4 jobs:
- [ ] Baseline training submitted
- [ ] MC Dropout training submitted
- [ ] Ensemble training submitted
- [ ] Evaluation submitted (waits for others)

### Step 5: Monitor Progress

```bash
# Check job queue
squeue -u $USER

# Expected output:
# JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
# 12345 gpu       clf_baseline YOUR_USER  R       0:30      1 node123
# 12346 gpu       clf_mc_dropout YOUR_USER  R       0:28      1 node124
# 12347 gpu       clf_ensemble YOUR_USER  R       0:25      1 node125
# 12348 gpu       clf_eval YOUR_USER PD       0:00      1 (Dependency)
```

- [ ] Jobs running
- [ ] No immediate errors

### Step 6: Check Logs

```bash
# Training logs
tail -f runs/classification/baseline/train_*.out
tail -f runs/classification/mc_dropout/train_*.out
tail -f runs/classification/ensemble/train_all_*.out

# Look for:
# - "Epoch 1/50"
# - "Train Loss: ... | Train Acc: ..."
# - "✓ Model created"
# - GPU being used
```

- [ ] Training started
- [ ] Loss decreasing
- [ ] Accuracy increasing

## During Training

### Daily Monitoring

```bash
# Check status
squeue -u $USER

# Check progress (baseline)
grep "Epoch" runs/classification/baseline/train_*.out | tail -5

# Check if best model saved
ls -lh runs/classification/baseline/best_model.pth

# Check GPU usage (if job running)
squeue -u $USER --format="%.18i %.9P %.8T %.10M %.6D %.20S %.10r"
```

### Expected Timeline

| Job | Duration | When Complete |
|-----|----------|---------------|
| Baseline | ~12 hours | Day 1 |
| MC Dropout | ~12 hours | Day 1 |
| Ensemble | ~24 hours | Day 2 |
| Evaluation | ~8 hours | Day 2-3 |

**Total: 2-3 days** (jobs run in parallel)

- [ ] Day 1: Training jobs submitted
- [ ] Day 2: Ensemble complete
- [ ] Day 3: Evaluation complete

## Post-Training

### Step 7: Check Results

```bash
cd /scratch/$USER/uq_capstone

# View evaluation results
cat runs/classification/evaluation/all_results.json

# Check model sizes
du -h runs/classification/*/best_model.pth

# Check training history
cat runs/classification/baseline/history.json
```

- [ ] Evaluation completed
- [ ] Results look reasonable
- [ ] All models saved

### Step 8: Download Results

```powershell
# On local machine (Windows PowerShell)
# Download everything
scp -r YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/runs/classification/ ./results_classification/

# Or just results
scp YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/runs/classification/evaluation/all_results.json ./
```

- [ ] Results downloaded
- [ ] Logs downloaded
- [ ] Models downloaded (optional, large files)

## Analysis

### Step 9: Analyze Results

- [ ] Compare accuracy across methods
- [ ] Examine ECE (calibration)
- [ ] Review CRC guarantees (empirical risk ≤ target)
- [ ] Compare coverage vs set size trade-offs
- [ ] Identify best method for each metric

### Questions to Answer

1. **Which method has highest accuracy?**
   - Expected: Ensemble
   - Actual: _____

2. **Which method is best calibrated?**
   - Metric: ECE
   - Best: _____

3. **Does CRC satisfy guarantees?**
   - FNR (α=0.05): Empirical ≤ 0.05? _____
   - Set Size (α=2.0): Avg size ≤ 2.0? _____

4. **What's the trade-off?**
   - Coverage vs Set Size
   - Observation: _____

## Troubleshooting

### Common Issues

**Job Pending Forever**
```bash
# Check reason
squeue -u $USER --format="%.18i %.9P %.8T %.10M %.6D %.20S %.10r"

# If partition busy, try different partition
sbatch --partition=main scripts/train_classifier_baseline.sbatch
```

**Out of Memory**
```bash
# Edit sbatch file, reduce batch size
nano scripts/train_classifier_baseline.sbatch
# Change: --batch_size 32 to --batch_size 16
```

**Import Errors**
```bash
# Reinstall packages
conda activate uq_capstone
pip install torch torchvision scikit-learn tqdm
```

**Dataset Not Found**
```bash
# Verify structure
ls -R data/chest_xray/
# Should see: train/, test/, val/
# Each with: NORMAL/, PNEUMONIA/
```

## Success Checklist

- [ ] All jobs completed without errors
- [ ] Baseline accuracy > 90%
- [ ] MC Dropout accuracy > 90%
- [ ] Ensemble accuracy > Baseline
- [ ] CRC empirical risk ≤ target risk (for all configurations)
- [ ] Results saved in JSON
- [ ] Results downloaded locally
- [ ] Ready to discuss with professor

## Optional: Additional Experiments

### Try Other Datasets

```bash
# OCT Retinal (4 classes, larger)
# 1. Download dataset
# 2. Update sbatch scripts: --dataset oct_retinal
# 3. Rerun experiments

# Brain Tumor (4 classes, medium)
# 1. Download dataset
# 2. Update sbatch scripts: --dataset brain_tumor
# 3. Rerun experiments
```

- [ ] OCT Retinal experiments
- [ ] Brain Tumor experiments

### Hyperparameter Tuning

```bash
# Try different learning rates
# Edit sbatch: --lr 0.0001 or --lr 0.01

# Try different architectures
# Edit sbatch: --arch resnet34 or --arch resnet50

# Try different dropout rates
# Edit sbatch: --dropout_rate 0.5
```

- [ ] Learning rate experiments
- [ ] Architecture experiments
- [ ] Dropout rate experiments

## Reporting

### For Your Professor

Prepare summary:
1. **Methods compared**: 4 (Baseline, MC Dropout, Ensemble, CRC)
2. **Best accuracy**: ____ (method: ____)
3. **Best calibration**: ____ (method: ____)
4. **CRC guarantees**: Verified ✓ / Not verified ✗
5. **Key finding**: _____

### For Your Thesis

Sections to include:
- [ ] Introduction (medical imaging + UQ importance)
- [ ] Background (conformal prediction, UQ methods)
- [ ] Methods (detailed CRC implementation)
- [ ] Experiments (3 datasets, 4 methods)
- [ ] Results (tables, figures)
- [ ] Discussion (clinical implications)
- [ ] Conclusion (contributions, future work)

## Timeline Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Implementation | 4 hours | ✅ Complete |
| Code upload | 30 min | ⏳ Pending |
| Dataset download | 1 hour | ⏳ Pending |
| Training | 2-3 days | ⏳ Pending |
| Evaluation | 8 hours | ⏳ Pending |
| Analysis | 4 hours | ⏳ Pending |
| **Total** | **~4 days** | ⏳ |

## Final Notes

- **Save everything**: Models, logs, configs
- **Document changes**: If you modify hyperparameters
- **Version control**: Commit to git regularly
- **Backup**: Download results from Amarel
- **Share with professor**: Results + code

---

**Ready to deploy! Follow this checklist step by step.** ✨

Last updated: October 20, 2025
