# 🎯 Retraining Execution Checklist

## Phase 1: Prepare for Submission (RIGHT NOW - You're Here)

- [x] Retrain scripts created and tested locally
  - `src/retrain_mc_dropout.py` (250+ lines)
  - `src/retrain_swag.py` (280+ lines)

- [x] SBATCH job scripts created
  - `scripts/retrain_mc_dropout.sbatch` (24h GPU)
  - `scripts/retrain_swag.sbatch` (24h GPU)

- [x] All code committed and pushed to GitHub
  - Commit cadef89 (latest)
  - Branch: main
  - Status: Ready for pull

- [x] Documentation created
  - QUICK_START_RETRAIN.md
  - RETRAINING_COMMANDS.md
  - RETRAINING_STATUS.md
  - RETRAINING_WORKFLOW.md
  - IMPLEMENTATION_COMPLETE.md

## Phase 2: Submit Jobs on Amarel (⏳ YOUR TURN - Run These Commands)

### Command Sequence (Copy & Paste into Amarel SSH Terminal)

```bash
# 1. SSH to Amarel
ssh hpl14@amarel.rutgers.edu

# 2. Navigate to project
cd /scratch/$USER/uq_capstone

# 3. Update code from GitHub
git fetch origin main
git reset --hard FETCH_HEAD

# 4. Backup old models
mv runs/classification/mc_dropout runs/classification/mc_dropout_old
mv runs/classification/swag_classification runs/classification/swag_classification_old

# 5. Create output directories
mkdir -p runs/classification/mc_dropout
mkdir -p runs/classification/swag_classification
mkdir -p logs

# 6. Verify baseline checkpoint exists
ls -lh runs/classification/baseline/best_model.pth

# 7. Submit MC Dropout retraining
sbatch scripts/retrain_mc_dropout.sbatch

# 8. Submit SWAG retraining
sbatch scripts/retrain_swag.sbatch

# 9. Check job status
squeue -u hpl14

# 10. Note the Job IDs from output (you'll need them)
```

### Expected Output After Submission
```
Submitted batch job 12345678  # MC Dropout Job ID
Submitted batch job 12345679  # SWAG Job ID
```

### Monitor Jobs While Running
```bash
# Run every 5 minutes to check progress
squeue -u hpl14

# View recent job output (replace ID with actual Job ID)
tail logs/retrain_mc_dropout_12345678.out

# Full output if job is still running
cat logs/retrain_mc_dropout_12345678.out
```

## Phase 3: Wait for Training (~24-48 hours)

**Estimated Timeline:**
- Job submission: Now
- MC Dropout training: ~24 hours (50 epochs × ~28 min/epoch)
- SWAG training: ~24 hours (50 epochs × ~28 min/epoch)
- Both run in parallel
- Estimated completion: ~24 hours from now

**What's happening:**
```
MC Dropout:
  Epoch 1-50: Fine-tuning from baseline weights
  Each epoch: ~28 minutes
  Total: ~24 hours

SWAG:
  Epoch 1-29: Fine-tuning from baseline weights
  Epoch 30-50: Fine-tuning + SWAG snapshot collection
  Each epoch: ~28 minutes
  Total: ~24 hours
```

**You can:**
- Check job status periodically with `squeue -u hpl14`
- View output logs with `tail -f logs/retrain_mc_dropout_*.out`
- Go grab a coffee ☕

## Phase 4: Verify Completion (After ~24 hours)

### On Amarel (SSH Terminal)

```bash
# Check job status
squeue -u hpl14

# When jobs are gone from queue, verify model files
ls -lh /scratch/$USER/uq_capstone/runs/classification/mc_dropout/best_model.pth
ls -lh /scratch/$USER/uq_capstone/runs/classification/swag_classification/swag_model.pth

# Check file sizes (they should be significant)
# Expected: best_model.pth ~95MB, swag_model.pth ~80MB

# View final output logs
tail -20 logs/retrain_mc_dropout_*.out
tail -20 logs/retrain_swag_*.out

# Check for errors (look for "✗" or "FAILED")
grep -i "error\|failed" logs/retrain_mc_dropout_*.out
grep -i "error\|failed" logs/retrain_swag_*.out
```

### If Models Exist ✅
```bash
# View training history
cat runs/classification/mc_dropout/history.json | head -50
cat runs/classification/swag_classification/history.json | head -50

# Check final accuracies (should be ~90%)
grep "final\|epoch 50" logs/retrain_mc_dropout_*.out
grep "final\|epoch 50" logs/retrain_swag_*.out
```

### If Something Went Wrong ❌
See "Troubleshooting" section below

## Phase 5: Pull Results to Local Machine

### On Windows (PowerShell Terminal)

```powershell
# Pull MC Dropout model
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/mc_dropout/best_model.pth ./runs/classification/mc_dropout/

# Pull SWAG model
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/swag_classification/swag_model.pth ./runs/classification/swag_classification/

# Pull training histories
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/mc_dropout/history.json ./runs/classification/mc_dropout/
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/swag_classification/history.json ./runs/classification/swag_classification/

# Verify files exist locally
Get-ChildItem runs/classification/mc_dropout/best_model.pth
Get-ChildItem runs/classification/swag_classification/swag_model.pth
```

## Phase 6: Re-run Metrics Evaluation

### On Windows (PowerShell Terminal)

```powershell
# Navigate to project
cd c:\Users\lpnhu\Downloads\uq_capstone

# Run comprehensive metrics
python src/comprehensive_metrics.py

# Expected output:
# - runs/classification/metrics/comprehensive_results.json
# - Performance metrics table
# - Uncertainty calibration analysis
# - Conformal Risk Control evaluation

# Generate visualizations
python analysis/visualize_metrics.py

# Expected output:
# - Accuracy comparison plots
# - Calibration curves
# - Uncertainty distribution plots
# - ROC curves
```

### Output Files Generated
```
runs/classification/metrics/
  ├── comprehensive_results.json  <- Main results
  ├── accuracy_comparison.png
  ├── calibration_curves.png
  ├── uncertainty_distributions.png
  ├── roc_curves.png
  └── metrics_summary.txt
```

## Phase 7: Analyze Results

### Check Accuracy Improvements

```powershell
# Open results file
Get-Content runs/classification/metrics/comprehensive_results.json | ConvertFrom-Json | Select-Object mc_dropout, swag | Format-Table -AutoSize

# Expected:
# mc_dropout accuracy: ~90% (was 63.3% ❌)
# swag accuracy: ~90% (was 79.3% ❌)
```

### Compare Before vs After

| Metric | MC Dropout Before | MC Dropout After | SWAG Before | SWAG After |
|--------|-------------------|------------------|------------|----------|
| Accuracy | 63.3% ❌ | ~90% ✅ | 79.3% ❌ | ~90% ✅ |
| ECE | High ❌ | Lower ✅ | High ❌ | Lower ✅ |
| Uncertainty | Invalid ❌ | Valid ✅ | Poor ❌ | Good ✅ |
| CRC Coverage | N/A | ✅ | N/A | ✅ |

## Troubleshooting Guide

### ❌ Job Failed (SBATCH error log shows issues)

**If you see "FAILED" in log:**

```bash
# 1. View full error log
cat logs/retrain_mc_dropout_<job_id>.err

# 2. Common issues:
# - Out of memory: Already 32GB, unlikely
# - GPU not available: Check sinfo
# - Data not found: Check data/ directory exists
# - Model loading failed: Check baseline checkpoint exists

# 3. Resubmit job
sbatch scripts/retrain_mc_dropout.sbatch

# 4. If still fails, debug locally first
cd /scratch/$USER/uq_capstone
python src/retrain_mc_dropout.py --test-mode  # (if available)
```

### ❌ Models Exist But Accuracy is Still Low

```bash
# This shouldn't happen, but if it does:

# 1. Check model was actually initialized from baseline
grep "Loading baseline" logs/retrain_mc_dropout_*.out

# 2. Verify training was happening (check loss values)
tail -100 logs/retrain_mc_dropout_*.out | grep -i "loss\|epoch"

# 3. Check if baseline checkpoint itself is good
python -c "
import torch
baseline = torch.load('runs/classification/baseline/best_model.pth')
print('Baseline model loaded successfully')
print('Model keys:', list(baseline.keys())[:5])
"

# 4. If baseline is corrupted, we have mc_dropout_old backup
mv runs/classification/mc_dropout runs/classification/mc_dropout_retrain_bad
mv runs/classification/mc_dropout_old runs/classification/mc_dropout
```

### ❌ SCP Pull Failed (Permission or Network Issue)

```powershell
# 1. Verify SSH works
ssh hpl14@amarel.rutgers.edu "pwd"

# 2. Verify files exist on Amarel
ssh hpl14@amarel.rutgers.edu "ls -lh /scratch/hpl14/uq_capstone/runs/classification/mc_dropout/best_model.pth"

# 3. Try pull again with verbose
scp -v hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/mc_dropout/best_model.pth ./runs/classification/mc_dropout/

# 4. If still fails, copy via intermediate storage
ssh hpl14@amarel.rutgers.edu "cp runs/classification/mc_dropout/best_model.pth /home/hpl14/"
scp hpl14@amarel.rutgers.edu:/home/hpl14/best_model.pth ./runs/classification/mc_dropout/
```

## ✅ Success Criteria

After Phase 7, you should have:

- [x] MC Dropout accuracy ~90% (improvement from 63.3%)
- [x] SWAG accuracy ~90% (improvement from 79.3%)
- [x] Valid uncertainty quantification for both methods
- [x] Improved calibration metrics (ECE, MCE)
- [x] Conformal Risk Control properly calibrated
- [x] Visualizations showing improvement
- [x] Ready for presentation and final report

## 📅 Timeline Summary

| Phase | Time | Status |
|-------|------|--------|
| 1. Prepare | Done ✅ | All scripts ready |
| 2. Submit | Now | Run commands above |
| 3. Wait | 24-48 hours | Coffee break ☕ |
| 4. Verify | 5 min | Check models exist |
| 5. Pull | 15 min | SCP to local |
| 6. Evaluate | 30 min | Run metrics |
| 7. Analyze | 30 min | Review results |
| **Total** | **~25-49 hours** | Including training |

## Next Steps After Completion

1. Generate comparison plots (before vs after)
2. Write analysis summary document
3. Prepare presentation slides
4. Create final report with all findings
5. Archive old models and results for reference

---

**Ready to start?** → Run the commands in Phase 2 on Amarel!
