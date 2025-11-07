# AMAREL COMMANDS - SWAG Proper Retraining
**Date:** November 7, 2025  
**Purpose:** Complete commands to retrain SWAG following Maddox et al. 2019 paper

---

## 🚀 STEP-BY-STEP COMMANDS

### Step 1: SSH to Amarel and Pull Latest Code

```bash
ssh hpl14@amarel.rutgers.edu
```

Once logged in:
```bash
cd /scratch/$USER/uq_capstone
git pull origin main
```

Expected output:
```
Updating 828b7ec..ac5bb5b
 5 files changed, 1041 insertions(+)
 create mode 100644 QUICK_START_SWAG_PROPER.md
 create mode 100644 SWAG_IMPLEMENTATION_ANALYSIS.md
 create mode 100644 SWAG_RETRAIN_DECISION.md
 create mode 100644 scripts/retrain_swag_proper.sbatch
 create mode 100644 src/retrain_swag_proper.py
```

---

### Step 2: Backup Old SWAG Model (Optional)

```bash
# Check if old model exists
ls -lh runs/classification/swag_classification/

# Backup (optional - creates backup folder)
mv runs/classification/swag_classification runs/classification/swag_classification_adam_old

# Create fresh directory
mkdir -p runs/classification/swag_classification
mkdir -p logs
```

---

### Step 3: Submit SWAG Retraining Job

```bash
# Submit the job
sbatch scripts/retrain_swag_proper.sbatch
```

Expected output:
```
Submitted batch job XXXXXX
```

**Save this job ID!** You'll need it to monitor progress.

---

### Step 4: Check Job Status

```bash
# Check if job is running
squeue -u hpl14

# See detailed job info (replace JOBID)
scontrol show job JOBID
```

Expected output:
```
  JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
 XXXXXX       gpu swag_pro    hpl14  R       1:23      1 node-name
```

Status codes:
- `PD` = Pending (waiting for resources)
- `R` = Running
- `CG` = Completing
- `CD` = Completed

---

### Step 5: Monitor Training Progress

```bash
# Watch output in real-time (replace JOBID)
tail -f logs/swag_proper_JOBID.out

# Exit tail: Press Ctrl+C

# Or view entire log
less logs/swag_proper_JOBID.out
```

What to look for:
```
✓ SGD with momentum=0.9
✓ Weight decay (L2): 0.0001
✓ SWAG snapshots start: epoch 30
✓ Training from scratch (random init)

Epoch 1/50
Learning Rate: 0.050000
Train Acc: XX.XX%
Val Acc:   XX.XX%
Test Acc:  XX.XX%
...
✓ Collected SWAG snapshot 1 (total: 1)
...
Updating Batch Normalization statistics...
✓ Batch norm updated
Training Complete!
```

---

### Step 6: Check Training Results (After ~24 hours)

```bash
# Verify training completed
ls -lh runs/classification/swag_classification/

# Should see:
# - swag_model.pth (~900MB)
# - best_model.pth (~40MB)
# - training_history.json
# - config.json
```

```bash
# Check configuration
cat runs/classification/swag_classification/config.json
```

Should show:
```json
{
  "optimizer": "SGD",
  "momentum": 0.9,
  "weight_decay": 0.0001,
  "scheduler": "CosineAnnealingLR + SWALR",
  "batch_norm_update": true,
  "training_from": "scratch",
  "paper": "Maddox et al. 2019"
}
```

```bash
# Check final accuracies
python -c "
import json
with open('runs/classification/swag_classification/training_history.json') as f:
    hist = json.load(f)
print(f'Final train accuracy: {hist[\"train_acc\"][-1]:.2f}%')
print(f'Final validation accuracy: {hist[\"val_acc\"][-1]:.2f}%')
print(f'Final test accuracy: {hist[\"test_acc\"][-1]:.2f}%')
print(f'SWAG snapshots collected: {hist[\"swag_snapshots\"]}')
print()
print(f'Best validation accuracy: {max(hist[\"val_acc\"]):.2f}%')
print(f'Overfitting check: Val={hist[\"val_acc\"][-1]:.2f}% vs Test={hist[\"test_acc\"][-1]:.2f}%')
"
```

Expected:
```
Final train accuracy: ~92-95%
Final validation accuracy: ~90-92%
Final test accuracy: ~90-92%
SWAG snapshots collected: 20
Overfitting check: Val≈Test (difference <2%)
```

---

### Step 7: Re-run Comprehensive Evaluation

```bash
# Submit evaluation job
sbatch scripts/eval_and_visualize_on_amarel.sbatch

# Check status
squeue -u hpl14

# Monitor (replace JOBID)
tail -f logs/eval_viz_JOBID.out
```

Wait for completion (~15 minutes), then:

```bash
# Check new metrics
cat runs/classification/metrics/metrics_summary.csv
```

Expected new results:
```
Method,Accuracy (%),ECE,Brier,FNR,Mean Unc
Baseline,91.67,0.0498,0.0704,0.0833,
MC Dropout,85.26,0.1172,0.1246,0.1474,8.18e-05
Deep Ensemble,91.67,0.0271,0.0630,0.0833,0.0167
SWAG,~90-91,~0.05-0.10,~0.07,~0.09,~1e-04  ← IMPROVED!
```

---

### Step 8: Download Results to Windows

On Amarel, verify files exist:
```bash
ls -lh runs/classification/metrics/
```

Then **on your Windows machine**:
```powershell
cd C:\Users\lpnhu\Downloads\uq_capstone

# Option 1: Use download script
.\download_results.ps1

# Option 2: Manual download
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics/metrics_summary.csv runs\classification\metrics\
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics/comprehensive_metrics.json runs\classification\metrics\
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/metrics/*.png runs\classification\metrics\
```

---

## ⚡ Quick Copy-Paste Commands

### All Commands in Sequence (Copy-Paste Ready)

```bash
# 1. SSH and navigate
ssh hpl14@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone

# 2. Pull latest code
git pull origin main

# 3. Backup old model
mv runs/classification/swag_classification runs/classification/swag_classification_adam_old
mkdir -p runs/classification/swag_classification logs

# 4. Submit retraining
sbatch scripts/retrain_swag_proper.sbatch

# 5. Check status
squeue -u hpl14

# 6. Monitor (replace JOBID with actual number)
tail -f logs/swag_proper_JOBID.out
```

---

## 🔍 Troubleshooting

### If job fails immediately:
```bash
# Check error log
cat logs/swag_proper_JOBID.err

# Common issue: conda environment
source ~/.bashrc
conda activate uq_capstone
```

### If CUDA out of memory:
```bash
# Edit the sbatch script to reduce batch size
nano scripts/retrain_swag_proper.sbatch
# Change: --batch_size 32 to --batch_size 16
# Resubmit
sbatch scripts/retrain_swag_proper.sbatch
```

### If accuracy still low (~83%):
```bash
# Check if weight decay was actually used
grep "weight_decay" runs/classification/swag_classification/config.json

# Check training logs for overfitting
grep "Val Acc" logs/swag_proper_JOBID.out | tail -10
grep "Test Acc" logs/swag_proper_JOBID.out | tail -10
```

---

## ✅ Success Criteria

After everything completes, verify:
- [ ] SWAG test accuracy ≥ 90%
- [ ] No major overfitting (val_acc ≈ test_acc, within 2%)
- [ ] ECE < 0.10 (good calibration)
- [ ] Config shows: SGD, weight_decay=1e-4, SWALR, batch_norm_update=true
- [ ] 20 SWAG snapshots collected
- [ ] Comprehensive metrics show SWAG competitive with Ensemble

---

## 📊 Expected Timeline

| Time | Event |
|------|-------|
| Now | Submit job |
| +5 min | Job starts running |
| +30 min | Epoch 1-2 complete |
| +12 hours | Epoch 25 complete |
| +18 hours | Epoch 30 (SWAG collection starts) |
| +24 hours | Training complete, batch norm updated |
| +24h 15min | Evaluation complete |
| +24h 30min | Results downloaded |

---

## 📝 What Changed from Previous Implementation

| Aspect | Old (Adam) | New (SGD Paper-Correct) |
|--------|------------|-------------------------|
| Optimizer | Adam | SGD + momentum=0.9 |
| Weight Decay | 0 (NONE!) | 1e-4 (L2 reg) |
| LR Schedule | CosineAnnealing | Cosine + SWALR |
| Batch Norm | Not updated | Updated after SWA |
| Initialization | From baseline | From scratch |
| Expected Acc | 83% | ~90%+ |

---

## 🎯 After Success

Once SWAG achieves ~90%:
1. Update presentation with new results
2. Document proper implementation in thesis
3. Compare all methods fairly (all ~90% now!)
4. Ready for publication/defense

**This fixes the critical SWAG underperformance issue!** 🎉
