# 🚀 QUICK START - Run UQ Experiments Now!

## Step 1: Upload Code (Run from Windows)

Open **NEW Command Prompt** (not the SSH terminal):

```cmd
cd C:\Users\lpnhu\Downloads\uq_capstone
scripts\upload_uq_code.bat
```

This uploads all UQ code to Amarel (~2 minutes).

---

## Step 2: Run Experiments (On Amarel)

### Option A: Run Everything Automatically ⭐ RECOMMENDED

```bash
# Make sure you're on Amarel
cd /scratch/hpl14/uq_capstone

# Submit all jobs with dependencies
bash scripts/run_all_experiments.sh

# Monitor progress
squeue -u hpl14

# Check results (after ~6-8 hours)
cat runs/evaluation/results.json
```

### Option B: Run Individual Jobs

```bash
cd /scratch/hpl14/uq_capstone

# Submit each job separately
sbatch scripts/train_baseline.sbatch
sbatch scripts/train_mc_dropout.sbatch
sbatch scripts/train_ensemble.sbatch

# Wait for all to finish, then:
sbatch scripts/evaluate_uq.sbatch
```

---

## Step 3: Monitor Jobs

```bash
# Check queue
squeue -u hpl14

# Watch baseline training
tail -f runs/baseline/train_*.out

# Check for errors
tail -f runs/baseline/train_*.err
```

---

## Step 4: Get Results (After Completion)

```bash
# View metrics
cat runs/evaluation/results.json

# Download comparison plot
scp hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/evaluation/comparison.png .
```

---

## ⏱️ Timeline

| Job | Time | Status |
|-----|------|--------|
| Baseline | 2-3 hrs | Trains standard U-Net |
| MC Dropout | 2-3 hrs | Trains with dropout |
| Ensemble (5x) | 2-3 hrs | Parallel, 5 models |
| Evaluation | 30 min | Compares all methods |
| **TOTAL** | **6-8 hrs** | Can run overnight |

---

## 📊 What You'll Get

**5 UQ Methods Compared:**
1. ✅ Baseline (no UQ)
2. ✅ Temperature Scaling (calibration)
3. ✅ MC Dropout (single model UQ)
4. ✅ Deep Ensemble (best UQ, expensive)
5. ✅ Conformal Prediction (guarantees)

**Results:**
- `runs/evaluation/results.json` - All metrics
- `runs/evaluation/comparison.png` - Plots
- Dice scores, ECE, uncertainty metrics

---

## ❓ Troubleshooting

**Jobs not starting?**
```bash
squeue -u hpl14  # Check queue position
scontrol show job JOBID  # See why pending
```

**Job failed?**
```bash
cat runs/baseline/train_*.err  # Check error log
```

**Need help?**
- Read: `UQ_EXPERIMENTS_GUIDE.md`
- Or: Ask Dr. Moran

---

## ✨ That's It!

Just run Step 1 and Step 2, then wait for results!

**Expected completion: Tomorrow morning if started now** ⏰

