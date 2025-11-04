# 🔍 Troubleshooting: Models Not Found

**Problem:** Training appears incomplete - `mc_dropout/best_model.pth` and `swag_classification/swag_model.pth` not found

## Quick Diagnostic (Run This First)

```bash
ssh hpl14@amarel.rutgers.edu

# Navigate to project
cd /scratch/$USER/uq_capstone

# Run diagnostic script
bash scripts/diagnose_retraining.sh
```

This will tell you:
- If retraining jobs are still running
- If they completed or failed
- Where output actually went
- What's in the logs

---

## Likely Scenarios & Fixes

### Scenario 1: Jobs Are Still Running ✓

**Check:**
```bash
squeue -u hpl14
```

**If you see retrain jobs in output:**
- They're still training! Wait 12-24 more hours
- Monitor with: `tail -f logs/retrain_mc_dropout_<job_id>.out`

**Expected output:**
```
JOBID PARTITION NAME          USER       ST TIME NODES CPUS
xxxxx gpu       retrain_mc_do hpl14      R  5:30 1    1
xxxxx gpu       retrain_swag  hpl14      R  5:45 1    1
```

---

### Scenario 2: Jobs Completed But Output Missing ❌

**Check the log files:**
```bash
# Check retrain job outputs
cat logs/retrain_mc_dropout_*.out
cat logs/retrain_mc_dropout_*.err

cat logs/retrain_swag_*.out
cat logs/retrain_swag_*.err
```

**Look for:**
- ✓ "COMPLETED SUCCESSFULLY" → Good, look for saved files
- ✗ "ERROR" or "FAILED" → Training failed, see error message below
- ✗ "No such file" → Baseline checkpoint missing
- ✗ "CUDA out of memory" → Memory issue

---

### Scenario 3: Training Failed (Most Likely)

**Check error logs:**
```bash
tail -50 logs/retrain_mc_dropout_*.err
tail -50 logs/retrain_swag_*.err
```

#### Error: "No such file or directory: runs/classification/baseline/best_model.pth"

**Cause:** Baseline checkpoint missing
**Fix:**
```bash
# Check if baseline exists
ls -lh runs/classification/baseline/

# If missing, pull from GitHub (assuming it's in old backup)
git checkout runs/classification/baseline/best_model.pth 2>/dev/null || echo "Not in git"

# If still missing, restore from old models (if they exist)
ls -lh runs/classification/mc_dropout_old/
ls -lh runs/classification/swag_classification_old/
```

#### Error: "CUDA out of memory"

**Cause:** GPU memory insufficient
**Fix:**
```bash
# Check GPU memory
nvidia-smi

# Re-submit with smaller batch size (edit SBATCH files)
# Change --batch_size 32 to --batch_size 16 in scripts/retrain_*.sbatch

# Resubmit jobs
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch
```

#### Error: "ImportError: cannot import..."

**Cause:** Missing dependency or conda env issue
**Fix:**
```bash
# Verify conda environment
conda activate uq_capstone
python -c "import torch; print('PyTorch OK')"

# If env corrupted, recreate
conda env remove -n uq_capstone
conda create -n uq_capstone python=3.11 pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Resubmit jobs
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch
```

#### Error: "Time limit exceeded"

**Cause:** Training took longer than 24 hours
**Fix:**
```bash
# Edit SBATCH files to increase time
sed -i 's/--time=24:00:00/--time=36:00:00/g' scripts/retrain_*.sbatch

# Resubmit
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch
```

---

### Scenario 4: Models Exist But in Wrong Location ⚠️

**Check alternative locations:**
```bash
# Search for model files
find runs/classification -name "best_model.pth" -type f
find runs/classification -name "swag_model.pth" -type f

# Check inside output subdirectories (if created automatically)
ls -R runs/classification/mc_dropout_retrain/
ls -R runs/classification/swag_classification_retrain/
```

**If found in wrong location:**
```bash
# Move them to correct location
mv runs/classification/mc_dropout_retrain/best_model.pth runs/classification/mc_dropout/
mv runs/classification/swag_classification_retrain/swag_model.pth runs/classification/swag_classification/
```

---

## Recovery: Restart Training from Scratch

If training truly failed and needs to restart:

```bash
# 1. Check baseline still exists
ls -lh runs/classification/baseline/best_model.pth

# 2. Clean up old failed output (optional)
rm -rf runs/classification/mc_dropout
rm -rf runs/classification/swag_classification

# 3. Create fresh directories
mkdir -p runs/classification/mc_dropout
mkdir -p runs/classification/swag_classification
mkdir -p logs

# 4. Resubmit jobs
sbatch scripts/retrain_mc_dropout.sbatch
sbatch scripts/retrain_swag.sbatch

# 5. Monitor
squeue -u hpl14
tail -f logs/retrain_mc_dropout_*.out
```

---

## After Training Actually Completes

Once you see "COMPLETED SUCCESSFULLY" in logs:

```bash
# Verify models exist and have content
ls -lh runs/classification/mc_dropout/best_model.pth
ls -lh runs/classification/swag_classification/swag_model.pth

# Check file sizes (should be ~100MB+)
du -h runs/classification/mc_dropout/best_model.pth
du -h runs/classification/swag_classification/swag_model.pth

# Then proceed to metrics evaluation
sbatch scripts/eval_and_visualize_on_amarel.sbatch
```

---

## Getting Help

**If you're still stuck:**

1. Run the diagnostic script:
   ```bash
   bash scripts/diagnose_retraining.sh
   ```

2. Share the output (it will show exactly what's wrong)

3. Check the most recent job error:
   ```bash
   tail -100 logs/retrain_*.err | head -50
   ```

4. Verify conda environment:
   ```bash
   conda env list
   conda activate uq_capstone
   python -c "import src.retrain_mc_dropout; print('Import OK')"
   ```

---

**Next:** Run the diagnostic script and let me know what you see! 🔍
