# Import Fix Guide - SOLVED! ✓

## The Problem
Python couldn't find the `src` module because PYTHONPATH wasn't set correctly.

## The Solution
Always set PYTHONPATH before running any Python scripts on Amarel.

---

## ✅ QUICK FIX - Copy & Paste This on Amarel

```bash
# SSH to Amarel
ssh hpl14@amarel.rutgers.edu

# Run the test script
cd /scratch/$USER/uq_capstone
chmod +x test_imports.sh
bash test_imports.sh
```

**Expected output:**
```
✓ data_utils_classification imported successfully
✓ conformal_risk_control imported successfully

✓✓✓ All imports successful! ✓✓✓
```

---

## Manual Test (if you want to do it step-by-step)

```bash
# SSH to Amarel
ssh hpl14@amarel.rutgers.edu

# Go to project directory
cd /scratch/$USER/uq_capstone

# Activate conda environment
conda activate uq_capstone

# Set PYTHONPATH (THE KEY STEP!)
export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH

# Test imports
python -c "
from src.data_utils_classification import get_classification_loaders
from src.conformal_risk_control import ConformalRiskControl
print('✓ All imports successful!')
"
```

---

## What Was Fixed

### 1. ✅ Removed unused `gdown` import
- **File:** `src/data_utils_classification.py`
- **Change:** Removed `import gdown` and `import zipfile` (lines 14-20)
- **Reason:** These were placeholders that were never actually used

### 2. ✅ Added PYTHONPATH to all SLURM scripts
- **Files:**
  - `scripts/train_classifier_baseline.sbatch`
  - `scripts/train_classifier_mc_dropout.sbatch`
  - `scripts/train_classifier_ensemble.sbatch`
  - `scripts/evaluate_classification.sbatch`
- **Added line:** `export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH`
- **Location:** Right after `conda activate uq_capstone`

### 3. ✅ Updated test command in UPLOAD_COMMANDS.md
- **File:** `UPLOAD_COMMANDS.md`
- **Change:** Added `cd` and `export PYTHONPATH` before import test
- **Reason:** Ensures correct working directory and module path

### 4. ✅ Created test script
- **File:** `test_imports.sh`
- **Purpose:** Easy one-command test to verify everything works
- **Upload:** Already uploaded to Amarel

---

## Why This Happened

Python needs to know where to find your custom modules. When you write:
```python
from src.data_utils_classification import get_classification_loaders
```

Python looks for a directory called `src` in these places:
1. Current directory
2. Directories in PYTHONPATH
3. System-wide Python packages

Since we're in `/scratch/hpl14/uq_capstone` and trying to import from `src/`, we need to tell Python:
"Look in `/scratch/hpl14/uq_capstone` to find the `src` directory"

That's what `export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH` does!

---

## Verification Checklist

Run these commands on Amarel to verify everything is fixed:

```bash
# 1. Check test script exists
ls -lh /scratch/$USER/uq_capstone/test_imports.sh

# 2. Run test
cd /scratch/$USER/uq_capstone
bash test_imports.sh

# 3. Check SLURM scripts have PYTHONPATH
grep -n "PYTHONPATH" scripts/train_classifier_*.sbatch

# 4. Verify data_utils_classification.py has no gdown
grep -n "gdown" src/data_utils_classification.py  # Should return nothing
```

---

## Next Steps After Verification

Once you see `✓✓✓ All imports successful! ✓✓✓`:

### Step 6: Download Dataset
```bash
# Option A: Install Kaggle CLI
pip install kaggle

# Option B: Download locally and upload (recommended)
# On your local machine:
# 1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# 2. Download and extract
# 3. Upload: scp -r chest_xray hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/data/
```

### Step 7: Verify Dataset
```bash
cd /scratch/$USER/uq_capstone
export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH
conda activate uq_capstone

python -c "
from src.data_utils_classification import get_classification_loaders
train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
    dataset_name='chest_xray',
    data_dir='data/chest_xray',
    batch_size=16
)
print(f'✓ Dataset loaded!')
print(f'  Classes: {num_classes}')
print(f'  Train batches: {len(train_loader)}')
print(f'  Test batches: {len(test_loader)}')
"
```

### Step 8: Launch Experiments! 🚀
```bash
cd /scratch/$USER/uq_capstone
bash scripts/run_all_classification_experiments.sh

# Monitor
squeue -u $USER
```

---

## Troubleshooting

**Still getting import errors?**
1. Make sure you're in `/scratch/hpl14/uq_capstone` when running commands
2. Always run `export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH` first
3. Check conda environment: `conda activate uq_capstone`
4. Verify files exist: `ls -lh src/*.py`

**Permission denied on test_imports.sh?**
```bash
chmod +x test_imports.sh
```

**Want to make PYTHONPATH permanent?**
Add this to your `~/.bashrc`:
```bash
export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH
```
Then run: `source ~/.bashrc`

---

## Summary

✅ **Fixed Issues:**
- Removed unused `gdown` import
- Added PYTHONPATH to all scripts
- Created easy test script
- Updated documentation

✅ **Files Uploaded to Amarel:**
- `src/data_utils_classification.py` (clean version)
- `scripts/*.sbatch` (4 files with PYTHONPATH)
- `test_imports.sh` (verification script)

✅ **What to Do Now:**
1. Run `bash test_imports.sh` on Amarel
2. If you see "All imports successful", proceed to download dataset
3. Launch experiments!

---

🎉 **You're ready to go!** 🎉
