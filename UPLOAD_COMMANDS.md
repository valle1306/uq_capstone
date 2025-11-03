# Quick Upload Commands - Copy & Paste These!

## ⚠️ IMPORTANT: Import Fix Applied!
All SLURM scripts have been updated to include PYTHONPATH.
A test script `test_imports.sh` is included for easy verification.

**See `IMPORT_FIX_GUIDE.md` for detailed troubleshooting.**

---

## ⚠️ FIRST: Update Your Username
Replace `YOUR_USERNAME` with your actual Amarel username in all commands below!

---

## Step 1: Upload Source Files (6 files)

```powershell
# Run these commands from: C:\Users\lpnhu\Downloads\uq_capstone

scp src/data_utils_classification.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/

scp src/conformal_risk_control.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/

scp src/train_classifier_baseline.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/

scp src/train_classifier_mc_dropout.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/

scp src/train_classifier_ensemble_member.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/

scp src/evaluate_uq_classification.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/
```

---

## Step 2: Upload SLURM Scripts (5 files)

```powershell
scp scripts/train_classifier_baseline.sbatch hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/

scp scripts/train_classifier_mc_dropout.sbatch hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/

scp scripts/train_classifier_ensemble.sbatch hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/

scp scripts/evaluate_classification.sbatch hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/

scp scripts/run_all_classification_experiments.sh hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/
```

---

## Step 3: Upload Test Script

```powershell
scp test_imports.sh hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/
```

---

## Step 4: Upload Documentation (4 files)

```powershell
scp docs/CLASSIFICATION_QUICK_START.md YOUR_USERNAME@amarel.rutgers.edu:/scratch/YOUR_USERNAME/uq_capstone/docs/

scp docs/CLASSIFICATION_SETUP_GUIDE.md YOUR_USERNAME@amarel.rutgers.edu:/scratch/YOUR_USERNAME/uq_capstone/docs/

scp docs/CLASSIFICATION_IMPLEMENTATION_SUMMARY.md YOUR_USERNAME@amarel.rutgers.edu:/scratch/YOUR_USERNAME/uq_capstone/docs/

scp docs/DEPLOYMENT_CHECKLIST.md YOUR_USERNAME@amarel.rutgers.edu:/scratch/YOUR_USERNAME/uq_capstone/docs/
```

---

## Step 5: SSH to Amarel and Make Scripts Executable

```bash
ssh hpl14@amarel.rutgers.edu

cd /scratch/$USER/uq_capstone

chmod +x test_imports.sh
chmod +x scripts/run_all_classification_experiments.sh
chmod +x scripts/*.sbatch

# Verify files uploaded
ls -lh src/*classification*
ls -lh scripts/*classifier*
ls -lh test_imports.sh
```

---

## Step 6: Test Imports

### Easy Way (Recommended):
```bash
# On Amarel
cd /scratch/$USER/uq_capstone
chmod +x test_imports.sh
bash test_imports.sh
```

### Manual Way:
```bash
# On Amarel
cd /scratch/$USER/uq_capstone
conda activate uq_capstone

# IMPORTANT: Set PYTHONPATH (this is the key!)
export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH

python -c "
from src.data_utils_classification import get_classification_loaders
from src.conformal_risk_control import ConformalRiskControl
print('✓ All imports successful!')
"
```

If you see "✓ All imports successful!" - you're ready to download the dataset!

**Note:** The PYTHONPATH line is CRITICAL. Without it, Python can't find the `src` module.

---

## Step 7: Download Dataset

### Option A: Using Kaggle CLI (on Amarel)

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle API
mkdir -p ~/.kaggle

# You need to upload your kaggle.json file (get from kaggle.com/settings)
# On your local machine:
# scp kaggle.json YOUR_USERNAME@amarel.rutgers.edu:~/.kaggle/
# Then on Amarel:
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
cd /scratch/$USER/uq_capstone/data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d chest_xray/
rm chest-xray-pneumonia.zip
```

### Option B: Download Locally Then Upload (Recommended if Kaggle API is complicated)

**On your local machine:**
1. Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click "Download" (requires Kaggle login)
3. Extract the zip file to get `chest_xray` folder
4. Upload:

```powershell
# From C:\Users\lpnhu\Downloads (or wherever you extracted it)
scp -r chest_xray YOUR_USERNAME@amarel.rutgers.edu:/scratch/YOUR_USERNAME/uq_capstone/data/
```

---

## Step 7: Verify Dataset

```bash
# On Amarel
cd /scratch/$USER/uq_capstone

# Activate environment and set PYTHONPATH
conda activate uq_capstone
export PYTHONPATH=/scratch/$USER/uq_capstone:$PYTHONPATH

# Check directory structure
ls -R data/chest_xray/ | head -20

# Test data loading
python -c "
from src.data_utils_classification import get_classification_loaders
try:
    train_loader, cal_loader, test_loader, num_classes = get_classification_loaders(
        dataset_name='chest_xray',
        data_dir='data/chest_xray',
        batch_size=16
    )
    print(f'✓ Dataset loaded successfully!')
    print(f'  Classes: {num_classes}')
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Cal batches: {len(cal_loader)}')
    print(f'  Test batches: {len(test_loader)}')
except Exception as e:
    print(f'✗ Error: {e}')
"
```

---

## Step 8: Launch Experiments! 🚀

```bash
# On Amarel
cd /scratch/$USER/uq_capstone

# Run all experiments
bash scripts/run_all_classification_experiments.sh

# Monitor jobs
squeue -u $USER

# Check logs
tail -f runs/classification/baseline/train_*.out
```

---

## Quick Checklist

- [ ] Step 1: Source files uploaded (6 files)
- [ ] Step 2: SLURM scripts uploaded (5 files)
- [ ] Step 3: Test script uploaded (test_imports.sh)
- [ ] Step 4: Documentation uploaded (4 files) - OPTIONAL
- [ ] Step 5: Scripts made executable
- [ ] Step 6: Imports tested successfully ← **MUST SEE "All imports successful!"**
- [ ] Step 7: Dataset downloaded
- [ ] Step 8: Dataset verified
- [ ] Step 9: Experiments launched!

---

## Troubleshooting

**"Permission denied (publickey)"**
- Make sure you're using YOUR Amarel username
- Check your SSH keys are set up

**"No such file or directory"**
- Make sure you're in `C:\Users\lpnhu\Downloads\uq_capstone` when running scp
- Make sure the remote directory exists: `/scratch/YOUR_USERNAME/uq_capstone`

**"Import errors" on Amarel**
- Activate environment: `conda activate uq_capstone`
- Install missing packages: `pip install torch torchvision scikit-learn tqdm`

---

## What Happens Next?

After launching experiments:
- **Day 1**: Baseline and MC Dropout training start (~12h each)
- **Day 1-2**: Ensemble training (5 members, ~24h total)
- **Day 2-3**: Evaluation runs (~8h) after all training completes
- **Day 3**: Results available in `runs/classification/evaluation/all_results.json`

**Total time: 2-3 days** (jobs run in parallel on GPU nodes)

---

## Getting Results

```powershell
# On your local machine (after experiments complete)
scp YOUR_USERNAME@amarel.rutgers.edu:/scratch/YOUR_USERNAME/uq_capstone/runs/classification/evaluation/all_results.json ./

# View results
cat all_results.json
```

---

Good luck! 🎉
