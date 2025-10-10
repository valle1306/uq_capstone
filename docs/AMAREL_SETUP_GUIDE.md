# Amarel Setup Guide for UQ Capstone Project

## 📋 Project Overview

This guide will help you upload your BraTS uncertainty quantification project to Rutgers' Amarel computing cluster and run your first experiments.

**What you have:**
- ✅ **528 BraTS slices** from 25 patients (train: 368, val: 80, test: 80)
- ✅ Data in `.npz` format (efficient, compressed)
- ✅ Each slice: 240×240 pixels with T1ce MRI modality
- ✅ Binary segmentation masks (tumor vs. non-tumor)

---

## 🚀 Step 1: Upload Data to Amarel

### Option A: Using WinSCP (Recommended for Windows)

1. **Download WinSCP**: https://winscp.net/eng/download.php

2. **Connect to Amarel:**
   - Host name: `amarel.rutgers.edu`
   - User name: Your NetID
   - Password: Your Rutgers NetID password
   - Port: 22

3. **Create project directory on Amarel** (right panel):
   ```
   /scratch/YOUR_NETID/uq_capstone/
   ```

4. **Upload folders** (drag and drop from left to right):
   - `data/brats/` → `/scratch/YOUR_NETID/uq_capstone/data/brats/`
   - `scripts/` → `/scratch/YOUR_NETID/uq_capstone/scripts/`
   - `src/` → `/scratch/YOUR_NETID/uq_capstone/src/`
   - `envs/` → `/scratch/YOUR_NETID/uq_capstone/envs/`
   - `requirements.txt` → `/scratch/YOUR_NETID/uq_capstone/`

### Option B: Using Command Line (WSL or Git Bash)

1. **Edit the upload script:**
   ```bash
   # Edit scripts/upload_to_amarel.sh
   # Change YOUR_NETID_HERE to your actual NetID
   ```

2. **Run the upload script:**
   ```bash
   bash scripts/upload_to_amarel.sh
   ```

### Option C: Using Windows Batch Script

1. **Edit the batch file:**
   ```batch
   # Edit scripts/upload_to_amarel.bat
   # Change YOUR_NETID_HERE to your actual NetID
   ```

2. **Follow the instructions** in the script

---

## 🔧 Step 2: Set Up Environment on Amarel

1. **SSH to Amarel:**
   ```bash
   ssh YOUR_NETID@amarel.rutgers.edu
   ```

2. **Navigate to your project:**
   ```bash
   cd /scratch/$USER/uq_capstone
   ```

3. **Load required modules:**
   ```bash
   module load conda
   module load cuda/11.8
   ```

4. **Create conda environment:**
   ```bash
   conda env create -f envs/conda_env.yml
   ```
   
   This will take 5-10 minutes.

5. **Activate the environment:**
   ```bash
   conda activate uq_capstone
   ```

6. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

---

## ✅ Step 3: Verify Data

**Check that data uploaded correctly:**

```bash
python scripts/validate_brats_data.py --data_root data/brats --n_samples 3
```

You should see:
- ✓ Train set: 368 samples
- ✓ Val set: 80 samples  
- ✓ Test set: 80 samples
- ✓ All files exist and load correctly

---

## 🧪 Step 4: Run Your First Test Job

### Quick Test (2 hours, 1 GPU)

```bash
# Make sure you're in the project directory
cd /scratch/$USER/uq_capstone

# Create runs directory
mkdir -p runs

# Submit test job
sbatch scripts/test_training.sbatch
```

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Watch job output in real-time
tail -f runs/test_*.out

# Check for errors
tail -f runs/test_*.err
```

### Cancel a Job (if needed)

```bash
scancel JOB_ID
```

---

## 📊 Step 5: Understanding the Data

### Dataset Statistics

```
Total: 528 slices from 25 patients
├── Training:   368 slices (69.7%)
├── Validation:  80 slices (15.2%)
└── Test:        80 slices (15.2%)
```

### Data Format

- **Image**: `.npz` file with key `'im'`
  - Shape: `(1, 240, 240)` - [channels, height, width]
  - Data type: `float32`
  - Range: [0.0, 1.0] (normalized)
  - Modality: T1ce (T1-weighted contrast-enhanced MRI)

- **Mask**: `.npz` file with key `'mask'`
  - Shape: `(240, 240)` - [height, width]
  - Data type: `uint8`
  - Values: 0 (background) or 1 (tumor)

### Loading Data Example

```python
import numpy as np

# Load one sample
img_data = np.load('data/brats/images/BraTS20_Training_013_slice078.npz')
mask_data = np.load('data/brats/masks/BraTS20_Training_013_slice078.npz')

image = img_data['im']  # Shape: (1, 240, 240)
mask = mask_data['mask']  # Shape: (240, 240)

print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
print(f"Tumor pixels: {mask.sum()} ({mask.mean()*100:.1f}%)")
```

---

## 🎯 Next Steps: Running UQ Experiments

### 1. Temperature Scaling (Simplest)
Already implemented in `src/model_utils.py`

### 2. Deep Ensembles
Use the existing ensemble script:
```bash
sbatch scripts/train_ensemble_array.sbatch
```

### 3. MC Dropout
- Modify model to include dropout layers
- Run multiple forward passes at inference

### 4. Conformal Prediction
- Calibrate on validation set
- Generate prediction sets with coverage guarantees

---

## 📝 Useful Amarel Commands

### Check GPU availability:
```bash
sinfo -p gpu --format="%20N %10c %10m %25f %10G"
```

### View your disk usage:
```bash
du -sh /scratch/$USER/uq_capstone/*
```

### Check available partitions:
```bash
sinfo
```

### Monitor job resources:
```bash
seff JOB_ID
```

### Interactive GPU session (for debugging):
```bash
srun --partition=gpu --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
```

---

## 🆘 Common Issues

### Issue: "Module not found"
**Solution:** Load modules and activate environment:
```bash
module load conda cuda/11.8
conda activate uq_capstone
```

### Issue: "Out of memory"
**Solution:** Reduce batch size or request more memory in SBATCH script:
```bash
#SBATCH --mem=32G
```

### Issue: "Job pending forever"
**Solution:** Check queue and try different partition:
```bash
squeue -u $USER
# Try: --partition=main instead of --partition=gpu
```

### Issue: "CUDA out of memory"
**Solution:** Reduce batch size or use gradient accumulation

---

## 📧 Contact & Resources

- **Amarel Documentation**: https://sites.google.com/view/cluster-user-guide
- **Amarel Support**: help@oarc.rutgers.edu
- **Your advisor**: Dr. Gemma Moran

---

## 🎉 Quick Reference

**Upload data:**
```bash
# Use WinSCP or upload_to_amarel.sh
```

**Set up environment:**
```bash
module load conda cuda/11.8
conda activate uq_capstone
```

**Submit job:**
```bash
sbatch scripts/test_training.sbatch
```

**Check status:**
```bash
squeue -u $USER
tail -f runs/test_*.out
```

**Kill job:**
```bash
scancel JOB_ID
```

---

## 📈 Your Experiment Plan

Based on Dr. Moran's suggestions:

1. ✅ **Data prepared**: Small subset (25 patients, 528 slices)
2. **Week 1**: Baseline model + temperature scaling
3. **Week 2**: Deep ensembles (5 models)
4. **Week 3**: MC Dropout + Conformal prediction
5. **Week 4**: Compare all methods, analyze results
6. **Week 5-6**: Optional: Sparse autoencoders for interpretability

**Goal**: Compare uncertainty quantification methods and understand tradeoffs between:
- Computational cost
- Calibration quality
- Reliability
- Interpretability

Good luck! 🚀
