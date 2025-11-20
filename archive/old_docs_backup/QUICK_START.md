# Quick Start Guide - Running on Amarel

## 🚀 First Time Setup (One-time)

### 1. Upload Data to Amarel
Use **WinSCP** (easiest for Windows):
- Download: https://winscp.net/
- Connect to: `amarel.rutgers.edu`
- Upload `data/brats/`, `scripts/`, `src/`, `envs/` to `/scratch/YOUR_NETID/uq_capstone/`

### 2. Set Up Environment
```bash
ssh YOUR_NETID@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone
module load conda cuda/11.8
conda env create -f envs/conda_env.yml
conda activate uq_capstone
```

### 3. Verify Data
```bash
python scripts/validate_brats_data.py --data_root data/brats
```

---

## 🎯 Every Time You Want to Run Jobs

### 1. SSH to Amarel
```bash
ssh YOUR_NETID@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone
```

### 2. Load Modules
```bash
module load conda cuda/11.8
conda activate uq_capstone
```

### 3. Submit Job
```bash
sbatch scripts/test_training.sbatch
```

### 4. Check Status
```bash
# List your jobs
squeue -u $USER

# Watch output
tail -f runs/test_*.out

# Cancel job if needed
scancel JOB_ID
```

---

## 📊 Your Dataset

- **528 slices** from **25 patients**
- **Train**: 368 slices
- **Val**: 80 slices  
- **Test**: 80 slices
- **Format**: 240×240 T1ce MRI, normalized to [0,1]

---

## 🔥 Common Commands

```bash
# View running jobs
squeue -u $USER

# Check job details
seff JOB_ID

# View output
cat runs/test_12345.out

# Check GPU availability
sinfo -p gpu

# Disk usage
du -sh data/

# Interactive session (debugging)
srun --partition=gpu --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash
```

---

## 🆘 Troubleshooting

**Can't find module?**
```bash
module load conda cuda/11.8
conda activate uq_capstone
```

**Out of memory?**
Edit `.sbatch` file: `#SBATCH --mem=32G`

**Job stuck in queue?**
Try different partition: `#SBATCH --partition=main`

---

## 📝 Files You Created

✅ `scripts/prepare_small_brats_subset.py` - Data preparation  
✅ `scripts/validate_brats_data.py` - Data validation  
✅ `scripts/test_training.sbatch` - Test job script  
✅ `scripts/upload_to_amarel.sh` - Upload helper  
✅ `data/brats/` - Your processed dataset  

---

## 🎯 Your UQ Experiment Plan

1. **Baseline** (Week 1)
   - Train single model
   - Apply temperature scaling
   
2. **Deep Ensembles** (Week 2)
   - Train 5 models with different initializations
   - Average predictions
   
3. **MC Dropout** (Week 3)
   - Multiple forward passes
   - Estimate uncertainty
   
4. **Conformal Prediction** (Week 3)
   - Calibrate on validation set
   - Generate prediction sets
   
5. **Compare & Analyze** (Week 4)
   - ECE (calibration)
   - Accuracy
   - Computational cost

---

See **AMAREL_SETUP_GUIDE.md** for detailed instructions!
