# 🚀 START HERE - Running on Amarel

**Date:** October 10, 2025  
**Status:** Data prepared locally ✅ | Ready to upload to Amarel

---

## 📋 What You Have Ready

✅ **528 BraTS slices** prepared and validated  
✅ **All scripts** created and tested  
✅ **Documentation** complete  
✅ **Upload scripts** ready to run  

**You are ready to go!**

---

## 🎯 OPTION 1: Automated Upload (EASIEST)

### Windows PowerShell (Recommended)

1. **Open PowerShell** in this directory:
   ```
   Right-click in this folder → "Open in Terminal" or "Open PowerShell window here"
   ```

2. **Run the upload script:**
   ```powershell
   .\upload_amarel.ps1
   ```

3. **Follow the prompts:**
   - Enter your Rutgers NetID
   - Enter your password when prompted (multiple times for each file transfer)
   - Wait 5-10 minutes for upload to complete

4. **When asked, open SSH connection** (press 'y')

5. **On Amarel, run:**
   ```bash
   cd /scratch/$USER/uq_capstone
   chmod +x amarel_setup.sh
   bash amarel_setup.sh
   ```

6. **Wait ~10 minutes** for environment setup, then submit test job:
   ```bash
   sbatch scripts/test_training.sbatch
   squeue -u $USER
   ```

**That's it! 🎉**

---

## 🎯 OPTION 2: Manual Upload with WinSCP (VISUAL)

### Step 1: Download WinSCP
- Go to: https://winscp.net/
- Download and install

### Step 2: Connect to Amarel
- **Host name:** `amarel.rutgers.edu`
- **User name:** Your NetID
- **Password:** Your Rutgers password
- **Port:** 22
- Click "Login"

### Step 3: Create Directory
On the **right side** (Amarel), navigate to:
```
/scratch/YOUR_NETID/
```

Right-click → New → Directory → Name it: `uq_capstone`

### Step 4: Upload Folders
From **left side** (your computer), drag and drop:
- `data/brats/` → `/scratch/YOUR_NETID/uq_capstone/data/`
- `scripts/` → `/scratch/YOUR_NETID/uq_capstone/scripts/`
- `src/` → `/scratch/YOUR_NETID/uq_capstone/src/`
- `envs/` → `/scratch/YOUR_NETID/uq_capstone/envs/`
- `requirements.txt` → `/scratch/YOUR_NETID/uq_capstone/`

Wait for upload to complete (~5-10 minutes)

### Step 5: Open SSH Session
In WinSCP, click **"Commands"** → **"Open Terminal"** or use PuTTY

### Step 6: Setup Environment
Run these commands:
```bash
cd /scratch/$USER/uq_capstone
module load conda cuda/11.8
conda env create -f envs/conda_env.yml
conda activate uq_capstone
python scripts/validate_brats_data.py --data_root data/brats
```

### Step 7: Submit Job
```bash
sbatch scripts/test_training.sbatch
squeue -u $USER
```

---

## 🎯 OPTION 3: Pure Command Line

### If you prefer typing commands:

```powershell
# 1. Test connection
ssh YOUR_NETID@amarel.rutgers.edu

# 2. Create directories (from Windows)
ssh YOUR_NETID@amarel.rutgers.edu "mkdir -p /scratch/YOUR_NETID/uq_capstone/{data/brats,scripts,src,envs,runs}"

# 3. Upload files (run each line separately)
scp -r data\brats YOUR_NETID@amarel.rutgers.edu:/scratch/YOUR_NETID/uq_capstone/data/
scp scripts\*.py YOUR_NETID@amarel.rutgers.edu:/scratch/YOUR_NETID/uq_capstone/scripts/
scp scripts\*.sbatch YOUR_NETID@amarel.rutgers.edu:/scratch/YOUR_NETID/uq_capstone/scripts/
scp src\*.py YOUR_NETID@amarel.rutgers.edu:/scratch/YOUR_NETID/uq_capstone/src/
scp envs\conda_env.yml YOUR_NETID@amarel.rutgers.edu:/scratch/YOUR_NETID/uq_capstone/envs/
scp requirements.txt YOUR_NETID@amarel.rutgers.edu:/scratch/YOUR_NETID/uq_capstone/
```

Replace `YOUR_NETID` with your actual NetID everywhere!

---

## ✅ Verification Checklist

After upload, SSH to Amarel and check:

```bash
ssh YOUR_NETID@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone
```

### Check files are there:
```bash
ls -lh
# Should see: data/ scripts/ src/ envs/ requirements.txt

ls data/brats/
# Should see: images/ masks/ train.csv val.csv test.csv

ls data/brats/images/ | wc -l
# Should show: 528 files

ls data/brats/masks/ | wc -l
# Should show: 528 files
```

If all looks good ✅, proceed with environment setup!

---

## 🔧 Environment Setup on Amarel

Once files are uploaded, run these commands **on Amarel**:

```bash
# Navigate to project
cd /scratch/$USER/uq_capstone

# Load modules
module purge
module load conda
module load cuda/11.8

# Create environment (takes 5-10 min)
conda env create -f envs/conda_env.yml

# Activate environment
conda activate uq_capstone

# Verify installation
python --version
# Should show: Python 3.10.x

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Should show: PyTorch: 2.x.x

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
# Should show: CUDA available: True

# Validate data
python scripts/validate_brats_data.py --data_root data/brats --n_samples 3
# Should show: ✓ VALIDATION PASSED
```

---

## 🚀 Submit Your First Job

```bash
# Create output directory
mkdir -p runs

# Submit job
sbatch scripts/test_training.sbatch

# Check status
squeue -u $USER

# Watch output (replace JOBID with actual number)
tail -f runs/test_JOBID.out

# Press Ctrl+C to stop watching
```

### Job Success Indicators:
- ✅ Job appears in `squeue`
- ✅ State changes from PD (pending) to R (running)
- ✅ Output file shows "Data validation passed"
- ✅ Model training starts
- ✅ No errors in `runs/test_JOBID.err`

---

## 🆘 Troubleshooting

### Can't SSH to Amarel?
- Check NetID spelling
- Verify password
- May need Rutgers VPN
- Contact: help@oarc.rutgers.edu

### Upload failed?
- Check internet connection
- Try WinSCP (more reliable for large files)
- Upload in smaller batches

### Conda environment fails?
- Check: `module list` shows conda
- Try: `module load conda/latest`
- Clean and retry: `conda clean --all`

### Job fails immediately?
- Check: `cat runs/test_JOBID.err`
- Verify: Data uploaded correctly
- Try: Interactive session to debug

### Need help?
- See: `AMAREL_SETUP_GUIDE.md` (detailed guide)
- See: `AMAREL_COMMANDS.txt` (all commands)
- See: `QUICK_START.md` (quick reference)

---

## 📞 Getting Help

**Amarel Support:**
- Website: https://sites.google.com/view/cluster-user-guide
- Email: help@oarc.rutgers.edu
- Office hours: Check website

**Your Advisor:**
- Dr. Gemma Moran

**Documentation:**
- `AMAREL_SETUP_GUIDE.md` - Complete walkthrough
- `AMAREL_CHECKLIST.md` - Step-by-step checklist
- `AMAREL_COMMANDS.txt` - Command reference
- `QUICK_START.md` - Quick commands

---

## 🎉 What Happens After Test Job?

Once your test job completes successfully:

1. ✅ **You've validated** the entire pipeline works
2. 🎯 **Next:** Train baseline model (50 epochs)
3. 🎯 **Next:** Implement Deep Ensembles
4. 🎯 **Next:** Implement MC Dropout
5. 🎯 **Next:** Implement Conformal Prediction
6. 🎯 **Next:** Compare all methods

**Timeline:** ~4-5 weeks to complete all experiments

---

## 🏁 Ready? Let's Go!

**Choose your upload method:**
- 🔷 **Easy:** Run `.\upload_amarel.ps1` in PowerShell
- 🔷 **Visual:** Use WinSCP GUI
- 🔷 **Manual:** Follow commands in `AMAREL_COMMANDS.txt`

**Then follow the on-screen instructions!**

Good luck! 🚀

---

**Last Updated:** October 10, 2025  
**Status:** Ready to execute  
**Estimated Time:** 30-45 minutes for complete setup
