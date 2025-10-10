# 📋 Amarel Deployment Checklist

## Phase 1: Local Preparation ✅ COMPLETED

- [x] Downloaded BraTS2020 data (369 patients)
- [x] Created small subset script (`prepare_small_brats_subset.py`)
- [x] Processed 25 patients → 528 slices
- [x] Validated data integrity (all files exist and load correctly)
- [x] Created validation script (`validate_brats_data.py`)
- [x] Created upload scripts (bash and Windows batch)
- [x] Created SLURM job script (`test_training.sbatch`)
- [x] Updated conda environment file
- [x] Written comprehensive documentation

**Data Ready:**
- ✅ 528 slices in `.npz` format
- ✅ Train/Val/Test splits: 368/80/80
- ✅ ~150 MB total size
- ✅ Fully validated

---

## Phase 2: Upload to Amarel 🔄 READY TO DO

### Option A: WinSCP (Recommended)
- [ ] Download WinSCP from https://winscp.net/
- [ ] Connect to `amarel.rutgers.edu` with your NetID
- [ ] Create directory: `/scratch/YOUR_NETID/uq_capstone/`
- [ ] Upload folders:
  - [ ] `data/brats/` → remote `data/brats/`
  - [ ] `scripts/` → remote `scripts/`
  - [ ] `src/` → remote `src/`
  - [ ] `envs/` → remote `envs/`
  - [ ] `notebooks/` → remote `notebooks/`
  - [ ] `requirements.txt` → remote root
- [ ] Verify upload (check file sizes match)

### Option B: Command Line
- [ ] Edit `scripts/upload_to_amarel.sh` (set YOUR_NETID)
- [ ] Run: `bash scripts/upload_to_amarel.sh`
- [ ] Check output for errors

**Estimated upload time:** 5-10 minutes (depends on connection)

---

## Phase 3: Amarel Environment Setup 🔄 READY TO DO

### 3.1 Initial Connection
```bash
ssh YOUR_NETID@amarel.rutgers.edu
```
- [ ] Successful SSH connection
- [ ] No 2FA issues

### 3.2 Navigate to Project
```bash
cd /scratch/$USER/uq_capstone
ls -lh
```
- [ ] Project directory exists
- [ ] All folders present (data, scripts, src, envs)

### 3.3 Load Modules
```bash
module purge
module load conda
module load cuda/11.8
```
- [ ] Modules loaded successfully
- [ ] No error messages

### 3.4 Create Environment
```bash
conda env create -f envs/conda_env.yml
```
- [ ] Environment creation started
- [ ] Wait 5-10 minutes for completion
- [ ] No dependency conflicts

### 3.5 Activate Environment
```bash
conda activate uq_capstone
```
- [ ] Environment activated
- [ ] Prompt shows `(uq_capstone)`

### 3.6 Verify Installation
```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
- [ ] Python 3.10.x
- [ ] PyTorch 2.x.x
- [ ] CUDA available: True

---

## Phase 4: Data Validation on Amarel 🔄 READY TO DO

```bash
python scripts/validate_brats_data.py --data_root data/brats --n_samples 5
```

**Check for:**
- [ ] Train set: 368 samples
- [ ] Val set: 80 samples
- [ ] Test set: 80 samples
- [ ] All files exist (✓ All files exist)
- [ ] Sample files load correctly
- [ ] No errors in validation

**If validation fails:**
- Check file permissions: `ls -lh data/brats/images/ | head`
- Verify CSV files: `head data/brats/train.csv`
- Re-upload if needed

---

## Phase 5: Test Job Submission 🔄 READY TO DO

### 5.1 Check GPU Availability
```bash
sinfo -p gpu --format="%20N %10c %10m %25f %10G"
```
- [ ] GPU nodes available
- [ ] Note available partitions

### 5.2 Create Output Directory
```bash
mkdir -p runs
```
- [ ] Directory created

### 5.3 Submit Test Job
```bash
sbatch scripts/test_training.sbatch
```
- [ ] Job submitted successfully
- [ ] Note job ID: __________

### 5.4 Monitor Job
```bash
# Check queue
squeue -u $USER

# Watch output (replace JOBID)
tail -f runs/test_JOBID.out

# Check errors
tail -f runs/test_JOBID.err
```

- [ ] Job appears in queue
- [ ] Job status changes to RUNNING
- [ ] Output file being created
- [ ] No errors in error file

### 5.5 Verify Test Results
**When job completes, check:**
- [ ] Exit status: 0 (success)
- [ ] Data validation passed
- [ ] Model training started
- [ ] No CUDA errors
- [ ] Output files in `runs/test_JOBID/`

```bash
# Check job efficiency
seff JOBID
```
- [ ] Memory usage reasonable (<16GB)
- [ ] CPU/GPU utilized
- [ ] Job completed without errors

---

## Phase 6: Next Steps Planning 🎯 UPCOMING

### Week 1: Baseline (This Week)
- [ ] Review `src/train_seg.py`
- [ ] Understand current training pipeline
- [ ] Run full training (50 epochs)
- [ ] Implement/verify temperature scaling
- [ ] Compute baseline metrics (accuracy, ECE)

### Week 2: Deep Ensembles
- [ ] Modify training to save multiple models
- [ ] Train 5 models with different seeds
- [ ] Implement ensemble prediction
- [ ] Compare with baseline

### Week 3: MC Dropout
- [ ] Add dropout layers to model
- [ ] Implement MC sampling at inference
- [ ] Compute epistemic/aleatoric uncertainty
- [ ] Compare with baseline and ensembles

### Week 4: Conformal Prediction
- [ ] Implement conformal predictor class
- [ ] Calibrate on validation set
- [ ] Generate prediction sets
- [ ] Measure coverage and set sizes

### Week 5-6: Analysis & Write-up
- [ ] Compare all methods
- [ ] Create visualizations
- [ ] Write results summary
- [ ] (Optional) Try sparse autoencoders

---

## 📞 Support & Resources

### If You Get Stuck:

**Amarel Issues:**
- Documentation: https://sites.google.com/view/cluster-user-guide
- Email: help@oarc.rutgers.edu
- Office Hours: Check Amarel website

**Project Issues:**
- Dr. Gemma Moran (advisor)
- Check `AMAREL_SETUP_GUIDE.md`
- Check `QUICK_START.md`

### Quick Debug Commands:

**Can't connect to Amarel:**
```bash
# Test connection
ping amarel.rutgers.edu
ssh -v YOUR_NETID@amarel.rutgers.edu
```

**Module issues:**
```bash
module avail               # List all modules
module list                # Show loaded modules
module spider conda        # Search for specific module
```

**Environment issues:**
```bash
conda env list             # List environments
conda activate uq_capstone # Activate
conda deactivate           # Deactivate
```

**Job issues:**
```bash
squeue -u $USER           # Your jobs
squeue -p gpu             # GPU partition queue
scancel JOBID             # Cancel job
scontrol show job JOBID   # Job details
```

**Disk space:**
```bash
du -sh /scratch/$USER/*   # Check usage
df -h /scratch            # Available space
```

---

## ✅ Success Criteria

**Phase 2 Success:** 
✓ All files uploaded, no missing folders

**Phase 3 Success:**
✓ Conda environment created and activated
✓ PyTorch with CUDA available

**Phase 4 Success:**
✓ All 528 samples validated
✓ No file loading errors

**Phase 5 Success:**
✓ Test job completes successfully
✓ Model trains for 2 epochs
✓ No CUDA/memory errors

**Overall Success:**
✓ Can submit jobs on Amarel
✓ Data loads correctly
✓ Training pipeline works
✓ Ready for full experiments

---

## 📝 Notes & Observations

**Date Started:** _______________

**Date Completed Phase 2:** _______________

**Date Completed Phase 3:** _______________

**Date Completed Phase 4:** _______________

**Date Completed Phase 5:** _______________

**Issues Encountered:**

```
[Write any issues you encounter and how you solved them]
```

**Tips for Future Reference:**

```
[Write any useful commands or tips you discover]
```

---

## 🎉 Celebration Checklist

When you complete each phase:
- [ ] Phase 2: Data uploaded! 🎈
- [ ] Phase 3: Environment ready! 🎯
- [ ] Phase 4: Data validated! ✨
- [ ] Phase 5: First job successful! 🚀
- [ ] Ready to start real experiments! 🔬

---

**Last Updated:** October 10, 2025
**Status:** Phases 1 ✅ | Phases 2-5 Ready to Execute
