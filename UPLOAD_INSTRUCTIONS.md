# ============================================================================
# FINAL PRE-UPLOAD VERIFICATION REPORT
# ============================================================================

## 🚨 CRITICAL BUGS FOUND & FIXED:

### Root Causes of Failure (Jobs 47439392-47439396):
1. ❌ Wrong dataset class: BratsSliceDataset (CSV) instead of BraTSSegmentationDataset (NPZ)
2. ❌ Wrong model: UNetSmall (3 layers, tiny) instead of UNet (8 layers, 31M params)
3. ❌ Wrong data path: data/brats (doesn't exist) instead of data/brats_subset_npz
4. ❌ Wrong training script arguments (--data_root vs --data_dir, etc.)

Result: Jobs completed in 20 seconds with Dice=0.25 instead of 2-3 hours with Dice=0.80-0.85


## ✅ FILES VERIFIED CORRECT (No upload needed):
1. ✅ src/model_utils.py - Full UNet implementation (~31M parameters)
2. ✅ src/train_baseline_FIXED.py - Correct dataset + model + paths
3. ✅ src/train_ensemble_member.py - Correct imports
4. ✅ src/train_swag.py - Correct imports
5. ✅ scripts/train_baseline.sbatch - Correct paths
6. ✅ scripts/train_ensemble.sbatch - Correct paths
7. ✅ scripts/train_swag.sbatch - Correct paths
8. ✅ src/data_utils.py - BraTSSegmentationDataset implementation
9. ✅ src/swag.py - Complete SWAG implementation


## 🔧 FILES FIXED (Must upload):

### 1. src/train_mc_dropout_FIXED.py
   - Status: ✅ CREATED NEW
   - Changes: 
     * Uses BraTSSegmentationDataset (NPZ files)
     * Uses UNet (full model, 31M params)
     * Correct data paths
     * Proper dropout integration for MC Dropout

### 2. scripts/train_mc_dropout.sbatch
   - Status: ✅ FIXED
   - Changes:
     * Points to train_mc_dropout_FIXED.py
     * Changed --data_root to --data_dir
     * Changed data/brats → /scratch/hpl14/uq_capstone/data/brats_subset_npz

### 3. src/evaluate_uq.py
   - Status: ✅ FIXED
   - Changes:
     * UNetSmall → UNet with in_channels/num_classes parameters
     * BratsSliceDataset → BraTSSegmentationDataset
     * Removed CSV transform, using NPZ dataset directly
     * Fixed checkpoint loading (handles 'model_state_dict' key)

### 4. scripts/evaluate_uq.sbatch
   - Status: ✅ FIXED
   - Changes:
     * data/brats → /scratch/hpl14/uq_capstone/data/brats_subset_npz


## 📦 SINGLE SCP UPLOAD COMMAND (Copy-paste this):

```bash
scp -r src/train_mc_dropout_FIXED.py src/evaluate_uq.py scripts/train_mc_dropout.sbatch scripts/evaluate_uq.sbatch hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/
```

**Alternative - Upload entire directories if above fails:**
```bash
scp src/train_mc_dropout_FIXED.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/
scp src/evaluate_uq.py hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/src/
scp scripts/train_mc_dropout.sbatch hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/
scp scripts/evaluate_uq.sbatch hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/scripts/
```


## 🚀 AFTER UPLOAD - RUN COMMANDS:

### 1. Cancel all old jobs:
```bash
scancel 47439392 47439393 47439394 47439395 47439396
```

### 2. Test baseline first (2-3 hours):
```bash
cd /scratch/hpl14/uq_capstone
sbatch scripts/train_baseline.sbatch
```

### 3. Monitor (should take 2-3 hours, NOT 20 seconds!):
```bash
watch -n 10 squeue -u hpl14
tail -f runs/baseline/train_*.out
```

### 4. Expected results:
   - ✅ Training time: ~2-3 hours per method
   - ✅ Loss decreases: 0.5-0.6 → 0.10-0.20
   - ✅ Dice score: 0.80-0.85 (NOT 0.25!)
   - ✅ Model size: ~31M parameters

### 5. If baseline succeeds, run ALL methods:
```bash
# MC Dropout (2-3 hours)
sbatch scripts/train_mc_dropout.sbatch

# Ensemble (10-15 hours total, 5 members in parallel)
sbatch scripts/train_ensemble.sbatch

# SWAG (4-5 hours)
sbatch scripts/train_swag.sbatch
```

### 6. After all training completes, evaluate:
```bash
sbatch scripts/evaluate_uq.sbatch
```


## ⚠️ CHECKLIST BEFORE UPLOAD:

- [x] All scripts use BraTSSegmentationDataset (NPZ-based)
- [x] All scripts use UNet (31M params, NOT UNetSmall)
- [x] All paths point to data/brats_subset_npz
- [x] All sbatch scripts have correct arguments
- [x] Model loading handles checkpoint format correctly
- [x] MC Dropout has dropout_rate enabled
- [x] Evaluation script fixed for all methods


## 📊 EXPECTED OUTPUT COMPARISON:

### BEFORE (WRONG):
- Time: 20 seconds
- Dice: 0.25
- Loss: ~0.60 (not improving)
- Model: UNetSmall (tiny)

### AFTER (CORRECT):
- Time: 2-3 hours
- Dice: 0.80-0.85
- Loss: 0.50 → 0.15
- Model: UNet (31M params)


## 🎯 FILES SUMMARY:

Total files to upload: 4
- 2 Python scripts (train_mc_dropout_FIXED.py, evaluate_uq.py)
- 2 SLURM scripts (train_mc_dropout.sbatch, evaluate_uq.sbatch)

Expected upload time: 1 password entry (scp -r) or 4 entries (individual files)
