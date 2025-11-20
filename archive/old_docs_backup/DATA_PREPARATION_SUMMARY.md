# Data Preparation Summary - October 10, 2025

## ✅ Completed Tasks

### 1. Data Analysis & Validation
- ✅ Examined BraTS2020 dataset structure (369 patients total)
- ✅ Verified NIfTI file formats (t1, t1ce, t2, flair, seg)
- ✅ Each patient has 5 files (~155 slices per volume)

### 2. Small Subset Creation
- ✅ Created `scripts/prepare_small_brats_subset.py`
- ✅ Selected **25 random patients** for initial experiments
- ✅ Converted NIfTI volumes to 2D slices in `.npz` format
- ✅ Applied preprocessing:
  - T1ce modality only (contrast-enhanced)
  - Slice stride: 3 (every 3rd slice)
  - Min tumor pixels: 50 (skip nearly empty slices)
  - Normalized to [0, 1] range
  - Binary masks (tumor vs. background)

### 3. Dataset Statistics
```
Total: 528 slices from 25 patients
├── Training:   368 slices (69.7%)
├── Validation:  80 slices (15.2%)
└── Test:        80 slices (15.2%)

File format:
- Images: (1, 240, 240) float32, range [0.0, 1.0]
- Masks:  (240, 240) uint8, values {0, 1}
```

### 4. Data Validation
- ✅ Created `scripts/validate_brats_data.py`
- ✅ All 528 files validated successfully
- ✅ Confirmed correct data format and ranges
- ✅ Verified CSV files reference correct paths

### 5. Amarel Upload Scripts
- ✅ Created `scripts/upload_to_amarel.sh` (Linux/Mac/WSL)
- ✅ Created `scripts/upload_to_amarel.bat` (Windows)
- ✅ Includes WinSCP instructions for easy GUI upload

### 6. Amarel Job Scripts
- ✅ Created `scripts/test_training.sbatch`
  - 2-hour job with 1 GPU
  - Data validation + quick training test
  - Error checking and logging

### 7. Environment Configuration
- ✅ Updated `envs/conda_env.yml`
  - Python 3.10
  - PyTorch 2.0+
  - All required dependencies
  - Jupyter for notebooks

### 8. Documentation
- ✅ Created `AMAREL_SETUP_GUIDE.md` (comprehensive guide)
- ✅ Created `QUICK_START.md` (quick reference)
- ✅ Generated `data/brats/dataset_summary.txt`

---

## 📁 Files Created/Modified

### Scripts
```
scripts/
├── prepare_small_brats_subset.py   [NEW] Data preparation
├── validate_brats_data.py          [NEW] Data validation
├── upload_to_amarel.sh             [NEW] Upload helper (Bash)
├── upload_to_amarel.bat            [NEW] Upload helper (Windows)
└── test_training.sbatch            [NEW] SLURM test job
```

### Data
```
data/brats/
├── images/                         528 .npz files
├── masks/                          528 .npz files
├── train.csv                       368 samples
├── val.csv                         80 samples
├── test.csv                        80 samples
└── dataset_summary.txt             Statistics
```

### Documentation
```
├── AMAREL_SETUP_GUIDE.md           [NEW] Complete setup guide
├── QUICK_START.md                  [NEW] Quick reference
└── envs/conda_env.yml              [UPDATED] Fixed dependencies
```

---

## 🎯 Next Steps for Amarel

### Immediate (Today/Tomorrow)
1. **Upload data to Amarel** using WinSCP or upload scripts
2. **Set up conda environment** on Amarel
3. **Run validation** to confirm upload was successful
4. **Submit test job** to verify everything works

### This Week
1. Review `src/train_seg.py` to understand training pipeline
2. Run baseline training with temperature scaling
3. Start implementing MC Dropout

### Next 2-3 Weeks (As per Dr. Moran's guidance)
1. Implement Deep Ensembles (5 models)
2. Implement MC Dropout
3. Implement Conformal Prediction
4. Compare all methods on test set

---

## 📊 Data Characteristics

### Why This Subset is Good for Initial Experiments:
- ✅ **Manageable size**: 528 slices vs. 50,000+ from full dataset
- ✅ **Representative**: 25 patients randomly selected
- ✅ **Balanced splits**: ~70/15/15 train/val/test
- ✅ **Quality filtered**: Only slices with ≥50 tumor pixels
- ✅ **Efficient format**: Compressed .npz (fast loading)

### Storage Requirements:
- **Original NIfTI** (25 patients): ~2.1 GB
- **Converted NPZ** (528 slices): ~150 MB
- **Total project size**: ~200 MB (including scripts, docs)

### Training Estimates (on Amarel GPU):
- **Single epoch**: ~2-3 minutes (batch size 8)
- **Full training** (50 epochs): ~2 hours
- **Ensemble** (5 models): ~10 hours
- **MC Dropout** inference: ~5 minutes (20 samples)

---

## 💡 Key Design Decisions

1. **T1ce only**: Most informative modality for tumors
2. **Slice stride 3**: Reduces redundancy while keeping diversity
3. **Min 50 tumor pixels**: Avoids class imbalance from empty slices
4. **Binary masks**: Simplified from 4-class (simplifies baseline)
5. **25 patients**: Small enough to iterate quickly, large enough to be meaningful

---

## 🔄 How to Regenerate Data (if needed)

If you want different parameters:

```bash
# More patients (slower, more data)
python scripts/prepare_small_brats_subset.py \
    --brats_root "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" \
    --out_dir data/brats_large \
    --n_patients 50

# Fewer slices (faster training)
python scripts/prepare_small_brats_subset.py \
    --brats_root "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" \
    --out_dir data/brats_sparse \
    --n_patients 25 \
    --slice_stride 5

# Different modality
python scripts/prepare_small_brats_subset.py \
    --brats_root "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData" \
    --out_dir data/brats_flair \
    --n_patients 25 \
    --modality flair
```

---

## 📧 Communication with Dr. Moran

### What She Suggested:
1. ✅ Small experiment to get baseline results
2. ✅ Compare: Temperature Scaling, Deep Ensembles, MC Dropout, Conformal Prediction
3. ✅ Possibly add: Sparse autoencoders for interpretability

### What You've Accomplished:
1. ✅ Prepared small dataset (528 slices, 25 patients)
2. ✅ Created all necessary scripts for Amarel
3. ✅ Validated data integrity
4. ✅ Ready to start experiments

### What to Tell Her in Next Meeting:
- "I've prepared a small BraTS subset with 25 patients (528 slices)"
- "Data is validated and ready to upload to Amarel"
- "Created scripts for training on Amarel cluster"
- "Ready to start with baseline + temperature scaling this week"

---

## ✨ Summary

You now have:
- ✅ Clean, validated dataset ready for experiments
- ✅ All scripts needed to work on Amarel
- ✅ Comprehensive documentation
- ✅ Clear path forward for UQ experiments

**You're ready to start running experiments on Amarel! 🚀**

Next immediate action: Upload data and test on Amarel.
