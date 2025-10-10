# Medical Image Segmentation with Uncertainty Quantification

This repository implements and evaluates **4 uncertainty quantification (UQ) methods** for medical image segmentation on the BraTS2020 brain tumor dataset.

## 🎯 Project Overview

**Goal**: Compare uncertainty quantification methods for reliable medical image segmentation.

**Methods Implemented**:
1. **Baseline** - Standard U-Net (no uncertainty)
2. **MC Dropout** - Monte Carlo Dropout sampling
3. **Deep Ensemble** - Multiple independent models
4. **SWAG** - Stochastic Weight Averaging-Gaussian

## 📊 Key Results

| Method | Dice Score | ECE | Uncertainty | Rank |
|--------|-----------|-----|-------------|------|
| **Deep Ensemble** | 0.7550 | 0.9589 | 0.0158 | 🥇 1st |
| **SWAG** | 0.7419 | 0.9656 | 0.0026 | 🥈 2nd |
| **MC Dropout** | 0.7403 | 0.9663 | 0.0011 | 🥉 3rd |
| **Baseline** | 0.7401 | 0.9673 | N/A | 4th |

- **SWAG Fix**: Achieved 427% improvement (Dice: 0.14 → 0.74) by fixing unbounded variance bug
- **Evaluation**: 80 test samples from BraTS2020 dataset
- **Platform**: Rutgers Amarel HPC with NVIDIA GPUs

## 📁 Repository Structure

```
uq_capstone/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
│
├── src/                     # Source code
│   ├── data_utils.py       # Data loading and preprocessing
│   ├── model_utils.py      # U-Net architecture
│   ├── uq_methods.py       # UQ method implementations
│   ├── swag.py             # SWAG implementation (FIXED)
│   ├── train_baseline.py   # Train baseline model
│   ├── train_mc_dropout.py # Train MC Dropout
│   ├── train_ensemble_member.py # Train ensemble member
│   ├── train_swag.py       # Train SWAG model
│   ├── evaluate_uq.py      # Original evaluation script
│   └── evaluate_uq_FIXED_v2.py # Fixed evaluation (uses max_var=1.0)
│
├── scripts/                 # SLURM batch scripts for Amarel HPC
│   ├── train_baseline.sbatch
│   ├── train_mc_dropout.sbatch
│   ├── train_ensemble.sbatch
│   ├── train_swag.sbatch
│   ├── evaluate_uq.sbatch
│   ├── run_all_experiments.sh
│   ├── monitor_jobs.sh
│   └── validate_brats_data.py
│
├── analysis/                # UQ analysis scripts (to be created)
│   ├── analyze_uq_metrics.py    # Compute calibration metrics
│   ├── visualize_uq.py          # Generate visualizations
│   └── generate_uq_report.py    # Create comprehensive report
│
├── docs/                    # Documentation
│   ├── START_HERE.md       # Quick start guide
│   ├── QUICK_START_UQ.md   # UQ experiments guide
│   ├── SWAG_FIXED_SUCCESS.md # SWAG fix documentation
│   ├── SYNC_COMPLETE_GUIDE.md # Repository sync guide
│   └── ... (other documentation)
│
├── utils/                   # Utility scripts
│   ├── upload_amarel.ps1   # Upload files to Amarel
│   ├── download_from_amarel.ps1 # Download from Amarel
│   └── test_swag_now.py    # SWAG testing script
│
├── data/                    # Dataset (gitignored)
│   └── brats/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       ├── images/
│       └── masks/
│
├── runs/                    # Training/evaluation outputs (gitignored)
│   ├── baseline/
│   ├── mc_dropout/
│   ├── ensemble/
│   ├── swag/
│   └── evaluation/
│       ├── results.json
│       └── eval_v2_47441209.out
│
├── envs/                    # Conda environment files
│   ├── conda_env.yml       # CPU environment
│   └── conda_env_cu118.yml # GPU environment (CUDA 11.8)
│
└── papers/                  # Reference papers
    └── baseline_for_uncertainty_DL.pdf
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# On Amarel HPC
conda env create -f envs/conda_env_cu118.yml
conda activate uq_capstone
```

### 2. Prepare Data

```bash
# Prepare BraTS dataset
python scripts/prepare_small_brats_subset.py
python scripts/validate_brats_data.py
```

### 3. Train Models

```bash
# Run all experiments
cd scripts
bash run_all_experiments.sh

# Or train individually
sbatch train_baseline.sbatch
sbatch train_mc_dropout.sbatch
sbatch train_ensemble.sbatch
sbatch train_swag.sbatch
```

### 4. Evaluate UQ Methods

```bash
sbatch scripts/evaluate_uq.sbatch
```

### 5. Analyze Results

```bash
# Compute UQ metrics
python analysis/analyze_uq_metrics.py

# Generate visualizations
python analysis/visualize_uq.py

# Create comprehensive report
python analysis/generate_uq_report.py
```

## 🔧 Key Technical Details

### SWAG Fix (Critical)

**Problem**: Original SWAG implementation had unbounded variance causing:
- Variance values up to 226M
- Weight explosion (sampled weights up to 249K)
- Catastrophic predictions (Dice = 0.14, Uncertainty = NaN)

**Solution**: Added `max_var` parameter to cap variance:
```python
# In swag.py
swag_model = SWAG(base_model, max_num_models=20, max_var=1.0)
var = torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp, self.max_var)
```

**Result**: SWAG now works correctly with Dice=0.74, competitive with other methods.

### Dataset

- **Source**: BraTS2020 (Brain Tumor Segmentation Challenge)
- **Training samples**: 320 slices
- **Validation samples**: 40 slices
- **Test samples**: 80 slices
- **Task**: Binary segmentation (tumor vs background)
- **Format**: `.npz` files with preprocessed 2D slices

### Training Configuration

- **Architecture**: U-Net with 4 encoder/decoder blocks
- **Loss**: Dice Loss
- **Optimizer**: Adam (lr=1e-3)
- **Epochs**: 30 for baseline, 20 for SWAG
- **Batch size**: 16
- **Hardware**: NVIDIA A100 GPUs on Amarel HPC

## 📈 Evaluation Metrics

1. **Segmentation Quality**:
   - Dice Score
   - IoU (Intersection over Union)

2. **Calibration Metrics**:
   - ECE (Expected Calibration Error)
   - MCE (Maximum Calibration Error)
   - Brier Score

3. **Uncertainty Quality**:
   - Uncertainty-Error Correlation (Pearson, Spearman)
   - AUROC for error detection
   - Reliability diagrams

## 📚 Documentation

- **[START_HERE.md](docs/START_HERE.md)** - Comprehensive setup guide
- **[QUICK_START_UQ.md](docs/QUICK_START_UQ.md)** - Run UQ experiments
- **[SWAG_FIXED_SUCCESS.md](docs/SWAG_FIXED_SUCCESS.md)** - SWAG debugging journey
- **[UQ_EXPERIMENTS_GUIDE.md](docs/UQ_EXPERIMENTS_GUIDE.md)** - Detailed experiment guide

## 🎓 References

This work is based on:
- **SWAG**: Maddox et al. "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (NeurIPS 2019)
- **MC Dropout**: Gal & Ghahramani "Dropout as a Bayesian Approximation" (ICML 2016)
- **Deep Ensembles**: Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty" (NIPS 2017)
- **BraTS Dataset**: Menze et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)" (IEEE TMI 2015)

## 🤝 Contributing

This is a research project for uncertainty quantification in medical image segmentation. For questions or issues, please open a GitHub issue.

## 📄 License

This project is for academic research purposes.

## 👥 Authors

- Phan Nguyen Huong Le; Advisor: Gemma Moran
