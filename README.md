# Medical Image Analysis with Uncertainty Quantification

This repository implements and evaluates **uncertainty quantification (UQ) methods** for medical imaging tasks, including both **segmentation** and **classification**, with a focus on the new **Conformal Risk Control** method.

## 🆕 New: Medical Image Classification + Conformal Risk Control

We've extended the project to include **medical image classification** with **Conformal Risk Control**, a state-of-the-art uncertainty quantification method that provides provable risk guarantees.

**Quick Start:** See [`docs/CLASSIFICATION_QUICK_START.md`](docs/CLASSIFICATION_QUICK_START.md)

## 🎯 Project Overview

This project explores uncertainty quantification in medical imaging through two complementary tasks:

### 1. Segmentation (Completed ✅)
**Goal**: Pixel-wise brain tumor segmentation with uncertainty estimates
**Dataset**: BraTS2020
**Methods**: Baseline, MC Dropout, Deep Ensemble, SWAG

### 2. Classification (New 🆕)
**Goal**: Image-level medical diagnosis with risk-controlled predictions
**Datasets**: Chest X-Ray Pneumonia, OCT Retinal, Brain Tumor MRI
**Methods**: Baseline, MC Dropout, Deep Ensemble, **Conformal Risk Control**

## 🔬 Uncertainty Quantification Methods

### Segmentation Methods
### Segmentation Methods

1. **Baseline** - Standard U-Net (no uncertainty)
2. **MC Dropout** - Monte Carlo Dropout sampling
3. **Deep Ensemble** - Multiple independent models
4. **SWAG** - Stochastic Weight Averaging-Gaussian

### Classification Methods 

1. **Baseline** - Standard ResNet-18 classifier
2. **MC Dropout** - Monte Carlo Dropout for uncertainty
3. **Deep Ensemble** - Multiple independent ResNet models
4. **Conformal Risk Control** ⭐ - Distribution-free risk control with provable guarantees

#### What is Conformal Risk Control?

Unlike standard conformal prediction (which only guarantees coverage), **Conformal Risk Control** allows you to control *any* risk metric:

- **False Negative Rate:** "Miss disease ≤ 5% of time"
- **Precision:** "False alarms ≤ 10%"
- **Set Size:** "Prediction set ≤ 2 labels on average"
- **Custom Metrics:** Define your own risk functional

**Key Advantage:** Provable guarantees with finite-sample correction, making it ideal for safety-critical medical applications.

**Paper:** Angelopoulos et al. "Conformal Risk Control" (2022) - See `papers/Conformal Risk Control.pdf`

## 📊 Key Results

### Segmentation Results (BraTS2020)

| Method | Dice Score | ECE | Uncertainty | Rank |
|--------|-----------|-----|-------------|------|
| **Deep Ensemble** | 0.7550 | 0.9589 | 0.0158 | 🥇 1st |
| **SWAG** | 0.7419 | 0.9656 | 0.0026 | 🥈 2nd |
| **MC Dropout** | 0.7403 | 0.9663 | 0.0011 | 🥉 3rd |
| **Baseline** | 0.7401 | 0.9673 | N/A | 4th |

- **Evaluation**: 80 test samples from BraTS2020 dataset
- **Platform**: Rutgers Amarel HPC with NVIDIA GPUs
- **Key Finding**: Deep Ensemble achieved best performance

### Classification Results (Expected)

Results will be available after running experiments. Expected accuracy on Chest X-Ray Pneumonia:

| Method | Expected Accuracy | Notes |
|--------|------------------|-------|
| **Baseline** | ~90-95% | Standard ResNet-18 |
| **MC Dropout** | ~90-95% | Similar accuracy + uncertainty |
| **Deep Ensemble** | ~92-96% | Typically 1-3% improvement |
| **CRC (FNR α=0.05)** | N/A | Guarantees FNR ≤ 5% |
| **CRC (Size α=2.0)** | N/A | Avg prediction set ≤ 2 |

## 📁 Repository Structure

```
uq_capstone/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
│
├── src/                     # Source code
│   # Segmentation (original)
│   ├── data_utils.py       # Data loading and preprocessing
│   ├── model_utils.py      # U-Net architecture
│   ├── uq_methods.py       # UQ method implementations
│   ├── swag.py             # SWAG implementation (FIXED)
│   ├── train_baseline.py   # Train baseline model
│   ├── train_mc_dropout.py # Train MC Dropout
│   ├── train_ensemble_member.py # Train ensemble member
│   ├── train_swag.py       # Train SWAG model
│   ├── evaluate_uq.py      # Original evaluation script
│   └── evaluate_uq_FIXED_v2.py # Fixed evaluation
│   
│   # Classification (NEW 🆕)
│   ├── data_utils_classification.py     # Medical image datasets
│   ├── conformal_risk_control.py        # CRC implementation
│   ├── train_classifier_baseline.py     # Train classifier
│   ├── train_classifier_mc_dropout.py   # Train with dropout
│   ├── train_classifier_ensemble_member.py # Train ensemble
│   └── evaluate_uq_classification.py    # Comprehensive evaluation
│
├── scripts/                 # SLURM batch scripts for Amarel HPC
│   # Segmentation
│   ├── train_baseline.sbatch
│   ├── train_mc_dropout.sbatch
│   ├── train_ensemble.sbatch
│   ├── train_swag.sbatch
│   ├── evaluate_uq.sbatch
│   └── run_all_experiments.sh
│   
│   # Classification (NEW 🆕)
│   ├── train_classifier_baseline.sbatch
│   ├── train_classifier_mc_dropout.sbatch
│   ├── train_classifier_ensemble.sbatch
│   ├── evaluate_classification.sbatch
│   └── run_all_classification_experiments.sh
│
├── analysis/                # UQ analysis scripts
│   ├── analyze_uq_metrics.py    # Compute calibration metrics
│   ├── visualize_uq.py          # Generate visualizations
│   └── generate_uq_report.py    # Create comprehensive report
│
├── docs/                    # Documentation
│   ├── START_HERE.md       # Original quick start guide
│   ├── CLASSIFICATION_QUICK_START.md  # 🆕 Classification quick start
│   ├── CLASSIFICATION_SETUP_GUIDE.md  # 🆕 Detailed setup guide
│   └── ... (other documentation)
│
├── papers/                  # Reference papers
│   ├── baseline_for_uncertainty_DL.pdf
│   └── Conformal Risk Control.pdf  # 🆕 CRC paper
│
├── data/                    # Datasets (gitignored)
│   ├── brats/              # BraTS segmentation data
│   ├── chest_xray/         # 🆕 Chest X-Ray classification
│   ├── oct_retinal/        # 🆕 OCT Retinal images
│   └── brain_tumor/        # 🆕 Brain Tumor MRI
│
└── runs/                    # Training/evaluation outputs (gitignored)
    ├── baseline/           # Segmentation runs
    ├── mc_dropout/
    ├── ensemble/
    ├── swag/
    ├── evaluation/
    └── classification/     # 🆕 Classification runs
        ├── baseline/
        ├── mc_dropout/
        ├── ensemble/
        └── evaluation/
```

## 🚀 Quick Start

### For Segmentation (Original)

See **[START_HERE.md](docs/START_HERE.md)** for the complete segmentation setup.

### For Classification (NEW 🆕)

**Fast Track:** Follow **[CLASSIFICATION_QUICK_START.md](docs/CLASSIFICATION_QUICK_START.md)**

```bash
# 1. Upload code to Amarel (see quick start guide)

# 2. Download dataset (Chest X-Ray Pneumonia recommended)
cd /scratch/$USER/uq_capstone/data
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d chest_xray/

# 3. Run all experiments
cd /scratch/$USER/uq_capstone
bash scripts/run_all_classification_experiments.sh

# 4. Monitor jobs
squeue -u $USER
tail -f runs/classification/*/train_*.out

# 5. Get results (after ~30-40 hours)
cat runs/classification/evaluation/all_results.json
```

### Datasets Available

| Dataset | Classes | Size | Medical Task | Recommended |
|---------|---------|------|--------------|-------------|
| **Chest X-Ray** | 2 | ~5,863 | Pneumonia detection | ⭐ Yes (start here) |
| **OCT Retinal** | 4 | ~84,495 | Retinopathy screening | Multi-class |
| **Brain Tumor** | 4 | ~7,023 | Tumor classification | Continuity with BraTS |

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

### Quick Start Guides
- **[CLASSIFICATION_QUICK_START.md](docs/CLASSIFICATION_QUICK_START.md)** 🆕 - Fast track for classification experiments
- **[START_HERE.md](docs/START_HERE.md)** - Original segmentation setup guide

### Detailed Guides
- **[CLASSIFICATION_SETUP_GUIDE.md](docs/CLASSIFICATION_SETUP_GUIDE.md)** 🆕 - Comprehensive classification documentation
- **[QUICK_START_UQ.md](docs/QUICK_START_UQ.md)** - UQ segmentation experiments
- **[SWAG_FIXED_SUCCESS.md](docs/SWAG_FIXED_SUCCESS.md)** - SWAG debugging journey
- **[UQ_EXPERIMENTS_GUIDE.md](docs/UQ_EXPERIMENTS_GUIDE.md)** - Detailed experiment guide

### Papers
- **[Conformal Risk Control.pdf](papers/Conformal%20Risk%20Control.pdf)** 🆕 - Angelopoulos et al. (2022)
- **baseline_for_uncertainty_DL.pdf** - General UQ reference

## 🎓 References

### Papers

**Conformal Risk Control (NEW):**
- Angelopoulos et al. "Conformal Risk Control" (2022)
- [Paper](https://arxiv.org/abs/2208.02814) | Local: `papers/Conformal Risk Control.pdf`

**Uncertainty Quantification Methods:**
- **SWAG**: Maddox et al. "A Simple Baseline for Bayesian Uncertainty in Deep Learning" (NeurIPS 2019)
- **MC Dropout**: Gal & Ghahramani "Dropout as a Bayesian Approximation" (ICML 2016)
- **Deep Ensembles**: Lakshminarayanan et al. "Simple and Scalable Predictive Uncertainty" (NIPS 2017)

**Datasets:**
- **BraTS**: Menze et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)" (IEEE TMI 2015)
- **Chest X-Ray Pneumonia**: Kermany et al. "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification" (Cell 2018)

### Medical Datasets

- **Chest X-Ray Pneumonia**: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **OCT Retinal Images**: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- **Brain Tumor MRI**: [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## 🤝 Contributing

This is a research project for uncertainty quantification in medical image segmentation. For questions or issues, please open a GitHub issue.

## 📄 License

This project is for academic research purposes.

## 👥 Authors

- Phan Nguyen Huong Le; Advisor: Gemma Moran
