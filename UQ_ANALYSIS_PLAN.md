# Uncertainty Quantification Analysis & Visualization Plan

## Overview
Comprehensive evaluation of uncertainty quality across all UQ methods (Baseline, MC Dropout, Deep Ensemble, SWAG) with detailed visualizations and metrics.

## 🎯 Goals

1. **Evaluate UQ Quality:** Do the uncertainty estimates actually correlate with prediction errors?
2. **Compare Methods:** Which UQ method provides the best uncertainty estimates?
3. **Visualize Results:** Create publication-quality visualizations
4. **Document Findings:** Comprehensive report with recommendations

## 📊 Metrics to Compute

### 1. Segmentation Performance
- ✅ **Dice Score** - Already computed
- ✅ **ECE (Expected Calibration Error)** - Already computed  
- **IoU (Intersection over Union)**
- **Precision & Recall**
- **Hausdorff Distance** (boundary quality)

### 2. Uncertainty Quality Metrics

#### A. Calibration Metrics
- **ECE (Expected Calibration Error)** - Already have
- **MCE (Maximum Calibration Error)**
- **Brier Score** - Measures sharpness and calibration
- **Negative Log-Likelihood**

#### B. Sharpness Metrics
- **Average Uncertainty** - Already have
- **Uncertainty Distribution** (mean, std, percentiles)
- **Confidence vs Coverage** curves

#### C. Correlation Metrics
- **Uncertainty-Error Correlation:**
  - Pearson correlation between uncertainty and pixel-wise error
  - Spearman rank correlation
- **AUROC for Error Detection:**
  - Treat high uncertainty as "error detector"
  - Compute ROC curve and AUC
- **AUPRC (Area Under Precision-Recall Curve)**

#### D. Reliability Metrics
- **Reliability Diagrams** - Predicted confidence vs actual accuracy
- **Adaptive Calibration Error** (ACE)

### 3. Method-Specific Analysis
- **MC Dropout:** Effect of number of samples (10, 20, 30)
- **Deep Ensemble:** Effect of ensemble size (3, 5, 10)
- **SWAG:** Effect of sampling scale (0.1, 0.5, 1.0) and n_samples

## 📈 Visualizations to Create

### 1. Uncertainty Maps (Per-Sample Visualizations)
For representative test samples, create 4-panel figures:
```
[Original Image] [Ground Truth Mask] [Prediction] [Uncertainty Map]
```

Show for:
- Best prediction (high confidence, correct)
- Worst prediction (high confidence, incorrect)  
- Uncertain but correct (high uncertainty, correct)
- Uncertain and incorrect (high uncertainty, incorrect)

Create for each method (Baseline has no uncertainty, so 3-panel).

### 2. Calibration Plots
**Reliability Diagrams:**
- X-axis: Predicted confidence (binned)
- Y-axis: Actual accuracy
- Diagonal = perfect calibration
- One plot per method, or combined plot

**ECE Bar Chart:**
- Compare ECE across all methods
- Lower is better

### 3. Uncertainty-Error Correlation
**Scatter Plots:**
- X-axis: Uncertainty (per pixel or per sample)
- Y-axis: Prediction error (|prediction - ground truth|)
- Color by Dice score
- Add correlation coefficient

**Box Plots:**
- Group pixels into "Correct" vs "Incorrect"
- Show uncertainty distribution for each group
- Significant difference = good uncertainty

### 4. ROC Curves for Error Detection
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- Treat uncertainty > threshold as "predicted error"
- Higher AUC = better uncertainty quality

### 5. Method Comparison
**Radar/Spider Charts:**
- Axes: Dice, ECE, Uncertainty Correlation, AUROC, Runtime
- One polygon per method
- Shows strengths/weaknesses

**Bar Charts:**
- Dice scores
- Calibration metrics
- Inference time

### 6. Uncertainty Distributions
**Histograms:**
- Distribution of uncertainty values per method
- Separate for correct vs incorrect predictions

**Violin Plots:**
- Compare uncertainty distributions across methods

### 7. Coverage vs Confidence Curves
- X-axis: Confidence threshold
- Y-axis: Coverage (% of data retained) and Accuracy
- Shows accuracy-coverage tradeoff

## 🔧 Implementation Plan

### Script 1: `analyze_uq_metrics.py`
**Purpose:** Compute all uncertainty quality metrics

**Inputs:**
- Model checkpoints
- Test dataset
- Predictions + uncertainties from each method

**Outputs:**
- `uq_metrics.json` - All computed metrics
- `per_sample_metrics.csv` - Sample-level analysis

**Key Functions:**
```python
def compute_calibration_metrics(probs, targets)
def compute_uncertainty_error_correlation(uncertainty, errors)
def compute_auroc_for_error_detection(uncertainty, errors)
def compute_reliability_curve(probs, targets, n_bins=10)
```

### Script 2: `visualize_uq.py`
**Purpose:** Generate all visualizations

**Inputs:**
- `uq_metrics.json`
- `per_sample_metrics.csv`
- Raw predictions and uncertainty maps

**Outputs:**
- `figures/uncertainty_maps/` - Per-sample visualizations
- `figures/calibration_plots/` - Calibration curves
- `figures/correlation_plots/` - Uncertainty-error analysis
- `figures/comparison_charts/` - Method comparisons
- `figures/summary_figure.png` - Main publication figure

**Key Functions:**
```python
def plot_uncertainty_map(image, mask, pred, uncertainty)
def plot_calibration_curve(method_name, probs, targets)
def plot_uncertainty_error_scatter(uncertainty, errors, dice_scores)
def plot_method_comparison_radar(metrics_dict)
```

### Script 3: `generate_uq_report.py`
**Purpose:** Create comprehensive markdown report

**Inputs:**
- `uq_metrics.json`
- All generated figures

**Outputs:**
- `UQ_ANALYSIS_REPORT.md` - Comprehensive report
- Includes all metrics, figures, and recommendations

## 📝 Report Structure

```markdown
# Uncertainty Quantification Analysis Report

## Executive Summary
- Key findings
- Best method for different use cases
- Recommendations

## 1. Introduction
- Dataset description
- Methods evaluated
- Metrics used

## 2. Segmentation Performance
- Dice scores comparison
- Precision/Recall analysis
- Qualitative examples

## 3. Uncertainty Quality Analysis
### 3.1 Calibration
- ECE/MCE results
- Reliability diagrams
- Findings

### 3.2 Sharpness
- Uncertainty distributions
- Confidence analysis

### 3.3 Correlation with Errors
- Correlation coefficients
- ROC/AUC analysis
- Scatter plots

## 4. Method-by-Method Analysis
### 4.1 Baseline (No UQ)
### 4.2 MC Dropout
### 4.3 Deep Ensemble
### 4.4 SWAG

## 5. Computational Cost Analysis
- Training time
- Inference time
- Memory requirements

## 6. Use Case Recommendations
- When to use MC Dropout
- When to use Deep Ensemble
- When to use SWAG
- Clinical deployment considerations

## 7. Limitations & Future Work

## 8. Conclusion

## Appendix
- All metrics tables
- Additional visualizations
```

## 🚀 Execution Steps

### Step 1: Fix SWAG and Re-evaluate ✅ (In Progress)
```bash
# After SWAG fixed:
sbatch scripts/evaluate_uq_v2.sbatch
```

### Step 2: Compute UQ Quality Metrics
```bash
python src/analyze_uq_metrics.py \
  --results runs/evaluation/results.json \
  --data_dir data/brats \
  --save_dir runs/uq_analysis
```

###  Step 3: Generate Visualizations
```bash
python src/visualize_uq.py \
  --metrics runs/uq_analysis/uq_metrics.json \
  --save_dir runs/uq_analysis/figures
```

### Step 4: Generate Report
```bash
python src/generate_uq_report.py \
  --metrics runs/uq_analysis/uq_metrics.json \
  --figures runs/uq_analysis/figures \
  --output UQ_ANALYSIS_REPORT.md
```

## 📦 Dependencies Needed
```python
# Already have:
- torch, numpy, matplotlib

# May need to add:
- seaborn (for nice plots)
- scipy (for statistical tests)
- sklearn (for metrics like ROC/AUC)
- pandas (for data analysis)
- plotly (for interactive plots - optional)
```

## ⏱️ Estimated Timeline

| Task | Time | Status |
|------|------|--------|
| Fix SWAG & re-evaluate | 1-2 hours | 🔄 In Progress |
| Implement metric computations | 2-3 hours | ⏳ Pending |
| Create visualizations | 2-3 hours | ⏳ Pending |
| Generate report | 1-2 hours | ⏳ Pending |
| **Total** | **6-10 hours** | |

## 🎓 Expected Insights

### Hypothesis 1: Deep Ensemble > MC Dropout > Baseline
- **Expectation:** Ensemble provides best uncertainty
- **Metric:** Highest AUROC for error detection

### Hypothesis 2: MC Dropout is Underconfident
- **Expectation:** Low average uncertainty (0.0010)
- **Metric:** High ECE, poor calibration

### Hypothesis 3: Ensemble Best for Safety-Critical Apps
- **Expectation:** Best uncertainty-error correlation
- **Metric:** Highest Spearman correlation

### Hypothesis 4: SWAG Competitive with Ensemble
- **Expectation:** After fixing, similar performance to Ensemble
- **Metric:** Comparable Dice and AUROC

## 📚 References for Metrics

1. **ECE/MCE:** Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
2. **Uncertainty Quality:** Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation" (NeurIPS 2017)
3. **Medical Imaging UQ:** Jungo et al., "Analyzing the Quality and Challenges of Uncertainty Estimations for Brain Tumor Segmentation" (2020)
4. **Reliability Diagrams:** DeGroot & Fienberg, "The Comparison and Evaluation of Forecasters" (1983)

---

**Ready to Execute:** Once SWAG testing completes and optimal `max_var` is identified, we can proceed with full UQ analysis pipeline.
