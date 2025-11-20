# Extended Work: Tier 1 Limitations Analysis

This directory contains analysis scripts for addressing publication-critical limitations identified in Section 6.2 of the thesis.

## Overview

**Tier 1** limitations are high-impact analyses that require **minimal compute** (no retraining needed). These analyses use existing trained models and results to provide deeper insights into model behavior.

## Implemented Analyses

### 1. Class-Conditional Calibration (`class_conditional_calibration.py`)

**Purpose**: Detect calibration bias across classes (Normal vs Pneumonia).

**Research Question**: Does the model exhibit different uncertainty quality for majority vs minority class?

**Metrics**:
- Per-class ECE (Expected Calibration Error)
- Calibration gap = max(ECE_i - ECE_j)
- Per-class conformal set sizes

**Clinical Relevance**: If ECE(Normal) >> ECE(Pneumonia), model is underconfident on healthy patients → excessive false alarms. Reverse pattern signals dangerous overconfidence on pneumonia.

**Usage**:
```bash
python class_conditional_calibration.py \
    --results_dir ../results/baseline_sgd \
    --calibration_file ../results/baseline_sgd/conformal_calibration.json \
    --output_dir outputs/class_conditional
```

**Expected Runtime**: 2 hours (analysis only)

**References**:
- Angelopoulos et al. (2021): Class-conditional conformal prediction
- Romano et al. (2020): Stratified conformal prediction

---

### 2. Selective Prediction (`selective_prediction.py`)

**Purpose**: Analyze accuracy-coverage trade-off for clinical workflows.

**Research Question**: At what coverage level does each method achieve target accuracy (e.g., 95%)?

**Metrics**:
- Accuracy-coverage curves
- Area Under Risk-Coverage Curve (AURC) - lower is better
- Misdiagnoses prevented at various coverage levels

**Clinical Relevance**: Emergency departments may reject 20% most uncertain cases for radiologist review. Which method provides best accuracy on automated 80%?

**Usage**:
```bash
python selective_prediction.py \
    --results_dirs ../results/baseline_sgd ../results/ensemble_sgd \
    --method_names "Baseline" "Ensemble" \
    --output_dir outputs/selective_prediction
```

**Expected Runtime**: 2 hours (analysis only)

**References**:
- Geifman & El-Yaniv (2017): Selective prediction
- Cordella et al. (2023): Risk-coverage curves for medical AI

---

### 3. Failure Mode Taxonomy (Coming Soon)

**Purpose**: Qualitative analysis of high-disagreement cases.

**Approach**: Manually review 54 cases (8.7% disagreement rate) where ensemble members disagree.

**Categories**:
1. **Subtle infiltrates** (30%): Faint opacities near diaphragm
2. **Atypical presentations** (25%): Viral pneumonia, ground-glass opacities
3. **Technical artifacts** (20%): Pacemakers, ECG leads occluding lung
4. **Borderline cases** (25%): Early-stage pneumonia requiring follow-up

**Expected Runtime**: 4 hours (requires radiologist collaboration)

---

### 4. Temperature Scaling + Conformal Prediction (Coming Soon)

**Purpose**: Two-stage calibration to reduce conformal set sizes.

**Experiment**:
1. Apply temperature scaling to raw model outputs (using 522 calibration samples)
2. Apply conformal prediction on scaled outputs (using separate 522 samples)
3. Compare set sizes before/after temperature scaling

**Research Question**: Does post-hoc calibration reduce ambiguous predictions while maintaining coverage?

**Expected Runtime**: 1 day (minimal retraining - only learns temperature parameter)

**References**:
- Guo et al. (2017): Temperature scaling
- Kumar et al. (2019): Combining post-hoc calibration with conformal prediction

---

## Running All Analyses

After training completes and results are available:

```bash
# 1. Class-conditional calibration for all methods
for method in baseline_sgd swag_sgd dropout_sgd ensemble_sgd; do
    python class_conditional_calibration.py \
        --results_dir ../results/$method \
        --calibration_file ../results/$method/conformal_calibration.json \
        --output_dir outputs/class_conditional/$method
done

# 2. Selective prediction comparison
python selective_prediction.py \
    --results_dirs ../results/baseline_sgd ../results/swag_sgd ../results/dropout_sgd ../results/ensemble_sgd \
    --method_names "Baseline" "SWAG" "MC Dropout" "Ensemble" \
    --output_dir outputs/selective_prediction/sgd_comparison

# 3. Failure mode analysis (manual review)
# Review high-disagreement cases in outputs/failure_modes/

# 4. Temperature scaling experiment
python temperature_scaling_conformal.py \
    --results_dir ../results/baseline_sgd \
    --output_dir outputs/temperature_scaling
```

## Expected Outputs

Each analysis generates:
- **JSON summary**: Quantitative metrics for thesis tables
- **Visualizations**: Publication-quality figures (PNG, 300 DPI)
- **Console output**: Detailed results with clinical interpretation

## Directory Structure

```
extended_work/
├── tier1_limitations/
│   ├── README.md (this file)
│   ├── class_conditional_calibration.py
│   ├── selective_prediction.py
│   ├── failure_mode_taxonomy.py (TODO)
│   ├── temperature_scaling_conformal.py (TODO)
│   └── outputs/
│       ├── class_conditional/
│       ├── selective_prediction/
│       ├── failure_modes/
│       └── temperature_scaling/
└── tier2_limitations/ (future work)
```

## Timeline

**Week 1** (Current): 
- ✅ Class-conditional calibration implemented
- ✅ Selective prediction implemented
- ⏳ Wait for SGD 300-epoch experiments to complete

**Week 2** (After job completion):
- Run class-conditional and selective prediction on all results
- Implement failure mode taxonomy
- Implement temperature scaling experiment
- Update thesis Section 6.2 with findings

**Week 3** (Optional - Tier 2):
- Subgroup fairness (if metadata available)
- Computational cost-benefit analysis

## Integration with Thesis

Results from these analyses will populate:
- **Table X**: Class-conditional calibration (Section 6.2.2)
- **Figure X**: Accuracy-coverage curves (Section 6.2.5)
- **Figure Y**: Failure mode examples (Section 6.2.9)
- **Table Y**: Temperature scaling improvements (Section 6.2.6)

## Notes

- All scripts use existing `src/` codebase utilities
- No new training required - analyses run on saved results
- Results are cached to avoid recomputation
- Compatible with both SGD and Adam experiments

## Contact

For questions or collaborations on extending these analyses, see main README.
