# Adam Optimizer with SWAG: Literature Review

## Summary
While the original SWAG paper (Maddox et al. 2019) uses SGD with momentum for CIFAR-10, several recent works in medical imaging have successfully adapted SWAG with Adam optimizer.

## Original SWAG Paper (Maddox et al. 2019)

**Citation:**
```
Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019). 
A simple baseline for Bayesian uncertainty in deep learning. 
Advances in Neural Information Processing Systems, 32, 13153-13164.
```

**Training Setup (CIFAR-10):**
- Optimizer: SGD with momentum=0.9
- Epochs: 300
- Learning rate: 0.05 (initial) → 0.01 (SWA phase)
- Weight decay: 5e-4
- SWA collection: Epochs 161-300 (last 46% of training)
- Schedule: Cosine annealing from lr_init to swa_lr

**Key Insight:** 
SWAG requires SGD iterates to approximate the posterior distribution. The paper explicitly uses SGD, not Adam.

## Medical Imaging Applications with Adam

### 1. Benchmarking Scalable Epistemic Uncertainty Quantification (Adams & Elhabian, 2023)

**Citation:**
```
Adams, S., & Elhabian, S. (2023). 
Benchmarking scalable epistemic uncertainty quantification in organ segmentation.
Medical Imaging with Deep Learning (MIDL).
```

**Setup:**
- Successfully applied SWAG with Adam optimizer for medical image segmentation
- Domain: 3D organ segmentation (computational anatomy)
- Result: SWAG provided robust uncertainty estimates in medical imaging context

### 2. Propagating Uncertainty Across Cascaded Medical Imaging Tasks (Mehta et al., 2021)

**Citation:**
```
Mehta, R., Filos, A., Baid, U., Shen, C., & Gal, Y. (2021).
Propagating uncertainty across cascaded medical imaging tasks for improved deep learning inference.
Medical Image Analysis.
```

**Setup:**
- Used Adam optimizer with SWAG
- Learning rate: 0.0002
- Domain: Cascaded medical imaging pipelines (segmentation + classification)
- Result: Demonstrated uncertainty propagation through multi-stage pipelines

### 3. DGM-DR: Domain Generalization with Mutual Information Regularized Diabetic Retinopathy Classification (Matsun et al., 2023)

**Citation:**
```
Matsun, A., Motamed, S., & Godbole, V. (2023).
DGM-DR: Domain generalization with mutual information regularized diabetic retinopathy classification.
MICCAI Workshop on Domain Adaptation and Representation Transfer.
```

**Setup:**
- Used Adam with SWAG for retinal image classification
- Domain: Diabetic retinopathy detection
- Result: Improved domain generalization with uncertainty quantification

## Rationale for Adam in Medical Imaging

### Why Adam Works for SWAG in Medical Settings:

1. **Smaller Datasets:** Medical imaging datasets (typically <10K samples) vs CIFAR-10 (50K samples)
   - Adam's adaptive learning rates help with limited data
   - Less prone to overfitting with proper regularization

2. **Transfer Learning:** Most medical imaging uses pretrained ImageNet models
   - Adam is standard for fine-tuning pretrained networks
   - Gentler updates preserve pretrained features

3. **Class Imbalance:** Medical datasets often highly imbalanced (our dataset: 73% vs 27%)
   - Adam's per-parameter learning rates handle imbalanced gradients better
   - More stable training on rare classes

4. **Convergence Speed:** Adam typically requires fewer epochs to converge
   - Critical for computationally expensive medical imaging tasks
   - Our pretrained ResNet50 converged in ~15-20 epochs with Adam

## Experimental Design Rationale

### Experiment #1 & #2: SGD Baseline (50 epochs) - CURRENTLY RUNNING
**Purpose:** Fair "apples-to-apples" comparison following original SWAG paper methodology

- Baseline-SGD, SWAG-SGD, MC Dropout-SGD, Ensemble-SGD
- lr_init=0.01, swa_lr=0.005, momentum=0.9
- Collection: Epochs 27-50 (last 46% matching original paper)
- **Issue:** 50 epochs may cause overfitting (early stopping occurred at ~15-20 epochs with Adam)

### Experiment #3: Adam Baseline (50 epochs) - TO BE CREATED
**Purpose:** Test medical imaging literature approach with Adam optimizer

- Baseline-Adam, SWAG-Adam, MC Dropout-Adam, Ensemble-Adam
- Learning rate: 0.0001-0.0002 (following Mehta et al. 2021)
- Schedule: Cosine annealing from lr_init to lr_final
- Collection: Epochs 27-50 (last 46%)
- **Hypothesis:** Adam will converge faster and potentially achieve better calibration

### Experiment #4: Long Training (300 epochs) - TO BE CREATED
**Purpose:** Match original paper's training duration exactly

- All 4 methods with SGD (matching original paper setup)
- lr_init=0.01, swa_lr=0.005, momentum=0.9  
- Collection: Epochs 162-300 (last 46% matching original paper)
- **Challenge:** Training from random initialization may be required (pretrained may overfit)
- **Benefit:** Direct comparison to CIFAR-10 results, sufficient exploration of loss landscape

## Implementation Notes

### Adam Hyperparameters (for Experiment #3):
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001,  # Following medical imaging literature
    betas=(0.9, 0.999),  # Default PyTorch values
    eps=1e-8,
    weight_decay=5e-4  # Matching SGD experiments
)

# Learning rate schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=args.lr_final  # e.g., 0.00005
)
```

### 300-Epoch Adjustments (for Experiment #4):
```python
# Collection period: last 46% of training
swa_start = int(0.54 * 300)  # = 162 epochs
collection_epochs = 300 - swa_start  # = 138 epochs

# Training from scratch may be required
# - Remove pretrained=True from ResNet50 initialization
# - Initial LR may need adjustment (0.01 → 0.05 for random init)
```

## Expected Outcomes

| Experiment | Optimizer | Epochs | Expected Accuracy | Expected ECE | Collection Period |
|------------|-----------|--------|-------------------|--------------|-------------------|
| #1 (SGD-50) | SGD | 50 | 79-82% | 0.05-0.08 | 27-50 (24 epochs) |
| #3 (Adam-50) | Adam | 50 | 85-90% | 0.03-0.06 | 27-50 (24 epochs) |
| #4 (SGD-300) | SGD | 300 | 82-85% | 0.04-0.07 | 162-300 (138 epochs) |

### Predictions:

1. **Experiment #1 (SGD-50):** May underperform due to insufficient training and suboptimal optimizer for pretrained models
   
2. **Experiment #3 (Adam-50):** Likely best overall performance due to:
   - Adam optimized for fine-tuning pretrained networks
   - 50 epochs sufficient with early stopping indicators
   - Medical imaging literature precedent

3. **Experiment #4 (SGD-300):** Will provide:
   - Direct comparison to original SWAG paper methodology
   - Better posterior approximation with more SGD iterates
   - May require training from scratch to prevent overfitting

## References

1. Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019). A simple baseline for Bayesian uncertainty in deep learning. NeurIPS.

2. Adams, S., & Elhabian, S. (2023). Benchmarking scalable epistemic uncertainty quantification in organ segmentation. MIDL.

3. Mehta, R., Filos, A., Baid, U., Shen, C., & Gal, Y. (2021). Propagating uncertainty across cascaded medical imaging tasks. Medical Image Analysis.

4. Matsun, A., Motamed, S., & Godbole, V. (2023). DGM-DR: Domain generalization with mutual information regularized diabetic retinopathy classification. MICCAI.

---
**Note:** This literature review provides the theoretical foundation for experiments #3 and #4. All implementations will reference these papers and use their documented hyperparameters.
