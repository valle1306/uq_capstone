# Repository Organization

```
uq_capstone/
│
├── README.md                              # Main project overview
├── thesis_draft.md                        # Draft thesis (DO NOT COMMIT - local only)
│
├── docs/                                  # Documentation
│   ├── SWAG_OVERFITTING_DISCOVERY.md     # Analysis of SWAG failure
│   ├── SWAG_TWO_STAGE_SUMMARY.md         # Two-stage solution
│   ├── EXPERIMENTAL_PLAN.md              # Instructor questions response
│   ├── INSTRUCTOR_RESPONSE.md            # Comprehensive Q&A
│   └── [other guides...]
│
├── src/                                   # Source code
│   ├── data_utils_classification.py      # Data loading
│   ├── swag.py                           # SWAG implementation
│   │
│   ├── train_classifier_baseline.py      # Baseline training
│   ├── train_classifier_mc_dropout.py    # MC Dropout training
│   ├── train_classifier_ensemble_member.py # Ensemble training
│   │
│   ├── retrain_swag_proper.py            # SWAG with SGD (failed)
│   ├── retrain_swag_conservative.py      # SWAG with conservative SGD
│   ├── retrain_swag_adam.py              # SWAG with Adam (overfit issue)
│   ├── retrain_swag_two_stage.py         # ✅ Two-stage solution (NEW)
│   │
│   ├── evaluate_uq_classification.py     # Comprehensive evaluation
│   ├── conformal_prediction.py           # Conformal methods
│   └── analyze_conformal_calibration.py  # Calibration analysis
│
├── scripts/                               # SLURM batch scripts
│   ├── train_classifier_baseline.sbatch
│   ├── train_classifier_mc_dropout.sbatch
│   ├── train_classifier_ensemble.sbatch
│   │
│   ├── retrain_swag_proper.sbatch
│   ├── retrain_swag_conservative.sbatch
│   ├── retrain_swag_adam.sbatch
│   ├── retrain_swag_two_stage.sbatch     # ✅ Two-stage job (NEW)
│   │
│   ├── evaluate_classification_comprehensive.sbatch
│   ├── run_conformal_prediction.sbatch
│   └── analyze_calibration.sbatch
│
├── runs/                                  # Training outputs (gitignored)
│   └── classification/
│       ├── baseline/                      # 91.67% accuracy ✓
│       ├── mc_dropout/                    # 85.26% accuracy ✓
│       ├── ensemble/                      # 91.67% accuracy, ECE=0.027 ✓
│       ├── swag_sgd/                      # 79.65% (failed)
│       ├── swag_adam/                     # 81-83% (overfit issue)
│       ├── swag_two_stage/                # ⏳ Target: 88-90% (NEW)
│       └── conformal/                     # Conformal results
│
├── data/                                  # Datasets (gitignored)
│   └── chest_xray/
│
├── analysis/                              # Analysis scripts
├── papers/                                # Reference papers
└── presentation/                          # Presentation files

```

## File Status

### Completed ✅
- Baseline, MC Dropout, Deep Ensemble: All trained and evaluated
- Conformal prediction: Implemented for all methods
- SWAG analysis: Root cause identified (overfitting timing)

### In Progress ⏳
- SWAG (Adam): Job 48295659 running (identifies problem)
- Calibration analysis: Job 48295688 running

### New Solution 🆕
- Two-stage SWAG: Ready to submit (fixes problem)

## Documentation Priority

### For Thesis
1. `SWAG_OVERFITTING_DISCOVERY.md` - Core finding
2. `SWAG_TWO_STAGE_SUMMARY.md` - Solution
3. `EXPERIMENTAL_PLAN.md` - Instructor Q&A
4. `thesis_draft.md` - Full paper (local only)

### For Code Review
1. `README.md` - Project overview
2. `src/retrain_swag_two_stage.py` - New implementation
3. `scripts/retrain_swag_two_stage.sbatch` - Job script

## Gitignore Strategy

### Excluded from Git
- `thesis_draft.md` (too large, work in progress)
- `runs/` (training outputs)
- `data/` (datasets)
- `.venv/` (Python environment)
- `__pycache__/` (Python bytecode)

### Included in Git
- All source code (`src/`, `scripts/`)
- All documentation (`docs/`, `*.md` except thesis)
- Configuration files
- Analysis scripts
