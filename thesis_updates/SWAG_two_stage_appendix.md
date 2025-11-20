# SWAG Two-Stage Update (Planned)

Since initial SWAG runs with both SGD and Adam showed rapid overfitting before snapshot collection, we include a targeted plan and placeholder for results from the proposed two-stage SWAG schedule.

Planned two-stage schedule:

- Stage 1 (epochs 1-20): cyclic learning rate (example: 0.0001 ↔ 0.001) to force exploration and escape narrow overfit minima
- Stage 2 (epochs 21-50): fixed low learning rate (0.0001) and collect snapshots every epoch (epochs 21-50)

Artifacts and scripts (added to repository):

- `src/retrain_swag_two_stage.py` — two-stage trainer skeleton that saves snapshot checkpoints for postprocessing with `swa_gaussian`
- `src/run_conformal_all.py` — conformal runner computing coverage and average set size for each method
- `scripts/download_results.ps1` — utility to fetch SWAG Adam and conformal outputs from Amarel

When two-stage results are available, we will fill in Table 1 (SWAG two-stage row) and Table 3 (conformal results per-method). The expected outcome is 88–90% test accuracy and conformal coverage ≥ 90% when applied to the improved base model.
