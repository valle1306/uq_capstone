#!/bin/bash

# Master script to run all classification experiments
# Runs training for all UQ methods, then evaluation

echo "========================================================================"
echo "Medical Image Classification with Uncertainty Quantification"
echo "Running All Experiments"
echo "========================================================================"
echo "Start time: $(date)"
echo ""

# Change to project directory
cd /scratch/$USER/uq_capstone || exit 1

# 1. Train Baseline
echo "Step 1/4: Training Baseline Model"
echo "------------------------------------------------------------------------"
baseline_job=$(sbatch --parsable scripts/train_classifier_baseline.sbatch)
echo "✓ Submitted baseline training (Job ID: $baseline_job)"
echo ""

# 2. Train MC Dropout
echo "Step 2/4: Training MC Dropout Model"
echo "------------------------------------------------------------------------"
mc_dropout_job=$(sbatch --parsable scripts/train_classifier_mc_dropout.sbatch)
echo "✓ Submitted MC Dropout training (Job ID: $mc_dropout_job)"
echo ""

# 3. Train Ensemble (all members)
echo "Step 3/4: Training Deep Ensemble (5 members)"
echo "------------------------------------------------------------------------"
ensemble_job=$(sbatch --parsable scripts/train_classifier_ensemble.sbatch)
echo "✓ Submitted ensemble training (Job ID: $ensemble_job)"
echo ""

# 4. Run evaluation (depends on all training jobs)
echo "Step 4/4: Comprehensive Evaluation"
echo "------------------------------------------------------------------------"
eval_job=$(sbatch --parsable --dependency=afterok:$baseline_job:$mc_dropout_job:$ensemble_job \
    scripts/evaluate_classification.sbatch)
echo "✓ Submitted evaluation (Job ID: $eval_job)"
echo "  (Will start after all training jobs complete)"
echo ""

echo "========================================================================"
echo "All Jobs Submitted!"
echo "========================================================================"
echo ""
echo "Job Summary:"
echo "  Baseline:     $baseline_job"
echo "  MC Dropout:   $mc_dropout_job"
echo "  Ensemble:     $ensemble_job"
echo "  Evaluation:   $eval_job (dependent)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs in:"
echo "  runs/classification/*/train_*.out"
echo ""
echo "========================================================================"
