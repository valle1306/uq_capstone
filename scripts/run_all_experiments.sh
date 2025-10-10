#!/bin/bash
# Master script to run all UQ experiments sequentially
# This submits jobs with dependencies so they run in order

echo "========================================="
echo "UNCERTAINTY QUANTIFICATION EXPERIMENT PIPELINE"
echo "========================================="
echo ""

# Check if on Amarel
if [[ ! "$HOSTNAME" =~ amarel ]]; then
    echo "ERROR: This script must be run on Amarel login node"
    echo "Please SSH to Amarel first: ssh hpl14@amarel.rutgers.edu"
    exit 1
fi

# Navigate to project
cd /scratch/hpl14/uq_capstone || exit 1

echo "Step 1: Submitting baseline training..."
JOB1=$(sbatch --parsable scripts/train_baseline.sbatch)
echo "  Job ID: $JOB1"

echo "Step 2: Submitting MC Dropout training..."
JOB2=$(sbatch --parsable scripts/train_mc_dropout.sbatch)
echo "  Job ID: $JOB2"

echo "Step 3: Submitting Deep Ensemble training (5 members in parallel)..."
JOB3=$(sbatch --parsable scripts/train_ensemble.sbatch)
echo "  Job ID: $JOB3 (array job, 5 members)"

echo "Step 4: Submitting SWAG training..."
JOB4=$(sbatch --parsable scripts/train_swag.sbatch)
echo "  Job ID: $JOB4"

# Wait for all training to complete before evaluation
echo "Step 5: Submitting evaluation (depends on completion of all training)..."
JOB5=$(sbatch --parsable --dependency=afterok:$JOB1:$JOB2:$JOB3:$JOB4 scripts/evaluate_uq.sbatch)
echo "  Job ID: $JOB5"

echo ""
echo "========================================="
echo "ALL JOBS SUBMITTED!"
echo "========================================="
echo ""
echo "Job Summary:"
echo "  Baseline:     $JOB1"
echo "  MC Dropout:   $JOB2"
echo "  Ensemble:     $JOB3 (array 0-4)"
echo "  SWAG:         $JOB4"
echo "  Evaluation:   $JOB5 (depends on above)"
echo ""
echo "Monitor jobs with: squeue -u hpl14"
echo ""
echo "Expected timeline:"
echo "  - Baseline training:  ~2-3 hours"
echo "  - MC Dropout:         ~2-3 hours"
echo "  - Ensemble (5x):      ~2-3 hours (parallel)"
echo "  - SWAG:               ~2-3 hours"
echo "  - Evaluation:         ~30 minutes"
echo "  - Total:              ~6-8 hours"
echo ""
echo "Check results:"
echo "  - Baseline:     runs/baseline/best_model.pth"
echo "  - MC Dropout:   runs/mc_dropout/best_model.pth"
echo "  - Ensemble:     runs/ensemble/member_*/best_model.pth"
echo "  - SWAG:         runs/swag/swag_model.pth"
echo "  - Evaluation:   runs/evaluation/results.json"
echo "  - Comparison:   runs/evaluation/comparison.png"
echo ""
echo "========================================="
