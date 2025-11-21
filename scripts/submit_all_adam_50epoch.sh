#!/bin/bash
# Submit all Adam 50-epoch experiments
# Run this script from /scratch/hpl14/uq_capstone

echo "Submitting Adam 50-epoch experiments..."
echo ""

# Submit Baseline
echo "Submitting Baseline Adam 50 epochs..."
JOB1=$(sbatch scripts/submit_adam_50epoch_baseline.sh | awk '{print $4}')
echo "  Job ID: $JOB1"

# Submit MC Dropout
echo "Submitting MC Dropout Adam 50 epochs..."
JOB2=$(sbatch scripts/submit_adam_50epoch_mcdropout.sh | awk '{print $4}')
echo "  Job ID: $JOB2"

# Submit Ensemble
echo "Submitting Ensemble Adam 50 epochs..."
JOB3=$(sbatch scripts/submit_adam_50epoch_ensemble.sh | awk '{print $4}')
echo "  Job ID: $JOB3"

# Submit SWAG
echo "Submitting SWAG Adam 50 epochs..."
JOB4=$(sbatch scripts/submit_adam_50epoch_swag.sh | awk '{print $4}')
echo "  Job ID: $JOB4"

echo ""
echo "All jobs submitted!"
echo "Job IDs: $JOB1, $JOB2, $JOB3, $JOB4"
echo ""
echo "Monitor with: squeue -u hpl14"
echo "Check logs: tail -f logs/baseline_adam_50_$JOB1.out"
