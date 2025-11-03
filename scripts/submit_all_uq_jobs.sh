#!/bin/bash

# Submit all UQ training jobs to Amarel queue
# Usage on Amarel: ./scripts/submit_all_uq_jobs.sh

set -e

echo "==========================================================="
echo "UQ Pipeline Job Submission to Amarel Queue"
echo "==========================================================="
echo "This script will submit:"
echo "  1. Baseline training (50 epochs)"
echo "  2. MC Dropout training (50 epochs)"
echo "  3. Deep Ensemble training (50 epochs x 5 members)"
echo "  4. SWAG training (50 epochs)"
echo "  5. Comprehensive evaluation"
echo ""
echo "Estimated total runtime: ~120 hours sequential"
echo "==========================================================="
echo ""

# Check if we're on Amarel (look for /scratch directory)
if [ ! -d "/scratch/$USER" ]; then
    echo "ERROR: Not on Amarel HPC (no /scratch/$USER found)"
    echo "Run this script on Amarel: ssh YOUR_USERNAME@amarel.rutgers.edu"
    exit 1
fi

cd /scratch/$USER/uq_capstone

# Check if SBATCH scripts exist
for script in train_classifier_baseline.sbatch train_classifier_mc_dropout.sbatch \
              train_classifier_ensemble.sbatch train_classifier_swag.sbatch; do
    if [ ! -f "scripts/$script" ]; then
        echo "ERROR: scripts/$script not found"
        exit 1
    fi
done

echo ""
echo "Step 1: Submitting Baseline training (50 epochs, ~12-15h)..."
BASELINE_JID=$(sbatch scripts/train_classifier_baseline.sbatch | awk '{print $NF}')
echo "  Baseline job ID: $BASELINE_JID"

echo ""
echo "Step 2: Submitting MC Dropout training (50 epochs, ~12-15h)..."
echo "  (Can run in parallel with Baseline)"
MC_JID=$(sbatch scripts/train_classifier_mc_dropout.sbatch | awk '{print $NF}')
echo "  MC Dropout job ID: $MC_JID"

echo ""
echo "Step 3: Submitting Deep Ensemble training (5 members x 50 epochs, ~60-75h)..."
echo "  (Can run in parallel, but will run sequentially within the job)"
ENSEMBLE_JID=$(sbatch scripts/train_classifier_ensemble.sbatch | awk '{print $NF}')
echo "  Ensemble job ID: $ENSEMBLE_JID"

echo ""
echo "Step 4: Submitting SWAG training (50 epochs, ~12-15h)..."
echo "  (Can run in parallel with other trainers)"
SWAG_JID=$(sbatch scripts/train_classifier_swag.sbatch | awk '{print $NF}')
echo "  SWAG job ID: $SWAG_JID"

echo ""
echo "==========================================================="
echo "All training jobs submitted!"
echo "==========================================================="
echo ""
echo "Job IDs:"
echo "  Baseline: $BASELINE_JID"
echo "  MC Dropout: $MC_JID"
echo "  Ensemble: $ENSEMBLE_JID"
echo "  SWAG: $SWAG_JID"
echo ""
echo "Monitor job status:"
echo "  squeue -u \$USER"
echo ""
echo "View job output (after completion):"
echo "  cat runs/classification/baseline/train_*.out"
echo "  cat runs/classification/mc_dropout/train_*.out"
echo "  cat runs/classification/ensemble/train_all_*.out"
echo "  cat runs/classification/swag_classification/train_*.out"
echo ""
echo "Once all training is complete, run evaluation:"
echo "  sbatch scripts/evaluate_classification.sbatch"
echo ""
echo "Or manually run:"
echo "  python src/evaluate_uq_classification.py \\"
echo "    --dataset chest_xray \\"
echo "    --data_dir data/chest_xray \\"
echo "    --baseline_path runs/classification/baseline/best_model.pth \\"
echo "    --mc_dropout_path runs/classification/mc_dropout/best_model.pth \\"
echo "    --ensemble_dir runs/classification/ensemble \\"
echo "    --n_ensemble 5 \\"
echo "    --swag_path runs/classification/swag_classification/swag_model.pth \\"
echo "    --mc_samples 20 \\"
echo "    --swag_samples 30 \\"
echo "    --dropout_rate 0.3 \\"
echo "    --batch_size 32 \\"
echo "    --num_workers 4 \\"
echo "    --output_dir runs/classification/evaluation \\"
echo "    --device cuda"
echo ""
echo "Download results to local machine:"
echo "  scp -r \$USER@amarel.rutgers.edu:/scratch/\$USER/uq_capstone/runs/classification ./runs/"
echo ""
echo "==========================================================="

