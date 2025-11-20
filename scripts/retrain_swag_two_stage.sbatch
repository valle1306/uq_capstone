#!/bin/bash
#SBATCH --job-name=swag_two_stage
#SBATCH --output=/scratch/hpl14/uq_capstone/logs/swag_two_stage_%j.out
#SBATCH --error=/scratch/hpl14/uq_capstone/logs/swag_two_stage_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Two-Stage SWAG Training for Chest X-Ray Pneumonia Classification
# 
# Motivation: Standard SWAG (SGD/Adam) shows poor performance (78-79% test acc)
# because the model overfits BEFORE snapshot collection begins.
#
# By epoch 30 (collection start):
#   - Train accuracy: ~98% (saturated)
#   - Test accuracy: ~78-82% (plateaued)
#   - SWAG snapshots all sample the same overfit point → no diversity
#
# Solution: Two-stage training
#   Stage 1 (epochs 1-20): Cyclic learning rate (0.0001 ↔ 0.001)
#                          Forces continued exploration of loss surface
#   Stage 2 (epochs 21-50): Fixed low LR (0.0001)
#                          Collects SWAG snapshots from diverse points
#
# Expected outcome: 88-90% test accuracy (vs 78.69% single-stage Adam SWAG)

echo "=========================================="
echo "Two-Stage SWAG Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU info unavailable')"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Activate conda environment
set -e  # Exit on error
source ~/.bashrc
conda activate uq_capstone

# Navigate to project directory
cd /scratch/$USER/uq_capstone

# Verify baseline model exists
if [ ! -f "runs/classification/baseline/best_model.pth" ]; then
    echo "ERROR: Baseline model not found at runs/classification/baseline/best_model.pth"
    echo "Please ensure baseline training completed successfully."
    exit 1
fi

# Create output directories
mkdir -p runs/classification/swag_two_stage
mkdir -p logs

echo "Configuration:"
echo "  Epochs: 50"
echo "  Stage 1 (Exploration): epochs 1-20, cyclic LR 0.0001 ↔ 0.001"
echo "  Stage 2 (Collection): epochs 21-50, fixed LR 0.0001"
echo "  Snapshot interval: every epoch (30 total snapshots)"
echo "  Batch size: 32"
echo "  Baseline: runs/classification/baseline/best_model.pth"
echo ""

# Run two-stage SWAG training
python src/retrain_swag_two_stage.py \
    --data_dir data \
    --baseline_path runs/classification/baseline/best_model.pth \
    --arch resnet18 \
    --epochs 50 \
    --collection_start 21 \
    --snap_interval 1 \
    --base_lr 0.0001 \
    --max_lr 0.001 \
    --cycle_length 5 \
    --batch_size 32 \
    --save_dir runs/classification/swag_two_stage

TRAIN_EXIT=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Training completed successfully!"
    
    # Count snapshots
    N_SNAPS=$(find runs/classification/swag_two_stage -name "snapshot_epoch_*.pth" 2>/dev/null | wc -l)
    echo "  Snapshots collected: $N_SNAPS"
    
    # Show final model
    if [ -f "runs/classification/swag_two_stage/final_model.pth" ]; then
        FINAL_SIZE=$(du -h runs/classification/swag_two_stage/final_model.pth | cut -f1)
        echo "  Final model size: $FINAL_SIZE"
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. Download results from local machine:"
    echo "   scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/swag_two_stage ."
    echo ""
    echo "2. Run conformal prediction:"
    echo "   python src/run_conformal_all.py --methods swag_two_stage"
    echo ""
    echo "3. Compare results in thesis_draft.md"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo "Job completed: $(date)"
echo "=========================================="
