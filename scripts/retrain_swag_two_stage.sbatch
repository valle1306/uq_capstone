#!/bin/bash
#SBATCH --job-name=swag_2stage
#SBATCH --output=logs/swag_two_stage_%j.out
#SBATCH --error=logs/swag_two_stage_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

echo "=========================================="
echo "Two-Stage SWAG Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Navigate to project directory
cd /scratch/$USER/uq_capstone

# Run two-stage SWAG training
python src/retrain_swag_two_stage.py \
    --dataset chest_xray \
    --data_path data \
    --baseline_path runs/classification/baseline/best_model.pth \
    --output_dir runs/classification/swag_two_stage \
    --arch resnet18 \
    --batch_size 32 \
    --epochs 50 \
    --collection_start 20 \
    --base_lr 0.0001 \
    --max_lr 0.001 \
    --cycle_length 5 \
    --weight_decay 0.0001 \
    --swag_lr 0.0001 \
    --num_snapshots 30 \
    --device cuda

echo ""
echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="
