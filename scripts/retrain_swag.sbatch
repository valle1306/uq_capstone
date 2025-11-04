#!/bin/bash
#SBATCH --job-name=retrain_swag
#SBATCH --output=logs/retrain_swag_%j.out
#SBATCH --error=logs/retrain_swag_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "========================================="
echo "Retraining SWAG from Baseline"
echo "========================================="
echo ""

source activate uq_capstone
cd /scratch/$USER/uq_capstone

python src/retrain_swag.py \
    --baseline_path runs/classification/baseline/best_model.pth \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --swag_start 30 \
    --device cuda \
    --output_dir runs/classification/swag_classification

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ SWAG retrain COMPLETED"
else
    echo ""
    echo "✗ SWAG retrain FAILED"
    exit 1
fi
