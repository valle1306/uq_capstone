#!/bin/bash
#SBATCH --job-name=retrain_mc_dropout
#SBATCH --output=logs/retrain_mc_dropout_%j.out
#SBATCH --error=logs/retrain_mc_dropout_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

echo "========================================="
echo "Retraining MC Dropout from Baseline"
echo "========================================="
echo ""

source activate uq_capstone
cd /scratch/$USER/uq_capstone

python src/retrain_mc_dropout.py \
    --baseline_path runs/classification/baseline/best_model.pth \
    --dropout_rate 0.2 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --device cuda \
    --output_dir runs/classification/mc_dropout

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ MC Dropout retrain COMPLETED"
else
    echo ""
    echo "✗ MC Dropout retrain FAILED"
    exit 1
fi
