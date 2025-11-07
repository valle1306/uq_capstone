#!/bin/bash
#SBATCH --job-name=swag_proper
#SBATCH --output=/scratch/%u/uq_capstone/logs/swag_proper_%j.out
#SBATCH --error=/scratch/%u/uq_capstone/logs/swag_proper_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

echo "========================================"
echo "SWAG Proper Retraining - Following Paper"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================"

# Navigate to project directory
cd /scratch/$USER/uq_capstone

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run SWAG retraining with proper paper settings
python src/retrain_swag_proper.py \
    --dataset chest_xray \
    --epochs 50 \
    --swag_start 30 \
    --batch_size 32 \
    --lr 0.05 \
    --swa_lr 0.01 \
    --momentum 0.9 \
    --weight_decay 1e-4 \
    --device cuda \
    --output_dir runs/classification/swag_classification

echo ""
echo "========================================"
echo "Training Complete!"
echo "End time: $(date)"
echo "========================================"

# Show output files
echo "Output files:"
ls -lh runs/classification/swag_classification/

echo ""
echo "Training history summary:"
python -c "
import json
with open('runs/classification/swag_classification/training_history.json') as f:
    hist = json.load(f)
print(f'Final train accuracy: {hist[\"train_acc\"][-1]:.2f}%')
print(f'Final val accuracy: {hist[\"val_acc\"][-1]:.2f}%')
print(f'Final test accuracy: {hist[\"test_acc\"][-1]:.2f}%')
print(f'SWAG snapshots collected: {hist[\"swag_snapshots\"]}')
"
