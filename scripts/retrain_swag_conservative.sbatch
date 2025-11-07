#!/bin/bash
#SBATCH --job-name=swag_cons       # Job name
#SBATCH --partition=gpu             # Partition
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=4           # CPU cores
#SBATCH --mem=32GB                  # Memory
#SBATCH --time=24:00:00             # Time limit
#SBATCH --output=logs/swag_conservative_%j.out
#SBATCH --error=logs/swag_conservative_%j.err

# SWAG Retraining with CONSERVATIVE Learning Rates
# Maddox et al. 2019 with adjustments for small dataset

echo "========================================"
echo "SWAG Training - Conservative LR"
echo "Start time: $(date)"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Navigate to project directory
cd /scratch/$USER/uq_capstone

# Run training
python src/retrain_swag_conservative.py \
    --dataset chest_xray \
    --output_dir runs/classification/swag_classification \
    --arch resnet18 \
    --epochs 50 \
    --swag_start 30 \
    --batch_size 32 \
    --lr 0.01 \
    --swa_lr 0.005 \
    --momentum 0.9 \
    --weight_decay 0.0001 \
    --max_models 20 \
    --num_workers 4

echo "========================================"
echo "Training Complete!"
echo "End time: $(date)"
echo "========================================"

# Show output files
echo "Output files:"
ls -lh runs/classification/swag_classification/

# Show training summary
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
