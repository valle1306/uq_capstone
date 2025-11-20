#!/bin/bash
#SBATCH --job-name=ensemble_sgd_300
#SBATCH --output=logs/ensemble_sgd_300_%A_%a.log
#SBATCH --error=logs/ensemble_sgd_300_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4

# Deep Ensemble Training with SGD - 300 Epochs
# Trains 5 ensemble members using job array
# Matches original SWAG paper training duration
# Expected runtime: ~6 hours per member

echo "========================================"
echo "Deep Ensemble Training with SGD - 300 Epochs"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Member ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Navigate to project root
cd /scratch/hpl14/uq_capstone

# Create logs directory
mkdir -p logs

# Print Python and CUDA info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "========================================"

# Run training for this ensemble member
python src/train_ensemble_sgd_300.py \
    --data_dir data/chest_xray \
    --batch_size 32 \
    --num_workers 4 \
    --arch resnet50 \
    --pretrained \
    --member_id $SLURM_ARRAY_TASK_ID \
    --epochs 300 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --scheduler cosine \
    --output_dir results/ensemble_sgd_300 \
    --seed 42

echo "========================================"
echo "Member $SLURM_ARRAY_TASK_ID finished: $(date)"
echo "========================================"
