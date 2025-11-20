#!/bin/bash
#SBATCH --job-name=ensemble_adam_300
#SBATCH --output=logs/ensemble_adam_300_%A_%a.log
#SBATCH --error=logs/ensemble_adam_300_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4

# Deep Ensemble Training with Adam - 300 Epochs
# Train 5 members in parallel using SLURM job array
# Expected runtime: ~6 hours per member

echo "========================================"
echo "Deep Ensemble Training with Adam - 300 Epochs"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
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

# Run training
python src/train_ensemble_adam_300.py \
    --data_dir data/chest_xray \
    --batch_size 32 \
    --num_workers 4 \
    --model resnet50 \
    --pretrained \
    --epochs 300 \
    --lr 0.0001 \
    --lr_final 0.00005 \
    --weight_decay 5e-4 \
    --num_members 5 \
    --save_dir results/ensemble_adam_300 \
    --seed 42

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
