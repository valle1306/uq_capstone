#!/bin/bash
#SBATCH --job-name=swag_adam
#SBATCH --output=logs/swag_adam_%j.log
#SBATCH --error=logs/swag_adam_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# SWAG Training with Adam Optimizer
# Expected runtime: ~1 hour for 50 epochs

echo "========================================"
echo "SWAG Training with Adam"
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
python src/train_swag_adam.py \
    --data_dir data/chest_xray \
    --batch_size 32 \
    --num_workers 4 \
    --model resnet50 \
    --pretrained \
    --epochs 50 \
    --lr 0.0001 \
    --lr_final 0.00005 \
    --weight_decay 5e-4 \
    --swa_start 27 \
    --max_num_models 20 \
    --swag_rank 20 \
    --num_samples 30 \
    --scale 0.5 \
    --save_dir results/swag_adam \
    --seed 42

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
