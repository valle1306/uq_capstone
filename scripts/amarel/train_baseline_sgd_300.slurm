#!/bin/bash
#SBATCH --job-name=baseline_sgd_300
#SBATCH --output=logs/baseline_sgd_300_%j.log
#SBATCH --error=logs/baseline_sgd_300_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Baseline Training with SGD - 300 Epochs
# Matches original SWAG paper training duration
# Expected runtime: ~6 hours for 300 epochs

echo "========================================"
echo "Baseline Training with SGD - 300 Epochs"
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
python src/train_baseline_sgd_300.py \
    --data_dir data/chest_xray \
    --batch_size 32 \
    --num_workers 4 \
    --arch resnet50 \
    --pretrained \
    --epochs 300 \
    --lr_init 0.01 \
    --swa_start 162 \
    --swa_lr 0.005 \
    --momentum 0.9 \
    --wd 1e-4 \
    --output_dir results/baseline_sgd_300 \
    --seed 1

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
