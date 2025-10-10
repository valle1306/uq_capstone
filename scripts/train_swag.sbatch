#!/bin/bash
#SBATCH --job-name=swag_train
#SBATCH --output=runs/swag/train_%j.out
#SBATCH --error=runs/swag/train_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

# SWAG Training Job for Brain Tumor Segmentation
# Trains U-Net with Stochastic Weight Averaging-Gaussian (Maddox et al. 2019)

echo "========================================"
echo "SWAG Training Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================"

# Load CUDA
module purge
module load cuda/12.1.0
echo "✓ Loaded CUDA 12.1.0"

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone
echo "✓ Activated uq_capstone environment"

# Check GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Create output directory
mkdir -p runs/swag
echo "✓ Created output directory: runs/swag"

# Set Python path
export PYTHONPATH=/scratch/hpl14/uq_capstone:$PYTHONPATH

# Run SWAG training
echo ""
echo "========================================"
echo "Starting SWAG Training..."
echo "========================================"
echo "Configuration:"
echo "  - Total epochs: 30"
echo "  - SWAG start: epoch 15"
echo "  - Initial LR: 1e-3"
echo "  - SWAG LR: 1e-4"
echo "  - Max models (K): 20"
echo "  - Batch size: 8"
echo "========================================"
echo ""

python src/train_swag.py \
    --data_dir /scratch/hpl14/uq_capstone/data/brats_subset_npz \
    --output_dir runs/swag \
    --epochs 30 \
    --swag_start 15 \
    --batch 8 \
    --lr 1e-3 \
    --swag_lr 1e-4 \
    --max_models 20 \
    --collect_freq 1 \
    --in_ch 1 \
    --device cuda

echo ""
echo "========================================"
echo "SWAG Training Complete!"
echo "========================================"
echo "Outputs saved to: runs/swag/"
echo "  - swag_model.pth (SWAG statistics)"
echo "  - best_base_model.pth (best checkpoint)"
echo "  - history.json"
echo "  - config.json"
echo "========================================"
