#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --output=runs/ensemble/member_%a/train_%j.out
#SBATCH --error=runs/ensemble/member_%a/train_%j.err
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "========================================="
echo "ENSEMBLE MEMBER $SLURM_ARRAY_TASK_ID TRAINING"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Setup
export PROJECT_ROOT=/scratch/hpl14/uq_capstone
cd $PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module load cuda/12.1.0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate uq_capstone

# Create output directory for this member
mkdir -p runs/ensemble/member_$SLURM_ARRAY_TASK_ID

# Different seed for each ensemble member
SEED=$((42 + SLURM_ARRAY_TASK_ID))

# Train ensemble member
python src/train_ensemble_member.py \
    --data_dir /scratch/hpl14/uq_capstone/data/brats_subset_npz \
    --output_dir runs/ensemble/member_$SLURM_ARRAY_TASK_ID \
    --epochs 30 \
    --batch 8 \
    --lr 1e-3 \
    --in_ch 1 \
    --dropout 0.0 \
    --seed $SEED \
    --member_id $SLURM_ARRAY_TASK_ID

echo "End time: $(date)"
