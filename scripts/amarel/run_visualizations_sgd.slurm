#!/bin/bash
#SBATCH --job-name=visualize_sgd
#SBATCH --output=logs/visualize_sgd_%j.out
#SBATCH --error=logs/visualize_sgd_%j.err
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Generate visualizations for all completed SGD 50-epoch experiments
# Creates reliability diagrams, calibration curves, and comparison plots

echo "========================================"
echo "Visualization - SGD Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "========================================"

# Activate conda environment
source ~/.bashrc
conda activate uq_capstone

# Navigate to project root
cd /scratch/hpl14/uq_capstone

echo "Python: $(which python)"
echo "========================================"

# Create output directory
mkdir -p visualizations/sgd_50epochs

# Run visualization script
echo ""
echo "=== Generating UQ Visualizations ==="
python src/visualize_uq_results.py \
    --results_dirs \
        runs/classification/baseline_sgd \
        runs/classification/mc_dropout_sgd \
        runs/classification/swag_proper \
        runs/classification/ensemble_sgd \
    --method_names \
        "Baseline (SGD)" \
        "MC Dropout (SGD)" \
        "SWAG (SGD)" \
        "Ensemble (SGD)" \
    --output_dir visualizations/sgd_50epochs \
    --include_conformal

echo ""
echo "=== Generating Detailed UQ Analysis ==="
python analysis/visualize_uq.py \
    --results_dir runs/classification \
    --methods baseline_sgd mc_dropout_sgd swag_proper ensemble_sgd \
    --output_dir visualizations/sgd_50epochs/detailed

echo "========================================"
echo "Visualizations saved to: visualizations/sgd_50epochs/"
echo "Finished: $(date)"
echo "========================================"
