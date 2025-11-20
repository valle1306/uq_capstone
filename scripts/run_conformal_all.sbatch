#!/bin/bash
#SBATCH --job-name=conformal_all
#SBATCH --output=/scratch/hpl14/uq_capstone/logs/conformal_all_%j.out
#SBATCH --error=/scratch/hpl14/uq_capstone/logs/conformal_all_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo "=========================================="
echo "Running Conformal Prediction on All Methods"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

cd /scratch/hpl14/uq_capstone || exit 1

# Activate environment
module purge
module load cuda/11.8.0
eval "$(conda shell.bash hook)"
conda activate uq || source venv/bin/activate || echo "Using system python"

# Create output directory
mkdir -p runs/classification/conformal

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Methods to evaluate
METHODS=("baseline" "swag_sgd" "swag_adam" "ensemble" "dropout")

for method in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Running conformal for: $method"
    echo "=========================================="
    
    MODEL_PATH="runs/classification/$method/best_model.pth"
    OUTPUT_DIR="runs/classification/$method"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "⚠️  Model not found: $MODEL_PATH"
        echo "Skipping $method"
        echo ""
        continue
    fi
    
    python src/conformal_prediction.py \
        --model_path "$MODEL_PATH" \
        --dataset chest_xray \
        --alpha 0.1 \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 64
    
    if [ $? -eq 0 ]; then
        echo "✓ Conformal completed for $method"
    else
        echo "❌ Conformal failed for $method"
    fi
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Checking generated files:"
for method in "${METHODS[@]}"; do
    CONF_FILE="runs/classification/$method/conformal_results.json"
    if [ -f "$CONF_FILE" ]; then
        echo "✓ $method: conformal_results.json exists"
    else
        echo "❌ $method: conformal_results.json NOT FOUND"
    fi
done

echo ""
echo "End time: $(date)"
echo "Job completed!"
