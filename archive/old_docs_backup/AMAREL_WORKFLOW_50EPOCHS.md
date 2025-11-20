# Full UQ Pipeline on Amarel - 50 Epoch Experiments

This guide walks through running the complete uncertainty quantification (UQ) pipeline on Rutgers Amarel HPC with **50 epochs per method** for improved performance.

## Quick Summary

**Methods to train:**
1. Baseline (ResNet-18) - 50 epochs
2. MC Dropout - 50 epochs  
3. Deep Ensemble (5 members) - 50 epochs each
4. SWAG - 50 epochs (snapshot collection from epoch 30)
5. Evaluation (Baseline + MC Dropout + Ensemble + SWAG + Conformal Risk Control)

**Total runtime estimate:**
- Baseline: ~12-15 hours (GPU)
- MC Dropout: ~12-15 hours (GPU)
- Ensemble: ~60-75 hours (GPU, 5 members sequential)
- SWAG: ~12-15 hours (GPU)
- Evaluation: ~1-2 hours (GPU)
- **Total: ~110-120 hours sequential** (or ~15-20 hours if run in parallel on separate GPUs)

## Step 1: Clone Repository on Amarel

```bash
# SSH to Amarel
ssh YOUR_USERNAME@amarel.rutgers.edu

# Navigate to scratch
cd /scratch/$USER

# Clone the repository
git clone https://github.com/valle1306/uq_capstone.git
cd uq_capstone

# Verify key files exist
ls -la scripts/train_classifier_*.sbatch
ls -la src/train_classifier_*.py
ls -la src/evaluate_uq_classification.py
```

## Step 2: Upload Dataset to Amarel

The dataset needs to be available on Amarel. You can either:

### Option A: Download directly on Amarel (if internet available)
```bash
cd /scratch/$USER/uq_capstone/data
# Download chest_xray dataset (if available via direct download link)
```

### Option B: Upload from your local machine
```bash
# On your local machine (Windows PowerShell)
scp -r data/chest_xray YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/data/
```

**Verify dataset:**
```bash
# On Amarel
cd /scratch/$USER/uq_capstone/data/chest_xray
ls -la train/ val/ test/
```

## Step 3: Set Up Conda Environment on Amarel

```bash
# SSH to Amarel
ssh YOUR_USERNAME@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone

# Initialize conda (if not already done)
source ~/.bashrc
conda init bash
source ~/.bashrc

# Create environment (if not already created)
conda create -n uq_capstone python=3.9 -y

# Activate environment
conda activate uq_capstone

# Install PyTorch with CUDA 12.1 (Amarel GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install scikit-learn tqdm numpy pandas

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x+cu121
CUDA: True
```

## Step 4: Submit Training Jobs to Amarel Queue

### Sequential Submission (Recommended for First Run)

```bash
cd /scratch/$USER/uq_capstone

# 1. Submit Baseline (50 epochs, ~12-15 hours)
sbatch scripts/train_classifier_baseline.sbatch
# Returns job ID, e.g., Submitted batch job 12345678

# 2. Wait for Baseline to complete, then submit MC Dropout
sbatch scripts/train_classifier_mc_dropout.sbatch

# 3. Submit Ensemble (5 members, ~60-75 hours total)
sbatch scripts/train_classifier_ensemble.sbatch

# 4. Submit SWAG (50 epochs, ~12-15 hours)
sbatch scripts/train_classifier_swag.sbatch
```

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job details
squeue -j <JOB_ID>

# Watch job status (update every 5 seconds)
watch -n 5 'squeue -u $USER'

# View job output in real-time
tail -f runs/classification/baseline/train_<JOB_ID>.out

# After job completes, view full output
cat runs/classification/baseline/train_<JOB_ID>.out
```

### Alternative: Parallel Submission (if GPUs available)

If Amarel has multiple GPUs available, you can submit all at once:

```bash
sbatch scripts/train_classifier_baseline.sbatch
sbatch scripts/train_classifier_mc_dropout.sbatch
sbatch scripts/train_classifier_ensemble.sbatch
sbatch scripts/train_classifier_swag.sbatch

# Monitor all jobs
squeue -u $USER
```

## Step 5: Monitor Training

```bash
# SSH to Amarel in a new terminal
ssh YOUR_USERNAME@amarel.rutgers.edu
cd /scratch/$USER/uq_capstone

# Check GPU usage during training
nvidia-smi

# View training log (live updates)
tail -f runs/classification/baseline/train_*.out

# Check disk space
du -sh runs/classification/

# List saved checkpoints
find runs/classification -name "*.pth" | head -20
```

## Step 6: Verify Training Outputs

After each training completes:

```bash
# Check Baseline outputs
ls -la runs/classification/baseline/
# Should contain: best_model.pth, history.json, config.json

# Check MC Dropout outputs
ls -la runs/classification/mc_dropout/

# Check Ensemble outputs (5 members)
ls -la runs/classification/ensemble/
for i in 0 1 2 3 4; do
    echo "Member $i:"
    ls -la runs/classification/ensemble/member_$i/
done

# Check SWAG outputs
ls -la runs/classification/swag_classification/
# Should contain: swag_model.pth, best_base_model.pth, history.json, config.json

# Verify all best_model.pth files exist
echo "=== Model Checkpoint Summary ==="
ls -lh runs/classification/baseline/best_model.pth
ls -lh runs/classification/mc_dropout/best_model.pth
ls -lh runs/classification/ensemble/member_*/best_model.pth
ls -lh runs/classification/swag_classification/swag_model.pth
```

## Step 7: Run Evaluation

Once all training is complete:

```bash
# Submit evaluation job
sbatch scripts/evaluate_classification.sbatch

# Or run directly for immediate results
python src/evaluate_uq_classification.py \
    --dataset chest_xray \
    --data_dir data/chest_xray \
    --baseline_path runs/classification/baseline/best_model.pth \
    --mc_dropout_path runs/classification/mc_dropout/best_model.pth \
    --ensemble_dir runs/classification/ensemble \
    --n_ensemble 5 \
    --swag_path runs/classification/swag_classification/swag_model.pth \
    --swag_samples 30 \
    --mc_samples 20 \
    --dropout_rate 0.3 \
    --batch_size 32 \
    --num_workers 4 \
    --output_dir runs/classification/evaluation \
    --device cuda

# View results
cat runs/classification/evaluation/all_results.json
```

## Step 8: Download Results to Local Machine

```bash
# On your local machine (Windows PowerShell)
cd c:\Users\lpnhu\Downloads\uq_capstone

# Download all results
scp -r YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/runs/classification ./runs/

# Download logs
scp -r YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/runs/classification/*/train_*.out ./runs/

# View results locally
cat runs/classification/evaluation/all_results.json
```

## Troubleshooting

### Job cancelled/timeout
```bash
# Increase time limit in SBATCH script
#SBATCH --time=48:00:00  # Increase hours as needed

# Resubmit
sbatch scripts/train_classifier_baseline.sbatch
```

### GPU memory issues
```bash
# Reduce batch size in SBATCH script
--batch_size 16  # instead of 32

# Or reduce num_workers
--num_workers 2  # instead of 4
```

### CUDA out of memory
```bash
# Check GPU memory
nvidia-smi

# Reduce batch size or use CPU (not recommended)
--device cpu
```

### Missing dataset
```bash
# Check if data exists
ls -la data/chest_xray/train/

# If not, upload it
# On local: scp -r data/chest_xray YOUR_USERNAME@amarel.rutgers.edu:/scratch/$USER/uq_capstone/data/
```

### Need to cancel jobs
```bash
# Cancel single job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER

# Kill training in progress
pkill -f train_classifier
```

## Expected Results

After 50-epoch training, expect:

- **Baseline**: ~88-92% accuracy
- **MC Dropout**: ~89-93% accuracy
- **Deep Ensemble**: ~90-94% accuracy (best performance)
- **SWAG**: ~87-91% accuracy
- **Calibration**: ECE ~0.02-0.04, Brier ~0.15-0.20

These should show significant improvement over the 5-epoch preliminary runs.

## Key Files Updated for 50-Epoch Runs

- `scripts/train_classifier_baseline.sbatch` - 50 epochs, 24h time limit
- `scripts/train_classifier_mc_dropout.sbatch` - 50 epochs, 24h time limit
- `scripts/train_classifier_ensemble.sbatch` - 50 epochs, 48h time limit
- `scripts/train_classifier_swag.sbatch` - NEW: 50 epochs with SWAG snapshot collection from epoch 30
- `scripts/evaluate_classification.sbatch` - Updated to include SWAG evaluation
- `src/evaluate_uq_classification.py` - Enhanced with SWAG evaluation

## Next Steps After Results

1. Download results locally
2. Run visualization script: `python scripts/visualize_uq_results.py`
3. Generate updated presentation: `python scripts/generate_classification_presentation.py`
4. Compare 5-epoch vs 50-epoch performance
5. Prepare paper/report with findings

---

**For more details on individual methods, see:**
- Baseline: `docs/CLASSIFICATION_SETUP_GUIDE.md`
- MC Dropout: `docs/CLASSIFICATION_SETUP_GUIDE.md`
- Ensemble: `docs/CLASSIFICATION_SETUP_GUIDE.md`
- SWAG: `docs/SWAG_GUIDE.md`
- Conformal Risk Control: `docs/CRITICAL_ISSUES_ANALYSIS.md`

