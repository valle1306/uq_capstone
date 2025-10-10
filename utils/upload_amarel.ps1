# PowerShell Upload Script for Amarel
# NetID: hpl14

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  UQ Capstone - Upload to Amarel (hpl14)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$NETID = "hpl14"
$REMOTE_DIR = "/scratch/hpl14/uq_capstone"

Write-Host "NetID: $NETID" -ForegroundColor Green
Write-Host "Remote directory: $REMOTE_DIR" -ForegroundColor Green
Write-Host ""

# Test connection
Write-Host "Step 1: Testing SSH connection to Amarel..." -ForegroundColor Yellow
Write-Host "(You'll be prompted for your password)" -ForegroundColor Yellow
Write-Host ""

ssh -o ConnectTimeout=10 hpl14@amarel.rutgers.edu "echo 'Connection successful!'"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Cannot connect to Amarel." -ForegroundColor Red
    Write-Host "Please check your password and network connection." -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 2: Creating remote directories..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

ssh hpl14@amarel.rutgers.edu "mkdir -p $REMOTE_DIR/{data/brats,scripts,src,envs,notebooks,runs}"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create directories" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "✓ Directories created!" -ForegroundColor Green
Write-Host ""

# Upload files
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 3: Uploading files to Amarel..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This will take 5-10 minutes. You'll be prompted for password multiple times." -ForegroundColor Yellow
Write-Host ""

Write-Host "[1/6] Uploading data (~150MB, this takes longest)..." -ForegroundColor Yellow
scp -r data\brats hpl14@amarel.rutgers.edu:$REMOTE_DIR/data/
if ($LASTEXITCODE -eq 0) { Write-Host "  ✓ Data uploaded" -ForegroundColor Green }

Write-Host "[2/6] Uploading scripts..." -ForegroundColor Yellow
scp scripts\*.py hpl14@amarel.rutgers.edu:$REMOTE_DIR/scripts/
scp scripts\*.sbatch hpl14@amarel.rutgers.edu:$REMOTE_DIR/scripts/
if ($LASTEXITCODE -eq 0) { Write-Host "  ✓ Scripts uploaded" -ForegroundColor Green }

Write-Host "[3/6] Uploading source code..." -ForegroundColor Yellow
scp src\*.py hpl14@amarel.rutgers.edu:$REMOTE_DIR/src/
if ($LASTEXITCODE -eq 0) { Write-Host "  ✓ Source code uploaded" -ForegroundColor Green }

Write-Host "[4/6] Uploading environment files..." -ForegroundColor Yellow
scp envs\conda_env.yml hpl14@amarel.rutgers.edu:$REMOTE_DIR/envs/
if ($LASTEXITCODE -eq 0) { Write-Host "  ✓ Environment files uploaded" -ForegroundColor Green }

Write-Host "[5/6] Uploading requirements.txt..." -ForegroundColor Yellow
scp requirements.txt hpl14@amarel.rutgers.edu:$REMOTE_DIR/
if ($LASTEXITCODE -eq 0) { Write-Host "  ✓ Requirements uploaded" -ForegroundColor Green }

Write-Host "[6/6] Uploading notebooks..." -ForegroundColor Yellow
scp notebooks\*.ipynb hpl14@amarel.rutgers.edu:$REMOTE_DIR/notebooks/ 2>$null
Write-Host "  ✓ Notebooks uploaded (if any)" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "✓ Upload Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Create setup script for Amarel
Write-Host "Creating setup script for Amarel..." -ForegroundColor Yellow

$setupScript = @"
#!/bin/bash
# Setup script for UQ Capstone on Amarel
# NetID: hpl14

echo "============================================================"
echo "  UQ Capstone Environment Setup"
echo "============================================================"
echo ""

cd /scratch/hpl14/uq_capstone

echo "Verifying uploaded files..."
echo "Files in project directory:"
ls -lh
echo ""
echo "Files in data/brats:"
ls data/brats/
echo ""

echo "Step 1: Loading modules..."
module purge
module load conda
module load cuda/11.8
echo "✓ Modules loaded"
echo ""

echo "Step 2: Creating conda environment..."
echo "(This takes 5-10 minutes, please be patient...)"
conda env create -f envs/conda_env.yml

if [ \$? -eq 0 ]; then
    echo "✓ Environment created successfully"
else
    echo "✗ Environment creation failed"
    exit 1
fi
echo ""

echo "Step 3: Activating environment..."
source activate uq_capstone
echo "✓ Environment activated"
echo ""

echo "Step 4: Verifying installation..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

echo "Step 5: Validating data..."
python scripts/validate_brats_data.py --data_root data/brats --n_samples 3

if [ \$? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Setup Complete! Everything is ready."
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Submit test job:  sbatch scripts/test_training.sbatch"
    echo "  2. Check status:     squeue -u hpl14"
    echo "  3. Watch output:     tail -f runs/test_*.out"
    echo ""
else
    echo ""
    echo "✗ Data validation failed. Please check the errors above."
    exit 1
fi
"@

$setupScript | Out-File -FilePath "amarel_setup.sh" -Encoding ASCII -NoNewline

Write-Host "Uploading setup script..." -ForegroundColor Yellow
scp amarel_setup.sh hpl14@amarel.rutgers.edu:$REMOTE_DIR/

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "SUCCESS! Next Steps:" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. SSH to Amarel:" -ForegroundColor White
Write-Host "   ssh hpl14@amarel.rutgers.edu" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Run the automated setup:" -ForegroundColor White
Write-Host "   cd /scratch/hpl14/uq_capstone" -ForegroundColor Yellow
Write-Host "   chmod +x amarel_setup.sh" -ForegroundColor Yellow
Write-Host "   bash amarel_setup.sh" -ForegroundColor Yellow
Write-Host ""
Write-Host "OR run commands manually:" -ForegroundColor White
Write-Host "   cd /scratch/hpl14/uq_capstone" -ForegroundColor Yellow
Write-Host "   module load conda cuda/11.8" -ForegroundColor Yellow
Write-Host "   conda env create -f envs/conda_env.yml" -ForegroundColor Yellow
Write-Host "   conda activate uq_capstone" -ForegroundColor Yellow
Write-Host "   python scripts/validate_brats_data.py --data_root data/brats" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Submit test job:" -ForegroundColor White
Write-Host "   sbatch scripts/test_training.sbatch" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. Monitor job:" -ForegroundColor White
Write-Host "   squeue -u hpl14" -ForegroundColor Yellow
Write-Host "   tail -f runs/test_*.out" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$response = Read-Host "Open SSH connection to Amarel now? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "Opening SSH session to hpl14@amarel.rutgers.edu..." -ForegroundColor Green
    Write-Host ""
    ssh hpl14@amarel.rutgers.edu
}

Write-Host ""
Write-Host "Done! See START_HERE.md for detailed instructions." -ForegroundColor Green
