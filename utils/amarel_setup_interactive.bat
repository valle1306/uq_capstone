@echo off
REM Interactive Amarel Setup Helper
REM This script will guide you through the Amarel setup process

echo ============================================================
echo   UQ Capstone - Amarel Setup Helper
echo ============================================================
echo.

REM Step 1: Get NetID
set /p NETID="Enter your Rutgers NetID: "
if "%NETID%"=="" (
    echo ERROR: NetID cannot be empty!
    pause
    exit /b 1
)

echo.
echo Great! Your NetID is: %NETID%
echo.

REM Step 2: Test SSH connection
echo Step 1: Testing SSH connection to Amarel...
echo (You'll be prompted for your password)
echo.
ssh -o ConnectTimeout=10 %NETID%@amarel.rutgers.edu "echo Connection successful!"

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Cannot connect to Amarel. Please check:
    echo 1. Your NetID is correct
    echo 2. Your password is correct
    echo 3. You have access to Amarel
    echo 4. VPN is connected if required
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Step 2: Creating remote directory structure...
echo ============================================================
echo.

ssh %NETID%@amarel.rutgers.edu "mkdir -p /scratch/%NETID%/uq_capstone/{data/brats,scripts,src,envs,notebooks,runs}"

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create directories
    pause
    exit /b 1
)

echo Directories created successfully!
echo.

REM Step 3: Upload files
echo ============================================================
echo Step 3: Uploading files to Amarel...
echo ============================================================
echo.
echo This will take several minutes depending on your connection...
echo.

echo [1/6] Uploading data (this is the largest)...
scp -r data\brats %NETID%@amarel.rutgers.edu:/scratch/%NETID%/uq_capstone/data/

echo [2/6] Uploading scripts...
scp scripts\*.py scripts\*.sbatch %NETID%@amarel.rutgers.edu:/scratch/%NETID%/uq_capstone/scripts/

echo [3/6] Uploading source code...
scp src\*.py %NETID%@amarel.rutgers.edu:/scratch/%NETID%/uq_capstone/src/

echo [4/6] Uploading environment files...
scp envs\conda_env.yml %NETID%@amarel.rutgers.edu:/scratch/%NETID%/uq_capstone/envs/

echo [5/6] Uploading requirements.txt...
scp requirements.txt %NETID%@amarel.rutgers.edu:/scratch/%NETID%/uq_capstone/

echo [6/6] Uploading notebooks...
scp notebooks\*.ipynb %NETID%@amarel.rutgers.edu:/scratch/%NETID%/uq_capstone/notebooks/ 2>nul

if %ERRORLEVEL% neq 0 (
    echo WARNING: Some files may not have uploaded. Check the output above.
)

echo.
echo ============================================================
echo Upload complete!
echo ============================================================
echo.

REM Step 4: Generate next steps script
echo Creating setup script for Amarel...
echo.

(
echo #!/bin/bash
echo # Auto-generated setup script for Amarel
echo # Run this after SSH-ing to Amarel
echo.
echo echo "============================================================"
echo echo "  Setting up UQ Capstone Environment on Amarel"
echo echo "============================================================"
echo echo ""
echo.
echo cd /scratch/%NETID%/uq_capstone
echo.
echo echo "Step 1: Loading modules..."
echo module purge
echo module load conda
echo module load cuda/11.8
echo.
echo echo "Step 2: Creating conda environment..."
echo echo "This will take 5-10 minutes..."
echo conda env create -f envs/conda_env.yml
echo.
echo echo "Step 3: Activating environment..."
echo source activate uq_capstone
echo.
echo echo "Step 4: Verifying installation..."
echo python --version
echo python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo.
echo echo "Step 5: Validating data..."
echo python scripts/validate_brats_data.py --data_root data/brats --n_samples 3
echo.
echo echo ""
echo echo "============================================================"
echo echo "Setup complete! You can now submit jobs."
echo echo "Try: sbatch scripts/test_training.sbatch"
echo echo "============================================================"
) > amarel_setup.sh

scp amarel_setup.sh %NETID%@amarel.rutgers.edu:/scratch/%NETID%/uq_capstone/

echo.
echo ============================================================
echo SUCCESS! Next steps:
echo ============================================================
echo.
echo 1. SSH to Amarel:
echo    ssh %NETID%@amarel.rutgers.edu
echo.
echo 2. Navigate to project:
echo    cd /scratch/%NETID%/uq_capstone
echo.
echo 3. Run the setup script:
echo    chmod +x amarel_setup.sh
echo    bash amarel_setup.sh
echo.
echo OR run commands manually:
echo    module load conda cuda/11.8
echo    conda env create -f envs/conda_env.yml
echo    conda activate uq_capstone
echo    python scripts/validate_brats_data.py --data_root data/brats
echo.
echo 4. Submit test job:
echo    sbatch scripts/test_training.sbatch
echo.
echo 5. Monitor job:
echo    squeue -u %NETID%
echo    tail -f runs/test_*.out
echo.
echo ============================================================
echo.
echo Press any key to open SSH connection to Amarel...
pause

echo.
echo Opening SSH session...
ssh %NETID%@amarel.rutgers.edu

exit /b 0
