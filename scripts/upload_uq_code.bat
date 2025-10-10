@echo off
REM Batch script to upload UQ code to Amarel from Windows
REM Run from Windows Command Prompt

echo =========================================
echo UPLOADING UQ EXPERIMENT CODE TO AMAREL
echo =========================================
echo.

set REMOTE_USER=hpl14
set REMOTE_HOST=amarel.rutgers.edu
set REMOTE_PATH=/scratch/hpl14/uq_capstone

echo Testing SSH connection...
ssh %REMOTE_USER%@%REMOTE_HOST% "echo Connection successful"
if errorlevel 1 (
    echo ERROR: Cannot connect to Amarel
    exit /b 1
)
echo [OK] Connection successful
echo.

echo Uploading Python modules...
scp src\uq_methods.py %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/src/
scp src\swag.py %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/src/
scp src\train_baseline.py %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/src/
scp src\train_mc_dropout.py %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/src/
scp src\train_ensemble_member.py %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/src/
scp src\train_swag.py %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/src/
scp src\evaluate_uq.py %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/src/
echo [OK] Python modules uploaded
echo.

echo Uploading SLURM scripts...
scp scripts\train_baseline.sbatch %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/scripts/
scp scripts\train_mc_dropout.sbatch %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/scripts/
scp scripts\train_ensemble.sbatch %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/scripts/
scp scripts\train_swag.sbatch %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/scripts/
scp scripts\evaluate_uq.sbatch %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/scripts/
scp scripts\run_all_experiments.sh %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/scripts/
scp scripts\monitor_jobs.sh %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/scripts/
echo [OK] SLURM scripts uploaded
echo.

echo Uploading documentation...
scp UQ_EXPERIMENTS_GUIDE.md %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/
scp SWAG_GUIDE.md %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/
scp IMPLEMENTATION_SUMMARY.md %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_PATH%/
echo [OK] Documentation uploaded
echo.

echo Making scripts executable...
ssh %REMOTE_USER%@%REMOTE_HOST% "chmod +x %REMOTE_PATH%/scripts/*.sh %REMOTE_PATH%/scripts/*.sbatch"
echo [OK] Scripts are executable
echo.

echo =========================================
echo UPLOAD COMPLETE!
echo =========================================
echo.
echo Next steps:
echo 1. Open new terminal and SSH: ssh hpl14@amarel.rutgers.edu
echo 2. Navigate: cd /scratch/hpl14/uq_capstone
echo 3. Run: bash scripts/run_all_experiments.sh
echo 4. Monitor: squeue -u hpl14
echo.
echo Estimated time: 6-8 hours total
echo.
pause
