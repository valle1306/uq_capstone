@echo off
REM Script to upload BRATS data to Amarel cluster from Windows
REM 
REM Prerequisites:
REM   - Install WinSCP: https://winscp.net/
REM   - Or install Windows Subsystem for Linux (WSL) and use the .sh script
REM
REM Usage:
REM   1. Edit the YOUR_NETID variable below
REM   2. Run: upload_to_amarel.bat

REM ========== CONFIGURATION ==========
SET YOUR_NETID=YOUR_NETID_HERE
SET AMAREL_HOST=amarel.rutgers.edu

SET LOCAL_DATA_DIR=data\brats
SET LOCAL_PROJECT_ROOT=%CD%

SET AMAREL_USER=%YOUR_NETID%
SET AMAREL_SCRATCH=/scratch/%AMAREL_USER%
SET AMAREL_PROJECT=/scratch/%AMAREL_USER%/uq_capstone
REM ===================================

echo ========================================
echo   Uploading UQ Capstone to Amarel
echo ========================================
echo Target: %AMAREL_USER%@%AMAREL_HOST%
echo Remote: %AMAREL_PROJECT%
echo ========================================
echo.

REM Check if NetID was changed
if "%YOUR_NETID%"=="YOUR_NETID_HERE" (
    echo ERROR: Please edit this script and set YOUR_NETID!
    pause
    exit /b 1
)

REM Check if data directory exists
if not exist "%LOCAL_DATA_DIR%" (
    echo ERROR: Data directory not found: %LOCAL_DATA_DIR%
    echo Please run prepare_small_brats_subset.py first!
    pause
    exit /b 1
)

echo OPTION 1: Using WinSCP
echo ========================================
echo 1. Download and install WinSCP: https://winscp.net/
echo 2. Open WinSCP and connect to:
echo    Host: %AMAREL_HOST%
echo    User: %AMAREL_USER%
echo    Password: Your Rutgers NetID password
echo.
echo 3. On Amarel (right side), navigate to: /scratch/%AMAREL_USER%/
echo 4. Create folder: uq_capstone
echo 5. Upload these folders from your local machine:
echo    - %LOCAL_DATA_DIR% --^> /scratch/%AMAREL_USER%/uq_capstone/data/brats/
echo    - scripts/ --^> /scratch/%AMAREL_USER%/uq_capstone/scripts/
echo    - src/ --^> /scratch/%AMAREL_USER%/uq_capstone/src/
echo    - envs/ --^> /scratch/%AMAREL_USER%/uq_capstone/envs/
echo    - requirements.txt --^> /scratch/%AMAREL_USER%/uq_capstone/
echo.
echo ========================================
echo.
echo OPTION 2: Using Windows PowerShell (if you have SSH/SCP)
echo ========================================
echo Run these PowerShell commands:
echo.
echo # Create remote directories
echo ssh %AMAREL_USER%@%AMAREL_HOST% "mkdir -p %AMAREL_PROJECT%/{data/brats,scripts,src,envs,notebooks,runs}"
echo.
echo # Upload files
echo scp -r %LOCAL_DATA_DIR% %AMAREL_USER%@%AMAREL_HOST%:%AMAREL_PROJECT%/data/
echo scp scripts/*.py scripts/*.sbatch %AMAREL_USER%@%AMAREL_HOST%:%AMAREL_PROJECT%/scripts/
echo scp src/*.py %AMAREL_USER%@%AMAREL_HOST%:%AMAREL_PROJECT%/src/
echo scp envs/*.yml requirements.txt %AMAREL_USER%@%AMAREL_HOST%:%AMAREL_PROJECT%/
echo.
echo ========================================
echo.
echo OPTION 3: Using WSL (Windows Subsystem for Linux)
echo ========================================
echo 1. Open WSL terminal
echo 2. Navigate to: cd /mnt/c/Users/lpnhu/Downloads/uq_capstone
echo 3. Run: bash scripts/upload_to_amarel.sh
echo.
echo ========================================
echo.
echo After uploading, SSH to Amarel and run:
echo   ssh %AMAREL_USER%@%AMAREL_HOST%
echo   cd %AMAREL_PROJECT%
echo   module load conda
echo   conda env create -f envs/conda_env.yml
echo   conda activate uq_capstone
echo   python scripts/validate_brats_data.py --data_root data/brats
echo.

pause
