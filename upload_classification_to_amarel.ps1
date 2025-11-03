# Upload Classification Code to Amarel
# Automated upload script for all new classification files

# Configuration - UPDATE THESE!
$AMAREL_USERNAME = "YOUR_USERNAME"  # Change this to your Amarel username
$AMAREL_HOST = "amarel.rutgers.edu"
$REMOTE_PATH = "/scratch/$AMAREL_USERNAME/uq_capstone"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }

Write-Info "=============================================="
Write-Info "  Classification Code Upload to Amarel"
Write-Info "=============================================="
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "src\data_utils_classification.py")) {
    Write-Error "ERROR: Not in the correct directory!"
    Write-Error "Please run this script from: C:\Users\lpnhu\Downloads\uq_capstone\"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Info "Current directory: $PWD"
Write-Host ""

# Prompt for username if not set
if ($AMAREL_USERNAME -eq "YOUR_USERNAME") {
    Write-Warning "Please enter your Amarel username:"
    $AMAREL_USERNAME = Read-Host "Username"
    $REMOTE_PATH = "/scratch/$AMAREL_USERNAME/uq_capstone"
}

Write-Info "Upload Configuration:"
Write-Host "  Username: $AMAREL_USERNAME"
Write-Host "  Host: $AMAREL_HOST"
Write-Host "  Remote path: $REMOTE_PATH"
Write-Host ""

# Confirm before proceeding
Write-Warning "This will upload classification files to Amarel."
$confirm = Read-Host "Continue? (y/n)"
if ($confirm -ne "y" -and $confirm -ne "Y") {
    Write-Info "Upload cancelled."
    exit 0
}

Write-Host ""
Write-Info "=============================================="
Write-Info "Step 1: Uploading Source Files"
Write-Info "=============================================="
Write-Host ""

$sourceFiles = @(
    "src/data_utils_classification.py",
    "src/conformal_risk_control.py",
    "src/train_classifier_baseline.py",
    "src/train_classifier_mc_dropout.py",
    "src/train_classifier_ensemble_member.py",
    "src/evaluate_uq_classification.py"
)

$uploadedCount = 0
$failedCount = 0

foreach ($file in $sourceFiles) {
    Write-Info "Uploading: $file"
    $destPath = "${AMAREL_USERNAME}@${AMAREL_HOST}:${REMOTE_PATH}/$file"
    
    try {
        scp $file $destPath
        if ($LASTEXITCODE -eq 0) {
            Write-Success "  ✓ Success"
            $uploadedCount++
        } else {
            Write-Error "  ✗ Failed"
            $failedCount++
        }
    } catch {
        Write-Error "  ✗ Error: $_"
        $failedCount++
    }
    Write-Host ""
}

Write-Host ""
Write-Info "=============================================="
Write-Info "Step 2: Uploading SLURM Scripts"
Write-Info "=============================================="
Write-Host ""

$scriptFiles = @(
    "scripts/train_classifier_baseline.sbatch",
    "scripts/train_classifier_mc_dropout.sbatch",
    "scripts/train_classifier_ensemble.sbatch",
    "scripts/evaluate_classification.sbatch",
    "scripts/run_all_classification_experiments.sh"
)

foreach ($file in $scriptFiles) {
    Write-Info "Uploading: $file"
    $destPath = "${AMAREL_USERNAME}@${AMAREL_HOST}:${REMOTE_PATH}/$file"
    
    try {
        scp $file $destPath
        if ($LASTEXITCODE -eq 0) {
            Write-Success "  ✓ Success"
            $uploadedCount++
        } else {
            Write-Error "  ✗ Failed"
            $failedCount++
        }
    } catch {
        Write-Error "  ✗ Error: $_"
        $failedCount++
    }
    Write-Host ""
}

Write-Host ""
Write-Info "=============================================="
Write-Info "Step 3: Uploading Documentation"
Write-Info "=============================================="
Write-Host ""

$docFiles = @(
    "docs/CLASSIFICATION_QUICK_START.md",
    "docs/CLASSIFICATION_SETUP_GUIDE.md",
    "docs/CLASSIFICATION_IMPLEMENTATION_SUMMARY.md",
    "docs/DEPLOYMENT_CHECKLIST.md"
)

foreach ($file in $docFiles) {
    Write-Info "Uploading: $file"
    $destPath = "${AMAREL_USERNAME}@${AMAREL_HOST}:${REMOTE_PATH}/$file"
    
    try {
        scp $file $destPath
        if ($LASTEXITCODE -eq 0) {
            Write-Success "  ✓ Success"
            $uploadedCount++
        } else {
            Write-Error "  ✗ Failed"
            $failedCount++
        }
    } catch {
        Write-Error "  ✗ Error: $_"
        $failedCount++
    }
    Write-Host ""
}

Write-Host ""
Write-Info "=============================================="
Write-Info "Step 4: Making Scripts Executable"
Write-Info "=============================================="
Write-Host ""

Write-Info "Setting executable permissions on Amarel..."
$sshCommand = "cd $REMOTE_PATH; chmod +x scripts/run_all_classification_experiments.sh; chmod +x scripts/*.sbatch"
$sshDest = "${AMAREL_USERNAME}@${AMAREL_HOST}"

try {
    ssh $sshDest $sshCommand
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✓ Scripts are now executable"
    } else {
        Write-Warning "⚠ Could not set permissions (you may need to do this manually)"
    }
} catch {
    Write-Warning "⚠ Could not set permissions: $_"
}

Write-Host ""
Write-Info "=============================================="
Write-Info "Upload Summary"
Write-Info "=============================================="
Write-Host ""
Write-Success "✓ Successfully uploaded: $uploadedCount files"
if ($failedCount -gt 0) {
    Write-Error "✗ Failed uploads: $failedCount files"
}

Write-Host ""
Write-Info "=============================================="
Write-Info "Next Steps"
Write-Info "=============================================="
Write-Host ""
Write-Host "1. Download the Chest X-Ray dataset:"
Write-Host "   - Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
Write-Host "   - Download and extract locally"
Write-Host "   - Upload to Amarel:"
Write-Host ""
Write-Host "   scp -r chest_xray ${AMAREL_USERNAME}@${AMAREL_HOST}:${REMOTE_PATH}/data/"
Write-Host ""
Write-Host "2. SSH to Amarel and verify:"
Write-Host "   ssh ${AMAREL_USERNAME}@${AMAREL_HOST}"
Write-Host "   cd ${REMOTE_PATH}"
Write-Host "   conda activate uq_capstone"
Write-Host "   python -c 'from src.data_utils_classification import get_classification_loaders; print(\"✓ Imports work!\")'"
Write-Host ""
Write-Host "3. Run experiments:"
Write-Host "   bash scripts/run_all_classification_experiments.sh"
Write-Host ""
Write-Host "4. Monitor jobs:"
Write-Host "   squeue -u $AMAREL_USERNAME"
Write-Host ""
Write-Info "=============================================="
Write-Success "Upload Complete!"
Write-Info "=============================================="
Write-Host ""

Read-Host "Press Enter to exit"
