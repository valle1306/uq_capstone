# Upload Corrected SWAG Implementation to Amarel
# Run this script from your local machine

Write-Host "========================================="
Write-Host "Uploading Corrected SWAG Implementation"
Write-Host "========================================="

$AMAREL_USER = "hpl14"
$AMAREL_HOST = "amarel.rutgers.edu"
$REMOTE_DIR = "/scratch/hpl14/uq_capstone"

# Files to upload
$FILES = @(
    "src/train_baseline_sgd.py",
    "src/train_swag_proper.py",
    "scripts/amarel/train_baseline_sgd.slurm",
    "scripts/amarel/train_swag_proper.slurm",
    "docs/SWAG_CORRECTED.md"
)

Write-Host "`nUploading files to Amarel..."
foreach ($file in $FILES) {
    Write-Host "  - $file"
    scp $file "${AMAREL_USER}@${AMAREL_HOST}:${REMOTE_DIR}/$file"
}

Write-Host "`n========================================="
Write-Host "Upload complete!"
Write-Host "========================================="
Write-Host "`nNext steps on Amarel:"
Write-Host "1. ssh ${AMAREL_USER}@${AMAREL_HOST}"
Write-Host "2. cd ${REMOTE_DIR}"
Write-Host "3. sbatch scripts/amarel/train_baseline_sgd.slurm"
Write-Host "4. sbatch scripts/amarel/train_swag_proper.slurm"
Write-Host "5. squeue -u ${AMAREL_USER}  # Check job status"
Write-Host "========================================="
