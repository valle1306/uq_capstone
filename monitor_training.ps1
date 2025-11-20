# Monitor SWAG Training Jobs on Amarel
# Run this script periodically to check job status

Write-Host "========================================="
Write-Host "SWAG Training Job Monitor"
Write-Host "========================================="
Write-Host ""

# Job IDs
$BASELINE_JOB = "48316072"
$SWAG_JOB = "48316073"

Write-Host "Checking job status..."
ssh hpl14@amarel.rutgers.edu "squeue -u hpl14"

Write-Host ""
Write-Host "========================================="
Write-Host "Baseline SGD Log (last 20 lines):"
Write-Host "========================================="
ssh hpl14@amarel.rutgers.edu "tail -20 /scratch/hpl14/uq_capstone/logs/baseline_sgd_${BASELINE_JOB}.out 2>/dev/null || echo 'Log not yet available'"

Write-Host ""
Write-Host "========================================="
Write-Host "SWAG Log (last 20 lines):"
Write-Host "========================================="
ssh hpl14@amarel.rutgers.edu "tail -20 /scratch/hpl14/uq_capstone/logs/swag_proper_${SWAG_JOB}.out 2>/dev/null || echo 'Log not yet available'"

Write-Host ""
Write-Host "========================================="
Write-Host "To download results when complete:"
Write-Host "========================================="
Write-Host "scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/baseline_sgd results/"
Write-Host "scp -r hpl14@amarel.rutgers.edu:/scratch/hpl14/uq_capstone/runs/classification/swag_proper results/"
Write-Host ""
