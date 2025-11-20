# Helper script for syncing repo and results with Amarel
# WARNING: This script prints commands but does not auto-run destructive operations.

$remoteUser = "hpl14@amarel.rutgers.edu"
$remoteBase = "/scratch/hpl14/uq_capstone"
$localBase = "C:\Users\lpnhu\Downloads\uq_capstone"

Write-Host "--- Pull latest code on Amarel (run on Amarel) ---"
Write-Host "ssh hpl14@amarel.rutgers.edu 'cd /scratch/hpl14/uq_capstone && git pull origin main'"

Write-Host "--- Pull latest code locally (run here) ---"
Write-Host "git pull origin main"

Write-Host "--- Push local changes (run here) ---"
Write-Host "git add -A && git commit -m 'Update: two-stage SWAG scripts and conformal runner' && git push origin main"

Write-Host "--- Download results (example) ---"
Write-Host "scp $remoteUser:`$remoteBase/runs/classification/swag_adam/* $localBase/runs/classification/swag_adam/"

Write-Host "Note: This helper prints commands to execute. Run them manually to avoid accidental remote changes."
