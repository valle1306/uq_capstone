# Helper: list and unstaged unnecessary markdown files
# This script prints git commands to unstage files you might want to remove from the repo
# It does NOT run git commands automatically.

$filesToConsider = @(
    'AMAREL_COMMANDS_REFERENCE.md',
    'AMAREL_READY_50EPOCHS.md',
    'AMAREL_WORKFLOW_50EPOCHS.md',
    'COMPLETE_WORKFLOW_SUMMARY.md',
    'IMPORT_FIX_GUIDE.md'
)

Write-Host "Files suggested for review (do NOT auto-remove):"
foreach ($f in $filesToConsider) { Write-Host " - $f" }

Write-Host "\nTo preview changes, run:" 
Write-Host "git status --porcelain"

Write-Host "\nTo unstage a file but keep it locally, run:" 
Write-Host "git reset HEAD -- <file>"

Write-Host "To remove from repo and delete locally, run:" 
Write-Host "git rm --cached <file> && git commit -m 'Remove noisy docs'"

Write-Host "\nReview the list above and run commands manually to avoid accidental data loss."
