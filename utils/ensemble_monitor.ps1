# Ensemble monitor script
 = 1..4
 = Join-Path  '..\runs\classification\ensemble'
 = 'C:\Users\lpnhu\Downloads\uq_capstone\runs\classification\ensemble\ensemble_monitor.log'
600 = 600
while (True) {
     = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
     = @()
    foreach (5 in ) {
         = Join-Path  "member_5_train.log"
         = ''
         = ''
        if (Test-Path ) {
             = Get-Content  -ErrorAction SilentlyContinue
             = ( | Select-String -Pattern 'Epoch\s+\d+/\d+' | Select-Object -Last 1).Line
             = ( | Select-String -Pattern '^\s*Train:' | Select-Object -Last 1).Line
        }
        if (-not ) {  = 'no epoch yet' }
        if (-not ) {  = 'no metrics yet' }
         += "	member_5		"
    }
     | Out-File -FilePath  -Append -Encoding utf8
    Start-Sleep -Seconds 600
}
