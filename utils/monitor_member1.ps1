# Monitor script for member_1 log
 = 'C:\Users\lpnhu\Downloads\uq_capstone\runs\classification\ensemble\member_1_train.log'
C:\Users\lpnhu\Downloads\uq_capstone\runs\classification\ensemble\member_1_monitor.log = 'C:\Users\lpnhu\Downloads\uq_capstone\runs\classification\ensemble\member_1_monitor.log'
while (True) {
     = @()
    if (Test-Path ) {
         = Get-Content  -ErrorAction SilentlyContinue
    }
     = ( | Select-String -Pattern 'Epoch\s+\d+/\d+' | Select-Object -Last 1).Line
     = ( | Select-String -Pattern '^\s*Train:' | Select-Object -Last 1).Line
     = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    if ( -or ) {
        "	Epoch: 	Metrics: " | Out-File -FilePath C:\Users\lpnhu\Downloads\uq_capstone\runs\classification\ensemble\member_1_monitor.log -Append -Encoding utf8
    } else {
        "	No data yet." | Out-File -FilePath C:\Users\lpnhu\Downloads\uq_capstone\runs\classification\ensemble\member_1_monitor.log -Append -Encoding utf8
    }
    Start-Sleep -Seconds 300
}
