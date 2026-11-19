param(
    [ValidateSet('Full','Smoke')][string]$Mode='Full',
    [string]$Project='F:\aki\mlp_factor',
    [string]$Python
)
$ErrorActionPreference='Stop'
if (-not $Python) { $Python=Join-Path $Project '.venv312\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$base=Join-Path $Project 'data\experiments\dcn_set_s42_public_v1'
$output=Join-Path $base $Mode.ToLower()
$snapshot=Join-Path $base ('source_'+$Mode.ToLower())
New-Item -ItemType Directory -Path $output,$snapshot -Force | Out-Null
$lock=[IO.File]::Open((Join-Path $output 'run.lock'),'OpenOrCreate','ReadWrite','None')
try {
    foreach ($item in (Get-ChildItem -LiteralPath $PSScriptRoot -File -Recurse)) {
        $relative=$item.FullName.Substring($PSScriptRoot.Length+1)
        if ($relative -match '__pycache__') { continue }
        $dst=Join-Path $snapshot $relative
        if (Test-Path -LiteralPath $dst) {
            if ((Get-FileHash -LiteralPath $dst).Hash -ne (Get-FileHash -LiteralPath $item.FullName).Hash) {
                throw "Saved experiment code differs: $relative"
            }
        } else {
            New-Item -ItemType Directory -Path (Split-Path $dst) -Force | Out-Null
            $tmp=$dst+'.partial'
            Copy-Item -LiteralPath $item.FullName -Destination $tmp -Force
            if ((Get-FileHash -LiteralPath $tmp).Hash -ne (Get-FileHash -LiteralPath $item.FullName).Hash) { throw "Incomplete copy: $relative" }
            [IO.File]::Move($tmp,$dst)
        }
    }
    $log=Join-Path $output ('console_'+(Get-Date -Format 'yyyyMMdd_HHmmss_fff')+'.log')
    Write-Host "Output: $output"
    Write-Host "Training log: $log"
    & $Python -B -u (Join-Path $snapshot 'check.py')
    if ($LASTEXITCODE -ne 0) { throw 'Hybrid architecture check failed' }
    $arguments=@('-B','-u',(Join-Path $snapshot 'study.py'),'--project-root',$Project,'--output',$output)
    if ($Mode -eq 'Smoke') { $arguments+='--smoke' }
    & $Python @arguments | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) { throw 'Training failed. Read the traceback above.' }
    if ($Mode -eq 'Full') {
        & $Python -B -u (Join-Path $snapshot 'compare.py') --project-root $Project --output $output | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { throw 'Historical comparison failed. Training checkpoints remain available.' }
        Write-Host "Comparison: $output\report\comparison_22m\summary.csv"
    }
} finally { $lock.Dispose() }
