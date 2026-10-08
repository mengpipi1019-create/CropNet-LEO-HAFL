param(
    [ValidateSet('Run','Pause','Status','Check','Plot','DownloadEval','PrepareInputs')][string]$Action = 'Run',
    [string]$Python = '',
    [string]$Output = 'results/joint_fusion/mmst_compare_balanced_20260930_01'
)
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
$compareTranscriptStarted = $false
try {
    if (-not $Python) {
        $compareVenv = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
        if (Test-Path -LiteralPath $compareVenv) { $Python = $compareVenv }
        else { $Python = (Get-Command python -ErrorAction Stop).Source }
    }
    if (-not (Test-Path -LiteralPath $Python)) { throw "Python not found: $Python" }
    if ($Action -in @('Run', 'DownloadEval', 'PrepareInputs')) {
        $compareStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $compareLogDirectory = 'results/joint_fusion'
        New-Item -ItemType Directory -Force -Path $compareLogDirectory | Out-Null
        $compareTranscript = "$compareLogDirectory/mmst_compare_${Action}_${compareStamp}.log"
        Start-Transcript -LiteralPath $compareTranscript | Out-Null
        $compareTranscriptStarted = $true
    }
    if ($Action -eq 'Run') {
        Write-Host 'Centralized / FedAvg / balanced hierarchical async. Seeds 10,20,30; 120 epochs each; nine runs.'
        Write-Host 'Same start command resumes automatically. Keep this window open and connect AC power.'
        Write-Host 'Safe pause: run pause_mmst_compare.cmd in another window, then wait for PAUSED.'
        Write-Host 'All input files and evaluation cache are checked BEFORE long training. Network retries are automatic.'
        Write-Host 'A supervisor resumes recoverable worker failures (up to 3 restarts). Permanent errors stop with a log.'
    }
    $comparePythonAction = switch ($Action) { 'DownloadEval' { 'download-eval' }; 'PrepareInputs' { 'prepare-inputs' }; default { $Action.ToLowerInvariant() } }
    if ($Action -eq 'Run') {
        & $Python -X utf8 -u -m research.joint_fusion.supervise_mmst_compare --out $Output
    } else {
        & $Python -X utf8 -u -m research.joint_fusion.run_mmst_compare --action $comparePythonAction --out $Output
    }
    if ($LASTEXITCODE -ne 0) { throw "Comparison stopped with exit code $LASTEXITCODE. Keep the log. Fix the reported error, then run the same start command to resume." }
} finally {
    if ($compareTranscriptStarted) { Stop-Transcript | Out-Null }
    Pop-Location
}
