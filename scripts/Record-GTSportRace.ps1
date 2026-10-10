# DIAG-056: records an Nsight Systems capture of a running GT Sport session started with
# Run-GTSportPerformance.ps1 -Nsight <nsys.exe>. Start it once the race is under way.
# -EarlyFencesFlag <file> (the SHADGT_EARLY_FENCES_FLAG the game runs with): records once as
# normal, then again with the EXP-063 measurement switch on, in the same race.
param(
    [Parameter(Mandatory = $true)]
    [string]$Nsight,
    [int]$Seconds = 30,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../Build/gt-sport-fixed/nsight'),
    [string]$EarlyFencesFlag = ''
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
# CPU samples and thread switches need administrator rights; without them only the Vulkan, GPU
# and WDDM traces are recorded.
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).
    IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$cpu = if ($admin) {
    @('--sample=process-tree', '--sampling-frequency=2000', '--cpuctxsw=process-tree')
} else {
    Write-Output 'Not elevated: recording without CPU samples.'
    @('--sample=none', '--cpuctxsw=none')
}

function Record-Capture([string]$Label) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $output = Join-Path (Resolve-Path -LiteralPath $OutputDirectory).Path "race-$stamp-$Label"
    & $Nsight start --session=gtsport @cpu --output=$output --force-overwrite=true
    if ($LASTEXITCODE -ne 0) {
        throw "nsys start failed ($LASTEXITCODE). Is the game running with -Nsight?"
    }
    Write-Output "Recording $Seconds s ($Label) to $output.nsys-rep"
    Start-Sleep -Seconds $Seconds
    & $Nsight stop --session=gtsport
    Write-Output "Saved $output.nsys-rep"
}

if ($EarlyFencesFlag) {
    Remove-Item -LiteralPath $EarlyFencesFlag -ErrorAction SilentlyContinue
    Record-Capture 'normal'
    Set-Content -LiteralPath $EarlyFencesFlag -Value 'on'
    Start-Sleep -Seconds 3
    try {
        Record-Capture 'early-fences'
    } finally {
        Remove-Item -LiteralPath $EarlyFencesFlag -ErrorAction SilentlyContinue
    }
} else {
    Record-Capture 'normal'
}
