# DIAG-056: records an Nsight Systems capture of a running GT Sport session started with
# Run-GTSportPerformance.ps1 -Nsight <nsys.exe>. Start it once the race is under way.
param(
    [Parameter(Mandatory = $true)]
    [string]$Nsight,
    [int]$Seconds = 30,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../Build/gt-sport-fixed/nsight')
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$output = Join-Path (Resolve-Path -LiteralPath $OutputDirectory).Path "race-$stamp"
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
& $Nsight start --session=gtsport @cpu --output=$output --force-overwrite=true
if ($LASTEXITCODE -ne 0) {
    throw "nsys start failed ($LASTEXITCODE). Is the game running with -Nsight?"
}
Write-Output "Recording $Seconds s to $output.nsys-rep"
Start-Sleep -Seconds $Seconds
& $Nsight stop --session=gtsport
Write-Output "Saved $output.nsys-rep"
