param(
    [ValidateSet('Release', 'Debug')]
    [string]$BuildType = 'Debug'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$profile = Join-Path $PSScriptRoot 'gt-sport-fixed'
$renderdoc = Join-Path $PSScriptRoot 'tools/renderdoc-1.46/RenderDoc_1.46_64'
$executable = Join-Path $PSScriptRoot "x64-Clang-$BuildType/shadps4.exe"
$game = 'E:/Console Games/PS4 Games/CUSA03220/eboot.bin'

if (Get-Process shadps4 -ErrorAction SilentlyContinue) {
    throw 'Close the current emulator before starting the capture run.'
}
foreach ($file in @($executable, $game, (Join-Path $renderdoc 'renderdoccmd.exe'))) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing capture prerequisite: $file"
    }
}

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $profile 'user/log/shad_log.txt'
if (Test-Path -LiteralPath $log) {
    Copy-Item -LiteralPath $log -Destination (Join-Path $profile "log-before-capture-$stamp.txt")
}
$variables = @('VK_LOADER_LAYERS_DISABLE', 'VK_LOADER_LAYERS_ENABLE', 'VK_LAYER_PATH')
$saved = @{}
foreach ($name in $variables) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
    $env:VK_LOADER_LAYERS_ENABLE = 'VK_LAYER_RENDERDOC_Capture'
    $env:VK_LAYER_PATH = $renderdoc
    $arguments = 'capture --working-dir "{0}" --capture-file "{1}" "{2}" "{3}"' -f
        $profile, (Join-Path $profile 'CUSA03220'), $executable, $game
    $launcher = Start-Process -FilePath (Join-Path $renderdoc 'renderdoccmd.exe') `
        -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $profile "stdout-capture-$stamp.txt") `
        -RedirectStandardError (Join-Path $profile "stderr-capture-$stamp.txt") -PassThru
    # Wait for injection, not the entire launched game process tree.
    $launcher.WaitForExit()
    Write-Output "Capture launcher exit code: $($launcher.ExitCode)"
    Get-Content (Join-Path $profile "stdout-capture-$stamp.txt") -Tail 15
} finally {
    foreach ($name in $variables) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
