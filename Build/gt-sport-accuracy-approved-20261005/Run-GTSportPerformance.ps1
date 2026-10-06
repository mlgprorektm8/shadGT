param(
    [string]$GamePath = 'E:/Console Games/PS4 Games/CUSA03220/eboot.bin',
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '../Build/x64-Clang-Release'),
    [string]$ProfileDirectory = (Join-Path $PSScriptRoot '../Build/gt-sport-fixed'),
    [switch]$FullLogging,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$executable = Join-Path $BuildDirectory 'shadps4.exe'
$configPath = Join-Path $ProfileDirectory 'user/config.json'
foreach ($file in @($executable, $configPath, $GamePath)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing performance-run prerequisite: $file"
    }
}
if (Get-Process shadps4 -ErrorAction SilentlyContinue) {
    throw 'Close the current emulator before starting another run.'
}
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if (!$config.General.redzone_patches) {
    throw 'This GT Sport profile must have Windows red-zone patching enabled.'
}
Write-Output "Executable: $((Resolve-Path -LiteralPath $executable).Path)"
Write-Output "Profile: $((Resolve-Path -LiteralPath $ProfileDirectory).Path)"
if ($CheckOnly) {
    return
}

# Override diagnostics only. Preserve resolution, readbacks, buffer copying, and saves.
$overrides = @{
    Debug = @{ debug_dump = $false; shader_collect = $false }
    GPU = @{ dump_shaders = $false }
    Log = @{ filter = $(if ($FullLogging) { $config.Log.filter } else { '*:Warning' }); sync = $false }
    Vulkan = @{
        pipeline_cache_enabled = $true
        renderdoc_enabled = $false
        vkvalidation_enabled = $false
        vkvalidation_core_enabled = $false
        vkvalidation_sync_enabled = $false
        vkvalidation_gpu_enabled = $false
        vkcrash_diagnostic_enabled = $false
        vkguest_markers = $false
        vkhost_markers = $false
    }
}
$previous = @{}
foreach ($section in $overrides.Keys) {
    $previous[$section] = @{}
    foreach ($key in $overrides[$section].Keys) {
        $previous[$section][$key] = $config.$section.$key
        $config.$section.$key = $overrides[$section][$key]
    }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $ProfileDirectory 'user/log/shad_log.txt'
if (Test-Path -LiteralPath $log) {
    Copy-Item -LiteralPath $log -Destination (Join-Path $ProfileDirectory "log-before-performance-$stamp.txt")
}
$variables = @('VK_LOADER_LAYERS_DISABLE', 'VK_LOADER_LAYERS_ENABLE', 'VK_LAYER_PATH')
$environment = @{}
foreach ($name in $variables) {
    $environment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $config | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $configPath -Encoding UTF8
    $env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
    $env:VK_LOADER_LAYERS_ENABLE = $null
    $env:VK_LAYER_PATH = $null
    $arguments = '"{0}" --show-fps' -f $GamePath
    $process = Start-Process -FilePath $executable -ArgumentList $arguments `
        -WorkingDirectory $ProfileDirectory -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $ProfileDirectory "stdout-performance-$stamp.txt") `
        -RedirectStandardError (Join-Path $ProfileDirectory "stderr-performance-$stamp.txt") -PassThru
    Write-Output "Emulator PID: $($process.Id)"
    $process.WaitForExit()
    Write-Output "Emulator exit code: $($process.ExitCode)"
} finally {
    foreach ($name in $variables) {
        [Environment]::SetEnvironmentVariable($name, $environment[$name], 'Process')
    }
    # Reload to retain settings changed by the player while the game was running.
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    foreach ($section in $previous.Keys) {
        foreach ($key in $previous[$section].Keys) {
            $config.$section.$key = $previous[$section][$key]
        }
    }
    $config | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $configPath -Encoding UTF8
}
