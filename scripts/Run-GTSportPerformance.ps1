param(
    [string]$GamePath = 'E:/Console Games/PS4 Games/CUSA03220/eboot.bin',
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '../Build/x64-Clang-Release'),
    [string]$ProfileDirectory = (Join-Path $PSScriptRoot '../Build/gt-sport-fixed'),
    [switch]$FullLogging,
    [ValidateSet(-1, 0, 1, 2)]
    [int]$ReadbacksMode = -1,
    [switch]$CrashDiagnostics,
    [switch]$SyncLog,
    [ValidateSet(-1, 0, 1)]
    [int]$CopyGpuBuffers = -1,
    [ValidateSet(-1, 0, 1)]
    [int]$ReadbackLinearImages = -1,
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
# Test-only readback override (0 disabled, 1 relaxed, 2 precise); -1 keeps the profile value.
if ($ReadbacksMode -ge 0) {
    $overrides.GPU.readbacks_mode = $ReadbacksMode
    Write-Output "Readbacks mode for this run: $ReadbacksMode"
}
# Test-only command-buffer copy override; -1 keeps the profile value.
if ($CopyGpuBuffers -ge 0) {
    $overrides.GPU.copy_gpu_buffers = [bool]$CopyGpuBuffers
    Write-Output "copy_gpu_buffers for this run: $([bool]$CopyGpuBuffers)"
}
# Write the log synchronously for crash investigation (slower).
if ($SyncLog) {
    $overrides.Log.sync = $true
    Write-Output 'Synchronous logging for this run'
}
# Test-only linear-image readback override; -1 keeps the profile value.
if ($ReadbackLinearImages -ge 0) {
    $overrides.GPU.readback_linear_images_enabled = [bool]$ReadbackLinearImages
    Write-Output "readback_linear_images_enabled for this run: $([bool]$ReadbackLinearImages)"
}
# Diagnostic runs only: the Vulkan crash-diagnostic layer reports the in-flight work on device loss.
$sdkLayers = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../Build/tools/VulkanSDK/Bin')).Path
$crashDumps = Join-Path $ProfileDirectory 'crash-dumps'
if ($CrashDiagnostics) {
    $overrides.Vulkan.vkcrash_diagnostic_enabled = $true
    New-Item -ItemType Directory -Force -Path $crashDumps | Out-Null
    Write-Output "Crash diagnostics enabled; reports go to $crashDumps"
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
$variables = @('VK_LOADER_LAYERS_DISABLE', 'VK_LOADER_LAYERS_ENABLE', 'VK_LAYER_PATH', 'CDL_OUTPUT_PATH')
$environment = @{}
foreach ($name in $variables) {
    $environment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $config | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $configPath -Encoding UTF8
    $env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
    $env:VK_LOADER_LAYERS_ENABLE = $null
    $env:VK_LAYER_PATH = $null
    if ($CrashDiagnostics) {
        $env:VK_LAYER_PATH = $sdkLayers
        $env:CDL_OUTPUT_PATH = $crashDumps
    }
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
