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
    [switch]$CheckOnly,
    # Experimental branch: PERF ids to switch off for an A/B run, e.g. '14,15'.
    [string]$DisablePerf = '',
    # DIAG-056: run under NVIDIA Nsight Systems (session 'gtsport', nothing recorded until
    # Record-GTSportRace.ps1 starts a capture). Path to nsys.exe.
    [string]$Nsight = '',
    # PERF-075: frames paced evenly at this rate (0: as the game flips). GT Sport's races vary
    # between 25 and 40 FPS, which shows as judder; 30 is steady.
    [int]$FpsLock = 30
)

$ErrorActionPreference = 'Stop'
$executable = Join-Path $BuildDirectory 'shadGT.exe'
$configPath = Join-Path $ProfileDirectory 'user/config.json'
foreach ($file in @($executable, $configPath, $GamePath)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing performance-run prerequisite: $file"
    }
}
if (Get-Process shadGT -ErrorAction SilentlyContinue) {
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
$variables = @('VK_LOADER_LAYERS_DISABLE', 'VK_LOADER_LAYERS_ENABLE', 'VK_LAYER_PATH', 'CDL_OUTPUT_PATH',
    'SHADGT_DISABLE_PERF', 'SHADGT_FPS_LOCK', 'SHADGT_DRAW_PIPE', 'SHADGT_PREWARM')
$environment = @{}
foreach ($name in $variables) {
    $environment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $config | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $configPath -Encoding UTF8
    $env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
    $env:VK_LOADER_LAYERS_ENABLE = $null
    $env:VK_LAYER_PATH = $null
    $env:SHADGT_DISABLE_PERF = $DisablePerf
    $env:SHADGT_FPS_LOCK = "$FpsLock"
    # The draw pipe (PERF-031) and background pipeline builds (PERF-038) are the tested default;
    # a value already set in the environment (for example 0) is kept.
    if (-not $env:SHADGT_DRAW_PIPE) { $env:SHADGT_DRAW_PIPE = '1' }
    if (-not $env:SHADGT_PREWARM) { $env:SHADGT_PREWARM = '1' }
    Write-Output "Frame-rate lock: $(if ($FpsLock -gt 0) { "$FpsLock FPS" } else { 'off' }); draw pipe $env:SHADGT_DRAW_PIPE, prewarm $env:SHADGT_PREWARM"
    if ($DisablePerf) {
        Write-Output "Disabled for this run: PERF $DisablePerf"
    }
    if ($CrashDiagnostics) {
        $env:VK_LAYER_PATH = $sdkLayers
        $env:CDL_OUTPUT_PATH = $crashDumps
        # The default 30 s "no GPU progress" watchdog fires during GT Sport's startup loading and
        # records an empty report; record real device losses instead, with the failing shaders.
        foreach ($prefix in 'CDL_', 'VK_LUNARG_CRASH_DIAGNOSTIC_') {
            Set-Item "env:${prefix}OUTPUT_PATH" $crashDumps
            Set-Item "env:${prefix}TRIGGER_WATCHDOG_TIMEOUT" 'false'
            # The layer kept firing at 30 s with only the switch above; push the timeout out too.
            Set-Item "env:${prefix}WATCHDOG_TIMEOUT_MS" '86400000'
            Set-Item "env:${prefix}DUMP_SHADERS" 'on_crash'
            Set-Item "env:${prefix}INSTRUMENT_ALL_COMMANDS" 'true'
        }
    }
    $arguments = '"{0}" --show-fps' -f $GamePath
    $launcher = $executable
    if ($Nsight) {
        if (!(Test-Path -LiteralPath $Nsight -PathType Leaf)) {
            throw "nsys.exe not found: $Nsight"
        }
        # Vulkan API calls, GPU work per submission and WDDM queue activity. The capture itself
        # (and CPU sampling, which needs an elevated PowerShell) starts and stops with
        # Record-GTSportRace.ps1.
        $arguments = ('launch --session-new=gtsport --trace=vulkan,nvtx,wddm ' +
            '--vulkan-gpu-workload=batch --wait=primary "{0}" {1}') -f `
            (Resolve-Path -LiteralPath $executable).Path, $arguments
        $launcher = $Nsight
        Write-Output "Running under Nsight Systems (session gtsport)"
    }
    $process = Start-Process -FilePath $launcher -ArgumentList $arguments `
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
# Keep every run's log with the build's commit, and print its performance summary.
if (Test-Path -LiteralPath $log) {
    $runs = Join-Path $ProfileDirectory 'runs'
    New-Item -ItemType Directory -Force -Path $runs | Out-Null
    $commit = (git -C $PSScriptRoot rev-parse --short HEAD 2>$null)
    if (-not $commit) { $commit = 'unknown' }
    $runLog = Join-Path $runs "$stamp-$commit.log"
    Copy-Item -LiteralPath $log -Destination $runLog
    $summary = & (Join-Path $PSScriptRoot 'Analyze-GTSportRun.ps1') -LogPath $runLog
    $summary | Set-Content -LiteralPath "$runLog.summary.txt" -Encoding UTF8
    $summary
}
