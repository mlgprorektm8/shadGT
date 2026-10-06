param(
    [string]$GamePath = 'E:/Console Games/PS4 Games/CUSA03220/eboot.bin',
    [ValidateSet('Release', 'Debug')]
    [string]$BuildType = 'Release',
    [switch]$Diagnostic,
    [switch]$QuickDiagnostic,
    [switch]$PreciseReadbacks
)

$ErrorActionPreference = 'Stop'
if ($Diagnostic -and $QuickDiagnostic) {
    throw 'Choose either -Diagnostic or -QuickDiagnostic.'
}
$gtDiagnosticRun = $Diagnostic -or $QuickDiagnostic
$gtExecutable = Join-Path $PSScriptRoot "x64-Clang-$BuildType/shadps4.exe"
$gtProfile = Join-Path $PSScriptRoot 'gt-sport-fixed'
if (!(Test-Path -LiteralPath $gtExecutable) -or !(Test-Path -LiteralPath $GamePath)) {
    throw 'The built emulator or game executable is missing.'
}
Write-Output "Using the isolated GT Sport profile at $gtProfile"
Write-Output "Using the Clang $BuildType executable at $gtExecutable"
$gtConfigPath = Join-Path $gtProfile 'user/config.json'
$gtConfig = Get-Content -LiteralPath $gtConfigPath -Raw | ConvertFrom-Json
$gtPreviousReadbacks = $gtConfig.GPU.readbacks_mode
$gtPreviousDumpShaders = $gtConfig.GPU.dump_shaders
$gtPreviousPipelineCache = $gtConfig.Vulkan.pipeline_cache_enabled
$gtPreviousValidation = $gtConfig.Vulkan.vkvalidation_enabled
$gtPreviousCoreValidation = $gtConfig.Vulkan.vkvalidation_core_enabled
$gtPreviousSyncValidation = $gtConfig.Vulkan.vkvalidation_sync_enabled
if ($PreciseReadbacks) {
    $gtConfig.GPU.readbacks_mode = 2
}
$gtConfig.GPU.dump_shaders = [bool]$Diagnostic
$gtConfig.Vulkan.pipeline_cache_enabled = $true
$gtConfig.Vulkan.vkvalidation_enabled = [bool]$gtDiagnosticRun
$gtConfig.Vulkan.vkvalidation_core_enabled = [bool]$gtDiagnosticRun
$gtConfig.Vulkan.vkvalidation_sync_enabled = [bool]$Diagnostic
$gtConfig.Vulkan.vkvalidation_gpu_enabled = $false
$gtConfig.Vulkan.vkcrash_diagnostic_enabled = $false
$gtConfig | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $gtConfigPath -Encoding UTF8
$gtPreviousLayerFilter = $env:VK_LOADER_LAYERS_DISABLE
$gtPreviousLayerPath = $env:VK_LAYER_PATH
# Match the test environment without changing the system's overlay settings.
$env:VK_LOADER_LAYERS_DISABLE = '~implicit~'
if ($gtDiagnosticRun) {
    $env:VK_LAYER_PATH = Join-Path $PSScriptRoot 'tools/VulkanSDK/Bin'
}
Push-Location -LiteralPath $gtProfile
try {
    & $gtExecutable $GamePath
} finally {
    Pop-Location
    $env:VK_LOADER_LAYERS_DISABLE = $gtPreviousLayerFilter
    $env:VK_LAYER_PATH = $gtPreviousLayerPath
    if ($gtDiagnosticRun -or $PreciseReadbacks) {
        # Reload after shutdown to preserve any settings the game/emulator saved.
        $gtConfig = Get-Content -LiteralPath $gtConfigPath -Raw | ConvertFrom-Json
        $gtConfig.GPU.readbacks_mode = $gtPreviousReadbacks
        if ($gtDiagnosticRun) {
            $gtConfig.GPU.dump_shaders = $gtPreviousDumpShaders
            $gtConfig.Vulkan.pipeline_cache_enabled = $gtPreviousPipelineCache
            $gtConfig.Vulkan.vkvalidation_enabled = $gtPreviousValidation
            $gtConfig.Vulkan.vkvalidation_core_enabled = $gtPreviousCoreValidation
            $gtConfig.Vulkan.vkvalidation_sync_enabled = $gtPreviousSyncValidation
        }
        $gtConfig | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $gtConfigPath -Encoding UTF8
    }
}
