param(
    [string]$LogPath = (Join-Path $PSScriptRoot '../Build/gt-sport-fixed/user/log/shad_log.txt')
)

$ErrorActionPreference = 'Stop'
if (!(Test-Path -LiteralPath $LogPath)) {
    throw "Game log not found: $LogPath"
}
$gtLog = Get-Content -LiteralPath $LogPath
$gtPatterns = [ordered]@{
    DeviceLoss = 'VK_ERROR_DEVICE_LOST|eErrorDeviceLost|device lost'
    DescriptorValidation = 'VUID-.*(07988|07990|descriptorType)'
    AttachmentAndLayoutValidation = 'VUID-.*(08914|imageView|imageLayout)'
    ImageCopyValidation = 'VUID-vkCmdCopyImage-'
    BufferCopyValidation = 'VUID-VkBufferCopy-|VUID-vkCmdCopyBuffer-'
    SamplerValidation = 'VUID-VkSamplerCreateInfo-'
    ShaderModuleValidation = 'VUID-VkShaderModuleCreateInfo-'
    ShaderInterface = 'VUID-RuntimeSpirv-OpEntryPoint'
    Synchronization = 'SYNC-HAZARD-'
    ValidationUnavailable = 'Requested layer VK_LAYER_KHRONOS_validation is not available'
    MissingShaderResources = 'Unexpected instruction for offset computation|Failed to compute offset for SRT walker|Sharp source was not flatenned|Unsupported.*(offset|Phi|GetAttribute)|Unable.*(sharp|resource)'
    InvalidGuestDescriptors = 'Rejecting invalid [TS]#'
    ImageAllocationFailure = 'Failed allocating image|ErrorInitializationFailed'
    UnsupportedLayerExport = 'RenderTargetIndex.*tessellation'
    MalformedCommands = 'Invalid PM4|malformed.*packet|Invalid.*packet'
    SwizzledAlphaEmulation = 'Using dual-source swizzled source-alpha blend emulation'
    FarDepthPrecisionWorkaround = 'Applying read-only far-depth precision workaround'
}
Write-Output "GT Sport log: $((Resolve-Path -LiteralPath $LogPath).Path)"
Write-Output "Log modified: $((Get-Item -LiteralPath $LogPath).LastWriteTime.ToString('o'))"
Write-Output 'Counts are matching log lines; duplicate suppression and run length affect comparisons.'
Write-Output "Validation layer loaded: $([bool]($gtLog -match 'Enabled instance layers:.*VK_LAYER_KHRONOS_validation'))"
foreach ($gtPattern in $gtPatterns.GetEnumerator()) {
    $gtMatches = @($gtLog | Select-String -Pattern $gtPattern.Value)
    [pscustomobject]@{ Check = $gtPattern.Key; LoggedLines = $gtMatches.Count }
}
Write-Output 'Visual checks for this build: camera lighting; frozen top strip; white track map/icons; map backgrounds; race loading.'
Write-Output 'Compare the same race, camera, weather, settings, and driving duration. A clean log does not establish correct graphics.'
