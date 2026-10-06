param(
    [string]$Destination = 'D:/Development/shadPS4-regression-baselines/GT-Sport-20261005-8021b5f2',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$commit = '8021b5f27b920a4f0c2357ee435089a1eec5c2a8'
$tag = 'gt-sport-graphics-baseline-20261005'
if (Get-Process shadps4 -ErrorAction SilentlyContinue) {
    throw 'Close the emulator before copying its profile.'
}
if ((Test-Path -LiteralPath $Destination) -and !$Resume) {
    throw "Reference destination already exists: $Destination"
}
if ($Resume) {
    foreach ($file in @('source.bundle', 'source-8021b5f2.zip', 'README.md',
                        'binaries/Debug/shadps4.exe', 'binaries/Release/shadps4.exe',
                        'profile-and-investigations/test-reference-focused.log')) {
        if (!(Test-Path -LiteralPath (Join-Path $Destination $file))) {
            throw "Cannot resume an incomplete copy: missing $file"
        }
    }
    foreach ($file in @('manifest.json', 'dependency-source.zip')) {
        if (Test-Path -LiteralPath (Join-Path $Destination $file)) {
            throw "Resume expects the dependency archive not yet created: $file"
        }
    }
}
foreach ($type in @('Debug', 'Release')) {
    if (!(Test-Path -LiteralPath (Join-Path $PSScriptRoot "x64-Clang-$type/shadps4.exe"))) {
        throw "Missing $type executable."
    }
}
if (!$Resume) { New-Item -ItemType Directory -Path $Destination | Out-Null }
Push-Location -LiteralPath $repo
try {
    if (!$Resume) {
    # Git writes successful bundle verification to stderr on Windows PowerShell.
    $ErrorActionPreference = 'Continue'
    $existing = git tag --list $tag
    if ($LASTEXITCODE -ne 0) { throw 'Could not query reference tag.' }
    if ($existing) {
        $tagCommit = git rev-list -n 1 $tag
        if ($LASTEXITCODE -ne 0) { throw 'Could not resolve reference tag.' }
        if ($tagCommit -ne $commit) { throw 'Existing reference tag points to another commit.' }
    } else {
        git tag -a $tag $commit -m 'Player-confirmed GT Sport graphics baseline: strip gone, white race icons; lower track HUD still cyan.'
        if ($LASTEXITCODE -ne 0) { throw 'Could not create reference tag.' }
    }
    git bundle create (Join-Path $Destination 'source.bundle') "refs/tags/$tag"
    if ($LASTEXITCODE -ne 0) { throw 'Git bundle creation failed.' }
    git bundle verify (Join-Path $Destination 'source.bundle')
    if ($LASTEXITCODE -ne 0) { throw 'Git bundle verification failed.' }
    git archive --format=zip "--output=$(Join-Path $Destination 'source-8021b5f2.zip')" $commit
    if ($LASTEXITCODE -ne 0) { throw 'Source archive creation failed.' }
    git diff --binary "--output=$(Join-Path $Destination 'local-notes.patch')" $commit
    if ($LASTEXITCODE -ne 0) { throw 'Local patch export failed.' }
    $submodules = git submodule status
    if ($LASTEXITCODE -ne 0) { throw 'Could not record dependency revisions.' }
    $ErrorActionPreference = 'Stop'
    $submodules | Set-Content -LiteralPath (Join-Path $Destination 'submodules.txt') -Encoding ASCII
    Write-Output 'Source checkpoint and verified Git bundle saved.'

    foreach ($type in @('Debug', 'Release')) {
        $build = Join-Path $PSScriptRoot "x64-Clang-$type"
        $binary = Join-Path $Destination "binaries/$type"
        New-Item -ItemType Directory -Path $binary | Out-Null
        Get-ChildItem -LiteralPath $build -File | Where-Object {
            $_.Name -eq 'shadps4.exe' -or $_.Name -eq 'shadps4.pdb' -or
            $_.Extension -eq '.dll' -or $_.Name -eq 'qt.conf'
        } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $binary }
        foreach ($directory in @('platforms', 'plugins', 'translations')) {
            $path = Join-Path $build $directory
            if (Test-Path -LiteralPath $path) { Copy-Item -LiteralPath $path -Destination $binary -Recurse }
        }
        $metadata = Join-Path $Destination "build-settings/$type"
        New-Item -ItemType Directory -Path $metadata | Out-Null
        foreach ($file in @('CMakeCache.txt', 'compile_commands.json', 'build.ninja')) {
            Copy-Item -LiteralPath (Join-Path $build $file) -Destination $metadata
        }
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'gt-sport-fixed') `
        -Destination (Join-Path $Destination 'profile-and-investigations') -Recurse
    Write-Output 'Binaries, symbols, closed profile, captures, and investigations saved.'

    $support = Join-Path $Destination 'launchers-and-notes'
    New-Item -ItemType Directory -Path $support | Out-Null
    foreach ($file in @('Run-GT-Sport.ps1', 'Run-GT-Sport-Capture.ps1', 'Save-GT-Sport-Reference.ps1', 'GT_SPORT_STATUS.md')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $file) -Destination $support
    }
    foreach ($file in @('scripts/Run-GTSportPerformance.ps1', 'scripts/Check-GTSportGraphics.ps1', 'GT_SPORT_BASELINE.md', 'CHANGES.MD')) {
        Copy-Item -LiteralPath (Join-Path $repo $file) -Destination $support
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'GT_SPORT_REFERENCE_README.md') `
        -Destination (Join-Path $Destination 'README.md')
    $tools = Join-Path $Destination 'capture-tools'
    New-Item -ItemType Directory -Path $tools | Out-Null
    foreach ($directory in @('capture-inspect', 'renderdoc-replay-api-1.46', 'renderdoc-1.46')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot "tools/$directory") -Destination $tools -Recurse
    }
    } else {
        $heads = git bundle list-heads (Join-Path $Destination 'source.bundle') "refs/tags/$tag"
        if ($LASTEXITCODE -ne 0) { throw 'Could not inspect saved bundle.' }
        $tagObject = git rev-parse "refs/tags/$tag"
        if ($LASTEXITCODE -ne 0 -or $heads -ne "$tagObject refs/tags/$tag") {
            throw 'Saved bundle does not match the local baseline tag.'
        }
        $tagCommit = git rev-list -n 1 $tag
        if ($LASTEXITCODE -ne 0 -or $tagCommit -ne $commit) { throw 'Baseline commit mismatch.' }
        Write-Output 'Resuming dependency archive and hash records for the verified checkpoint.'
    }
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $Destination 'launchers-and-notes/Save-GT-Sport-Reference.ps1')
    Add-Type -AssemblyName System.IO.Compression
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $externalRoot = (Resolve-Path (Join-Path $repo 'externals')).Path
    $archive = [System.IO.Compression.ZipFile]::Open(
        (Join-Path $Destination 'dependency-source.zip'), [System.IO.Compression.ZipArchiveMode]::Create)
    try {
        Get-ChildItem -LiteralPath $externalRoot -Recurse -File -Force | Where-Object {
            $_.Name -ne '.git' -and $_.FullName -notmatch '[\\/]\.git[\\/]'
        } | ForEach-Object {
            $relative = $_.FullName.Substring($externalRoot.Length + 1).Replace('\', '/')
            [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
                $archive, $_.FullName, $relative, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
        }
    } finally {
        $archive.Dispose()
    }
    Write-Output 'Dependency source and replay tools saved. Computing SHA-256 records.'
    $root = (Resolve-Path -LiteralPath $Destination).Path
    $files = @(Get-ChildItem -LiteralPath $root -Recurse -File | ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($root.Length + 1).Replace('\', '/')
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    })
    $manifest = [ordered]@{
        created_utc = [DateTime]::UtcNow.ToString('o')
        source_commit = $commit
        source_tag = $tag
        confirmed_build = 'Debug'
        optimized_build = 'Release; Clang /O2 /Ob2 /DNDEBUG; Tracy disabled'
        player_confirmed = @('Frozen top strip gone', 'In-race icons white', 'Map backgrounds correct')
        remaining = @('Bottom in-race track HUD cyan', 'Release speed/visual parity needs player check', 'Long-term stability unproven')
        external_dependencies = @('Game/update files at existing E: path', 'Dumped system modules at D:/Emulators/shadPS4/sys_modules')
        focused_tests = '69 selected regressions; see profile-and-investigations/test-reference-focused.log'
        files = $files
    }
    $manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $Destination 'manifest.json') -Encoding UTF8
    Write-Output "Reference saved: $Destination ($($files.Count) hashed files)"
} finally {
    Pop-Location
}
