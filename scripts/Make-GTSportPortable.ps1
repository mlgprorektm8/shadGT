param(
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '../Build/x64-Clang-Release'),
    [string]$ProfileConfig = (Join-Path $PSScriptRoot '../Build/gt-sport-fixed/user/config.json'),
    # The unpacked win64 release of https://github.com/shadps4-emu/shadps4-qtlauncher.
    [string]$QtLauncherDirectory = (Join-Path $PSScriptRoot '../Build/tools/shadPS4QtLauncher-win64-2026-10-07-4e32b32'),
    [string]$OutputDirectory = '',
    [switch]$FullLogging
)

# Builds a self-contained GT Sport folder: the shadPS4 Qt launcher next to this fork's
# shadps4.exe, set up to use it. Every setting, save, shader cache and log stays in the "user"
# (emulator) and "launcher" folders beside them; neither program uses AppData when those
# folders exist. It copies no game files, firmware modules, fonts, saves, keys or caches: the
# player supplies those.

$ErrorActionPreference = 'Stop'
$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).Path
$executable = Join-Path $BuildDirectory 'shadps4.exe'
if (!(Test-Path -LiteralPath $executable)) {
    throw "Missing $executable; build the Release configuration first."
}
$launcherExe = Join-Path $QtLauncherDirectory 'shadPS4QtLauncher.exe'
if (!(Test-Path -LiteralPath $launcherExe)) {
    throw "Missing $launcherExe; unpack the Qt launcher's win64 release there."
}
$commit = (git -C $PSScriptRoot rev-parse --short HEAD 2>$null)
if (!$commit) { $commit = 'unknown' }
if (!$OutputDirectory) {
    $OutputDirectory = Join-Path $PSScriptRoot "../../shadPS4-portable/GTSport-shadPS4-$(Get-Date -Format 'yyyyMMdd')-$commit"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "$OutputDirectory already exists; choose another -OutputDirectory."
}
foreach ($dir in 'user/sys_modules', 'user/fonts', 'launcher') {
    New-Item -ItemType Directory -Force -Path (Join-Path $OutputDirectory $dir) | Out-Null
}
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
Copy-Item -LiteralPath $executable -Destination $OutputDirectory
Copy-Item -Path (Join-Path $QtLauncherDirectory '*') -Destination $OutputDirectory -Recurse

# The tested settings, with every machine-specific path cleared so the user folder's own
# subfolders (sys_modules, fonts, home) are used.
$config = Get-Content -LiteralPath $ProfileConfig -Raw | ConvertFrom-Json
$config.General.install_dirs = @()
$config.General.addon_install_dir = ''
$config.General.font_dir = ''
$config.General.home_dir = ''
$config.General.sys_modules_dir = ''
$config.General.show_fps_counter = $true
$config.Log.filter = if ($FullLogging) { '*:Info' } else { '*:Warning' }
$config | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'user/config.json') -Encoding UTF8

# The launcher starts the selected emulator with its own folder as the working directory, so
# a path relative to that folder keeps the install movable. No update checks: this fork's
# shadps4.exe is the version to run.
$versions = @(
    [ordered]@{ name = "GT Sport fork $commit"; path = 'shadps4.exe'; date = (Get-Date -Format 'yyyy-MM-dd'); codename = $commit; type = 2 }
)
ConvertTo-Json -InputObject $versions -Depth 3 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'launcher/versions.json') -Encoding ASCII
$launcherSettings = @"
[general_settings]
checkForUpdates=false
showChangeLog=false

[version_manager]
versionSelected=shadps4.exe
checkOnStartup=false
showChangeLog=false
"@
Set-Content -LiteralPath (Join-Path $OutputDirectory 'launcher/qt_ui.ini') -Value $launcherSettings -Encoding ASCII

$readme = @"
GT Sport on shadPS4 (fork build $commit)

Start shadPS4QtLauncher.exe. It is the shadPS4 Qt launcher, set up to run this folder's
shadps4.exe (the fork build tested with GT Sport; keep it selected in the launcher's version
list). Everything both programs write stays in this folder: the emulator's settings
(user\config.json), saves, shader cache and logs in "user", the launcher's settings in
"launcher". Nothing goes to AppData. Copy or move the whole folder to move an installation.

You need to provide, from your own console:
1. GT Sport (CUSA03220, update 1.69 tested) as a dumped game folder with eboot.bin.
2. The PS4 firmware system modules, copied into user\sys_modules (the folder with
   libSceJpegDec.sprx, libSceNgs2.sprx, libSceFont.sprx and the others). GT Sport uses
   libSceJpegDec and libSceJson2, which have no built-in replacement in the emulator.
3. Optionally the system fonts, copied into user\fonts.

The first time: in the launcher's settings, add the folder that contains your GT Sport
folder (CUSA03220) as a game folder. GT Sport then appears in the game list; double-click it
to play.

If the game crashes on start, a third-party overlay (screen recorder, FPS counter, GPU
tuning tool) may be hooking Vulkan; close it and try again.

The first launch compiles shaders as the game needs them, so it stutters until the shader
cache in user\cache has been built up; later launches precompile it with a progress counter
in the window title.

Settings tested for accuracy (do not change unless comparing): readbacks_mode 1 and
readback_linear_images_enabled true in user\config.json.
"@
Set-Content -LiteralPath (Join-Path $OutputDirectory 'README.txt') -Value $readme -Encoding UTF8

Write-Output "Portable folder: $OutputDirectory"
Get-ChildItem -LiteralPath $OutputDirectory | Select-Object Name, Length
