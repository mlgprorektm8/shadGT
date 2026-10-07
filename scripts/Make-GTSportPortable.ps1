param(
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '../Build/x64-Clang-Release'),
    [string]$ProfileConfig = (Join-Path $PSScriptRoot '../Build/gt-sport-fixed/user/config.json'),
    [string]$OutputDirectory = '',
    [switch]$FullLogging
)

# Builds a self-contained GT Sport folder: shadps4.exe with every setting, save, shader cache
# and log in the "user" folder beside it (this fork never uses AppData on Windows). It copies
# no game files, firmware modules, fonts, saves, keys or caches: the player supplies those.

$ErrorActionPreference = 'Stop'
$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).Path
$executable = Join-Path $BuildDirectory 'shadps4.exe'
if (!(Test-Path -LiteralPath $executable)) {
    throw "Missing $executable; build the Release configuration first."
}
$commit = (git -C $PSScriptRoot rev-parse --short HEAD 2>$null)
if (!$commit) { $commit = 'unknown' }
if (!$OutputDirectory) {
    $OutputDirectory = Join-Path $PSScriptRoot "../../shadPS4-portable/GTSport-shadPS4-$(Get-Date -Format 'yyyyMMdd')-$commit"
}
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "$OutputDirectory already exists; choose another -OutputDirectory."
}
New-Item -ItemType Directory -Force -Path (Join-Path $OutputDirectory 'user/sys_modules') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutputDirectory 'user/fonts') | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
Copy-Item -LiteralPath $executable -Destination $OutputDirectory

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

$launcher = @'
@echo off
setlocal
rem Starts GT Sport with this folder's shadPS4. Drag eboot.bin (or the game folder) onto this
rem file, or put the path to eboot.bin in game_path.txt beside it.
cd /d "%~dp0"
set "GAME=%~1"
if "%GAME%"=="" if exist "game_path.txt" set /p GAME=<"game_path.txt"
if "%GAME%"=="" (
    echo Path to GT Sport's eboot.bin, for example E:\PS4\CUSA03220\eboot.bin:
    set /p GAME=
)
if exist "%GAME%\eboot.bin" set "GAME=%GAME%\eboot.bin"
if not exist "%GAME%" (
    echo Not found: %GAME%
    pause
    exit /b 1
)
> "game_path.txt" echo %GAME%
rem Third-party Vulkan overlays (recorders, tuning tools) can crash the emulator.
set "VK_LOADER_LAYERS_DISABLE=~implicit~"
"%~dp0shadps4.exe" "%GAME%"
endlocal
'@
Set-Content -LiteralPath (Join-Path $OutputDirectory 'Launch GT Sport.bat') -Value $launcher -Encoding ASCII

$readme = @"
GT Sport on shadPS4 (fork build $commit)

Everything this build writes stays in the "user" folder next to shadps4.exe: settings
(user\config.json), saves, the shader cache, logs (user\log\shad_log.txt) and screenshots.
Nothing goes to AppData. Copy or move the whole folder to move an installation.

You need to provide, from your own console:
1. GT Sport (CUSA03220, update 1.69 tested) as a dumped game folder with eboot.bin.
2. The PS4 firmware system modules, copied into user\sys_modules (the folder with
   libSceJpegDec.sprx, libSceNgs2.sprx, libSceFont.sprx and the others). GT Sport uses
   libSceJpegDec and libSceJson2, which have no built-in replacement in the emulator.
3. Optionally the system fonts, copied into user\fonts.

To play: drag eboot.bin (or the game folder) onto "Launch GT Sport.bat". It remembers the
path in game_path.txt, so next time a double-click is enough.

The first launch compiles shaders as the game needs them, so it stutters until the shader
cache in user\cache has been built up; later launches precompile it with a progress counter
in the window title.

Settings tested for accuracy (do not change unless comparing): readbacks_mode 1 and
readback_linear_images_enabled true in user\config.json.
"@
Set-Content -LiteralPath (Join-Path $OutputDirectory 'README.txt') -Value $readme -Encoding UTF8

Write-Output "Portable folder: $OutputDirectory"
Get-ChildItem -LiteralPath $OutputDirectory -Recurse | Select-Object FullName, Length
