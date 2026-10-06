param(
    [int]$DurationSeconds = 20,
    [string]$BuildDirectory = (Join-Path $PSScriptRoot '../Build/x64-Clang-Profile'),
    [string]$OutputDirectory = ''
)

# Records a CPU profile of GT Sport during a race: per-thread CPU use and sampled functions.
# Windows Performance Recorder needs administrator rights, so the script relaunches itself
# elevated once. The game runs through Run-GTSportPerformance.ps1 with its isolated profile.

$ErrorActionPreference = 'Stop'
$principal = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
if (!$principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-NoExit', '-File', "`"$PSCommandPath`"",
        '-DurationSeconds', $DurationSeconds, '-BuildDirectory', "`"$BuildDirectory`"")
    if ($OutputDirectory) {
        $arguments += @('-OutputDirectory', "`"$OutputDirectory`"")
    }
    Start-Process -FilePath powershell.exe -Verb RunAs -ArgumentList $arguments
    Write-Output 'Continuing in the elevated PowerShell window.'
    return
}

$BuildDirectory = (Resolve-Path -LiteralPath $BuildDirectory).Path
$profileDirectory = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../Build/gt-sport-fixed')).Path
$launcher = Join-Path $PSScriptRoot 'Run-GTSportPerformance.ps1'
$xperf = 'C:\Program Files (x86)\Windows Kits\10\Windows Performance Toolkit\xperf.exe'
foreach ($file in @((Join-Path $BuildDirectory 'shadps4.exe'), (Join-Path $BuildDirectory 'shadps4.pdb'),
        $launcher, $xperf)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing profiling prerequisite: $file"
    }
}
$wpr = (Get-Command wpr.exe).Source
if (Get-Process shadps4 -ErrorAction SilentlyContinue) {
    throw 'Close the current emulator before starting a profiling run.'
}
if (!$OutputDirectory) {
    $OutputDirectory = Join-Path $profileDirectory ("profiles/" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class GtSportThreadNames {
    [DllImport("kernel32.dll")] static extern IntPtr OpenThread(uint access, bool inherit, uint id);
    [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
    [DllImport("kernel32.dll")] static extern int GetThreadDescription(IntPtr handle, out IntPtr text);
    [DllImport("kernel32.dll")] static extern IntPtr LocalFree(IntPtr memory);
    public static string Get(int id) {
        IntPtr handle = OpenThread(0x1000, false, (uint)id); // THREAD_QUERY_LIMITED_INFORMATION
        if (handle == IntPtr.Zero) return "";
        try {
            IntPtr text;
            if (GetThreadDescription(handle, out text) < 0 || text == IntPtr.Zero) return "";
            string name = Marshal.PtrToStringUni(text);
            LocalFree(text);
            return name;
        } finally {
            CloseHandle(handle);
        }
    }
}
'@

function Get-ThreadTimes([System.Diagnostics.Process]$process) {
    $process.Refresh()
    $times = @{}
    foreach ($thread in $process.Threads) {
        try {
            $times[$thread.Id] = $thread.TotalProcessorTime.TotalMilliseconds
        } catch {
            # The thread exited between enumeration and the query.
        }
    }
    return $times
}

Write-Output "Profiling build: $BuildDirectory"
Write-Output "Results: $OutputDirectory"
$runner = Start-Process -FilePath powershell.exe -PassThru -ArgumentList @('-NoProfile',
    '-ExecutionPolicy', 'Bypass', '-File', "`"$launcher`"", '-BuildDirectory', "`"$BuildDirectory`"")
$emulator = $null
for ($i = 0; $i -lt 120 -and !$emulator; ++$i) {
    Start-Sleep -Milliseconds 500
    $emulator = Get-Process shadps4 -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (!$emulator) {
    throw 'The emulator did not start; see the performance launcher window.'
}

Read-Host "Start a race and drive normally. Note the FPS counter, then press Enter to record $DurationSeconds seconds"
$before = Get-ThreadTimes $emulator
$clock = [Diagnostics.Stopwatch]::StartNew()
$trace = Join-Path $OutputDirectory 'cpu.etl'
& $wpr -start CPU -filemode | Out-Null
Start-Sleep -Seconds $DurationSeconds
& $wpr -stop $trace | Out-Null
$elapsed = $clock.Elapsed.TotalMilliseconds
$after = Get-ThreadTimes $emulator
Write-Output 'Capture finished. You can keep playing or close the game.'

$threads = foreach ($id in $after.Keys) {
    $start = if ($before.ContainsKey($id)) { $before[$id] } else { 0 }
    [pscustomobject]@{
        ThreadId = $id
        Name = [GtSportThreadNames]::Get($id)
        CpuPercentOfOneCore = [math]::Round(100 * ($after[$id] - $start) / $elapsed, 1)
    }
}
$threads = $threads | Sort-Object CpuPercentOfOneCore -Descending
$threads | Format-Table -AutoSize | Out-String -Width 200 | Set-Content -LiteralPath (Join-Path $OutputDirectory 'threads.txt')
$threads | Select-Object -First 15 | Format-Table -AutoSize

Write-Output 'Resolving sampled functions with the profiling build symbols (this can take a few minutes)...'
$env:_NT_SYMBOL_PATH = $BuildDirectory
& $xperf -i $trace -symbols -quiet -a profile -detail -o (Join-Path $OutputDirectory 'functions.txt') 2>&1 |
    Out-File -Encoding utf8 (Join-Path $OutputDirectory 'xperf-errors.txt')
Write-Output "Done. Share the folder: $OutputDirectory"
