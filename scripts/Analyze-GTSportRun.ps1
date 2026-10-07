# Summarizes the 2-second performance reports in a shadPS4 log (PERF-DIAG-001/002/006/007).
param(
    [Parameter(Mandatory = $true)][string]$LogPath,
    [int]$MinFps = 5
)

$windows = @()
$current = $null
foreach ($line in [System.IO.File]::ReadLines($LogPath)) {
    if ($line -match 'Flips in ([\d.]+) s: (\d+) flips') {
        $current = [ordered]@{ Fps = [double]$Matches[2] / [double]$Matches[1]; GpuBusy = $null; Threads = @{}; WaitMs = 0.0; Faults = 0 }
        $windows += $current
        continue
    }
    if ($null -eq $current) { continue }
    if ($line -match 'Perf monitor [\d.]+ s: GPU busy (\d+)%, (\d+) submits; CPU process (\d+)% of \d+%;(.*)$') {
        $current.GpuBusy = [double]$Matches[1]
        $current.Submits = [int]$Matches[2]
        $current.Process = [double]$Matches[3]
        foreach ($m in [regex]::Matches($Matches[4], '(\S+) (\d+)%')) {
            $current.Threads[$m.Groups[1].Value] = [double]$m.Groups[2].Value
        }
    } elseif ($line -match 'GPU waits in ([\d.]+) s \(count/total\):(.*)$') {
        foreach ($m in [regex]::Matches($Matches[2], '=(\d+)/([\d.]+)ms')) {
            $current.WaitMs += [double]$m.Groups[2].Value
        }
    } elseif ($line -match 'Readback pages in [\d.]+ s: (\d+) CPU faults') {
        $current.Faults = [int]$Matches[1]
    } elseif ($line -match 'Frontend work in [\d.]+ s: (\d+) PM4 packets, (\d+) draws, (\d+) dispatches, (\d+) image lookups, (\d+) buffer binds \((\d+) streamed\), (\d+) uploads \((\d+) KB\), (\d+) protection calls(?: \(\d+ KB\), (\d+) shaders and (\d+) pipelines compiled)?') {
        $current.Work = @{ Packets = [double]$Matches[1]; Draws = [double]$Matches[2]; Dispatches = [double]$Matches[3]
            ImageLookups = [double]$Matches[4]; BufferBinds = [double]$Matches[5]; Uploads = [double]$Matches[7]
            UploadKB = [double]$Matches[8]; Protects = [double]$Matches[9]
            ShadersCompiled = [double]$Matches[10]; PipelinesCompiled = [double]$Matches[11] }
        $current.Phases = [ordered]@{}
        if ($line -match ';(.*)$') {
            foreach ($m in [regex]::Matches($Matches[1], ' ([a-z/-]+)=([\d.]+)ms')) {
                $current.Phases[$m.Groups[1].Value] = [double]$m.Groups[2].Value
            }
        }
    } elseif ($line -match 'Command processor waits in [\d.]+ s \(count/total\):([^;]*);') {
        $current.FrontendWaits = @{}
        foreach ($m in [regex]::Matches($Matches[1], ' ([a-z]+ [A-Z_ ]+?|[a-z]+ VO label)=(\d+)/([\d.]+)ms')) {
            $current.FrontendWaits[$m.Groups[1].Value] = [double]$m.Groups[3].Value
        }
    }
}

$play = @($windows | Where-Object { $_.Fps -ge $MinFps })
if ($play.Count -eq 0) {
    Write-Output "No 2-second windows at $MinFps FPS or more in $LogPath"
    return
}
function Stat($values) {
    $sorted = @($values | Sort-Object)
    if ($sorted.Count -eq 0) { return 'n/a' }
    $avg = ($sorted | Measure-Object -Average).Average
    $p10 = $sorted[[int][Math]::Floor(($sorted.Count - 1) * 0.1)]
    $p90 = $sorted[[int][Math]::Floor(($sorted.Count - 1) * 0.9)]
    return ('avg {0:N1}, 10%-90% {1:N1}-{2:N1}' -f $avg, $p10, $p90)
}

Write-Output ("Run summary: {0} ({1} windows of 2 s at {2}+ FPS, {3:N0} s)" -f (Split-Path $LogPath -Leaf), $play.Count, $MinFps, ($play.Count * 2))
Write-Output ("  FPS:              " + (Stat ($play | ForEach-Object { $_.Fps })))
Write-Output ("  GPU busy %:       " + (Stat ($play | Where-Object { $null -ne $_.GpuBusy } | ForEach-Object { $_.GpuBusy })))
Write-Output ("  Process CPU %:    " + (Stat ($play | Where-Object { $null -ne $_.Process } | ForEach-Object { $_.Process })))
Write-Output ("  Submits / 2 s:    " + (Stat ($play | Where-Object { $null -ne $_.Submits } | ForEach-Object { $_.Submits })))
Write-Output ("  GPU wait ms / 2 s:" + (Stat ($play | ForEach-Object { $_.WaitMs })))
Write-Output ("  CPU faults / 2 s: " + (Stat ($play | ForEach-Object { $_.Faults })))

# Frontend work per presented frame.
$withWork = @($play | Where-Object { $_.Work -and $_.Fps -gt 0 })
foreach ($key in 'Packets', 'Draws', 'Dispatches', 'ImageLookups', 'BufferBinds', 'Uploads', 'UploadKB', 'Protects') {
    if ($withWork.Count -gt 0) {
        Write-Output ('  {0,-17} per frame: {1}' -f $key, (Stat ($withWork | ForEach-Object { $_.Work[$key] / ($_.Fps * 2) })))
    }
}
foreach ($key in 'ShadersCompiled', 'PipelinesCompiled') {
    if ($withWork.Count -gt 0) {
        Write-Output ('  {0,-17} total in run: {1}' -f $key, (($withWork | ForEach-Object { $_.Work[$key] } | Measure-Object -Sum).Sum))
    }
}
$withPhases = @($withWork | Where-Object { $_.Phases -and $_.Phases.Count -gt 0 })
if ($withPhases.Count -gt 0) {
    Write-Output '  Command thread ms per frame by draw step:'
    foreach ($key in $withPhases[0].Phases.Keys) {
        Write-Output ('    {0,-16} {1}' -f $key, (Stat ($withPhases | ForEach-Object { $_.Phases[$key] / ($_.Fps * 2) })))
    }
}
$waitKeys = @($play | Where-Object { $_.FrontendWaits } | ForEach-Object { $_.FrontendWaits.Keys } | Sort-Object -Unique)
foreach ($key in $waitKeys) {
    Write-Output ('  Waits {0,-18} ms / 2 s: {1}' -f $key, (Stat ($play | Where-Object { $_.FrontendWaits } | ForEach-Object { [double]$_.FrontendWaits[$key] })))
}

# Busiest threads, averaged over the windows they appeared in among the top 10.
$totals = @{}
$counts = @{}
foreach ($w in $play) {
    foreach ($name in $w.Threads.Keys) {
        $totals[$name] = $totals[$name] + $w.Threads[$name]
        $counts[$name] = $counts[$name] + 1
    }
}
if ($totals.Count -gt 0) {
    Write-Output '  Busiest threads (% of one core, averaged over all play windows):'
    $totals.Keys | ForEach-Object {
        [pscustomobject]@{ Name = $_; Avg = $totals[$_] / $play.Count }
    } | Sort-Object Avg -Descending | Select-Object -First 10 | ForEach-Object {
        Write-Output ('    {0,-40} {1,5:N0}%' -f $_.Name, $_.Avg)
    }
}
