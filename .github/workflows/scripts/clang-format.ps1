# SPDX-FileCopyrightText: 2023 Citra Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

$ErrorActionPreference = 'Stop'
if ($env:GITHUB_EVENT_NAME -eq 'pull_request') {
    $files = @(git diff --name-only --diff-filter=ACMRTUXB $env:COMMIT_RANGE -- src)
} else {
    $files = @(git ls-files -- src)
}
if ($LASTEXITCODE -ne 0) { throw 'Unable to list source files.' }
$failed = $false
foreach ($file in $files) {
    if ($file -match '\.(cpp|h)$') {
        & clang-format --dry-run --Werror $file
        if ($LASTEXITCODE -ne 0) { $failed = $true }
    }
}
if ($failed) { throw 'C++ formatting checks failed.' }
