# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

$RootDir = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path

$Config = "RelWithDebInfo"
$TimeoutSec = 60 * 60

$CTestExe = (Join-Path $RootDir "ctest.exe")
$CTestTestFile = (Join-Path $RootDir "CTestTestfile.cmake")

# Extract the build path from CTestTestfile.cmake
$OldBuildDirectoryRegex = (Select-String -Path $CTestTestFile -Pattern "# Build directory: (.*)$").Matches.Groups[1].Value
$OldBuildDirectoryBackslashesRegex = ($OldBuildDirectoryRegex -replace "/", "\\\\")

# Substitutions to point CTest at this directory
$NewBuildDirectory = ($RootDir -replace "\\", "/")
$NewBuildDirectoryBackslashes = ($NewBuildDirectory -replace "/", "\\")

# Rewrite CTestTestfile.cmake
(Get-Content $CTestTestFile) `
    -replace $OldBuildDirectoryRegex, $NewBuildDirectory `
    -replace $OldBuildDirectoryBackslashesRegex, $NewBuildDirectoryBackslashes |
    Out-File -Encoding ascii $CTestTestFile

# Run CTest
Push-Location $RootDir
& $CTestExe --build-config $Config --verbose --timeout $TimeoutSec

$Failed = $false
if (-not $?) {
    $Failed = $true
}
else {
    # Run ONNX model tests
    & "$RootDir\$Config\onnx_test_runner.exe" `
        -j 1 `
        -e qnn `
        -i "backend_type|cpu" `
        "_deps\onnx-src\onnx/backend\test\data\node"
    if (-not $?) {
        $Failed = $true
    }
}

# If it looks like we're running in QDC, copy logs to the directory they'll scan to find them.
if (Test-Path "C:\Temp\TestContent") {
    $QdcLogsDir = "C:\Temp\QDC_logs"
    if (Test-Path $QdcLogsDir) {
        Remove-Item -Recurse -Force -Path $QdcLogsDir
        if (-not $?) {
            throw "Failed to clear old QDC logs dir $QdcLogsDir"
        }
    }

    New-Item -ItemType Directory -Force -Path $QdcLogsDir
    if (-not $?) {
        throw "Failed to create QDC logs dir $QdcLogsDir"
    }

    $LocalLogsDir = (Join-Path $RootDir $Config)
    Write-Host "Copying logs $LocalLogsDir\*.xml --> $QdcLogsDir"
    Copy-Item $LocalLogsDir\*.xml $QdcLogsDir
}

if ($Failed) {
    throw "Tests failed"
}
