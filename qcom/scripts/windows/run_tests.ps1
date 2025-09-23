# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(HelpMessage = "Path to onnx/models models")]
    $OnnxModelsRoot = $null
)

$RootDir = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path

if (-not $OnnxModelsRoot) {
    $OnnxModelsRoot = (Join-Path $RootDir (Join-Path "model_tests" "onnx_models"))
}

$Config = "Release"
$TimeoutSec = 60 * 60

$CTestExe = (Join-Path $RootDir "ctest.exe")

# Single-config generators like Ninja put runners in the parent directory
# of the rest of the ONNX Runtime build.
if (Test-Path (Join-Path $RootDir "onnx_test_runner.exe")) {
    $OnnxTestRunnerExe = (Join-Path $RootDir "onnx_test_runner.exe")
} else {
    $OnnxTestRunnerExe = (Join-Path (Join-Path $RootDir $Config) "onnx_test_runner.exe")
}

$CTestTestFile = (Join-Path $RootDir "CTestTestfile.cmake")

if (-not (Test-Path $CTestTestFile)) {
    throw "$CTestTestFile not found"
}

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

# Figure out if HTP is available
if ((Get-CimInstance Win32_operatingsystem).OSArchitecture -eq "ARM 64-bit Processor") {
    $QdqBackend = "htp"
} else {
    $QdqBackend = "cpu"
}

$Failed = $false

# Run CTest
Push-Location $RootDir
Write-Host "--=-=-=- Running unit tests -=--=-=-"
& $CTestExe --build-config $Config --verbose --timeout $TimeoutSec

if (-not $?) {
    Write-Host "Unit tests failed. Will exit with error after running model tests."
    $Failed = $true
}

# AISW-152430 - Do not run Python tests on Windows ARM for now due to frankenstein build process for ARM Python wheel
if ((Get-CimInstance Win32_Processor).Architecture -ne 12) {  # Architecture code 12 corresponds to ARM64
    Write-Host "--=-=-=- Running Python tests -=--=-=-"
    $PythonTestFilesPath = (Join-Path $RootDir "python_test_files.txt")

    if (Test-Path $PythonTestFilesPath) {
        $PythonTestFiles = Get-Content $PythonTestFilesPath

        foreach ($PythonFile in $PythonTestFiles) {
            $PythonFile = $PythonFile.Trim()
            if ($PythonFile -and (Test-Path $PythonFile)) {

                # TODO - AISW-139802 - Tests in the following files hang on Windows - skip them for now
                if ($PythonFile -like "*onnxruntime_test_python_backend.py" -or
                        $PythonFile -like "*onnxruntime_test_python_global_threadpool.py") {
                    Write-Host "Skipping $PythonFile - contains a test that hangs on Windows"
                    continue
                }

                Write-Host "Running $PythonFile..."
                & python $PythonFile
                if (-not $?) {
                    Write-Error "Python test $PythonFile failed."
                    $Failed = $true
                }
            } else {
                Write-Warning "Failed to find $PythonFile - may be OK on platforms which do not support Python."
            }
        }
    } else {
        Write-Error "Python test files list not found at $PythonTestFilesPath"
        $Failed = $true
    }

    if (Test-Path "quantization" -PathType Container) {
        Write-Host "Running quantization tests..."
        & python -m unittest discover -s quantization
    } else {
        Write-Warning "Failed to find directory 'quantization' - may be OK on platforms which do not support Python."
    }
    if (-not $?) {
        Write-Error "Quantization tests failed."
        $Failed = $true
    }
} else {
    Write-Warning "Host is Windows ARM - skipping Python testing for now."
}

Write-Host "--=-=-=- Running ONNX model tests -=--=-=-"
& $OnnxTestRunnerExe `
    -j 1 `
    -e qnn `
    -i "backend_type|cpu" `
    "_deps\onnx-src\onnx\backend\test\data\node"
if (-not $?) {
    $Failed = $true
}

Write-Host "-=-=-=- Running onnx/models float32 tests -=-=-=-"
Push-Location $OnnxModelsRoot
if (-not $?) {
    throw "Could not cd to $OnnxModelsRoot"
}

& $OnnxTestRunnerExe `
    -j 1 `
    -e qnn `
    -i "backend_type|cpu" `
    "testdata\float32"
if (-not $?) {
    $Failed = $true
}

Write-Host "-=-=-=- Running onnx/models qdq tests -=-=-=-"
& $OnnxTestRunnerExe `
    -j 1 `
    -e qnn `
    -i "backend_type|$QdqBackend" `
    "testdata\qdq"
if (-not $?) {
    $Failed = $true
}

if ($QdqBackend -ne "cpu") {
    Write-Host "-=-=-=- Running onnx/models qdq tests with context cache enabled -=-=-=-"
    # Scrub old context caches
    Get-ChildItem -Path "testdata\qdq-with-context-cache" -Recurse -Filter "*_ctx.onnx" | Remove-Item -Force
    & $OnnxTestRunnerExe `
        -j 1 `
        -e qnn `
        -f -i "backend_type|$QdqBackend" `
        "testdata\qdq-with-context-cache"
    if (-not $?) {
        $Failed = $true
    }
} else {
    Write-Host "Not running onnx/models qdq tests with context cache enabled on CPU backend."
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

    New-Item -ItemType Directory -Force -Path $QdcLogsDir | Out-Null
    if (-not $?) {
        throw "Failed to create QDC logs dir $QdcLogsDir"
    }

    # This location depends on whether we built with Ninja or Visual Studio.
    $LocalLogsDir = (Join-Path $RootDir $Config)
    if (-not (Test-Path $LocalLogsDir)) {
        $LocalLogsDir = $RootDir
    }
    Write-Host "Copying logs $LocalLogsDir\*.xml --> $QdcLogsDir"
    Copy-Item $LocalLogsDir\*.xml $QdcLogsDir
}

if ($Failed) {
    throw "Tests failed"
}
