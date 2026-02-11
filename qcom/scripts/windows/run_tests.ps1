# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

param(
    [Parameter(HelpMessage = "The build config")]
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$Config = "Release",
    [Parameter(HelpMessage = "Path to onnx/models models")]
    $OnnxModelsRoot = $null
)

$RootDir = (Resolve-Path -Path "$(Split-Path -Parent $MyInvocation.MyCommand.Definition)").Path
$RepoRoot= (Resolve-Path -Path ("$RootDir\..\..\.."))

if (-not $OnnxModelsRoot) {
    $OnnxModelsRoot = (Join-Path $RootDir (Join-Path "model_tests" "onnx_models"))
}

$TimeoutSec = 60 * 60

$CTestExe = (Join-Path $RootDir "ctest.exe")

# Single-config generators like Ninja put runners in the parent directory
# of the rest of the ONNX Runtime build.
if (Test-Path (Join-Path $RootDir "onnx_test_runner.exe")) {
    $OnnxTestRunnerExe = (Join-Path $RootDir "onnx_test_runner.exe")
    $OnnxEpTestRunnerExe = (Join-Path $RootDir "onnxruntime_plugin_ep_onnx_test.exe")
} else {
    $OnnxTestRunnerExe = (Join-Path (Join-Path $RootDir $Config) "onnx_test_runner.exe")
    $OnnxEpTestRunnerExe = (Join-Path (Join-Path $RootDir $Config) "onnxruntime_plugin_ep_onnx_test.exe")
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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# CTest
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

Push-Location $RootDir
Write-Host "--=-=-=- Running unit tests -=--=-=-"
& $CTestExe --build-config $Config --verbose --timeout $TimeoutSec

if (-not $?) {
    Write-Host "Unit tests failed. Will exit with error after running model tests."
    $Failed = $true
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Python tests
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# AISW-152430 - Do not run Python tests on Windows ARM for now due to frankenstein build process for ARM Python wheel
## TODO: Will need to rewrite the python test by transforming non-ABI to ABI path

# if ((Get-CimInstance Win32_Processor).Architecture -ne 12) {  # Architecture code 12 corresponds to ARM64
#     Write-Host "--=-=-=- Running Python tests -=--=-=-"
#     $PythonTestFilesPath = (Join-Path $RootDir "python_test_files.txt")

#     if (Test-Path $PythonTestFilesPath) {
#         $PythonTestFiles = Get-Content $PythonTestFilesPath

#         foreach ($PythonFile in $PythonTestFiles) {
#             $PythonFile = $PythonFile.Trim()
#             if ($PythonFile -and (Test-Path $PythonFile)) {

#                 # TODO - AISW-139802 - Tests in the following files hang on Windows - skip them for now
#                 if ($PythonFile -like "*onnxruntime_test_python_backend.py" -or
#                         $PythonFile -like "*onnxruntime_test_python_global_threadpool.py") {
#                     Write-Host "Skipping $PythonFile - contains a test that hangs on Windows"
#                     continue
#                 }

#                 Write-Host "Running $PythonFile..."
#                 & python $PythonFile
#                 if (-not $?) {
#                     Write-Error "Python test $PythonFile failed."
#                     $Failed = $true
#                 }
#             } else {
#                 Write-Warning "Failed to find $PythonFile - may be OK on platforms which do not support Python."
#             }
#         }
#     } else {
#         Write-Error "Python test files list not found at $PythonTestFilesPath"
#         $Failed = $true
#     }

#     if (Test-Path "quantization" -PathType Container) {
#         Write-Host "Running quantization tests..."
#         & python -m unittest discover -s quantization
#     } else {
#         Write-Warning "Failed to find directory 'quantization' - may be OK on platforms which do not support Python."
#     }
#     if (-not $?) {
#         Write-Error "Quantization tests failed."
#         $Failed = $true
#     }
# } else {
#     Write-Warning "Host is Windows ARM - skipping Python testing for now."
# }

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Model tests
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

$TestModelsViaEpPlugin = {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("cpu", "gpu", "htp")]
        [string]$Backend,
        [Parameter(Mandatory = $true)]
        [string]$Suite,
        [string]$TestPath = "testdata/$Suite"
    )

    Write-Host "--=-=-=- Running ONNX model $Suite tests with the ABI-stable EP plugin -=--=-=-"
    & $OnnxEpTestRunnerExe `
        -j 1 `
        -e qnn `
        --plugin_ep_libs "qnn|onnxruntime_providers_qnn.dll" `
        --plugin_eps qnn `
        -i "backend_type|$Backend" `
        $TestPath | Write-Host
    if (-not $?) {
        return $true
    }
    return $false
}

$TestModelsViaLegacy = {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("cpu", "gpu", "htp")]
        [string]$Backend,
        [Parameter(Mandatory = $true)]
        [string]$Suite,
        [string]$TestPath = "testdata/$Suite"
    )

    Write-Host "--=-=-=- Running ONNX model $Suite tests with the legacy ProviderBridge EP -=--=-=-"
    & $OnnxTestRunnerExe `
        -j 1 `
        -e qnn `
        -i "backend_type|$Backend" `
        $TestPath | Write-Host
    if (-not $?) {
        return $true
    }
    return $false
}


Write-Host "--=-=-=- Running ONNX model tests -=--=-=-"

Push-Location $OnnxModelsRoot
if (-not $?) {
    throw "Could not cd to $OnnxModelsRoot"
}

foreach ($RunModelTests in ($TestModelsViaEpPlugin)) {
    $Failed = $Failed -or (& $RunModelTests -Backend cpu -Suite node -TestPath "$RepoRoot\cmake\external\onnx\onnx\backend\test\data\node")
    $Failed = $Failed -or (& $RunModelTests -Backend cpu -Suite float32)
    $Failed = $Failed -or (& $RunModelTests -Backend $QdqBackend -Suite qdq)

    if ($QdqBackend -ne "cpu") {
        Write-Host "-=-=-=- Running onnx/models qdq tests with context cache enabled -=-=-=-"
        # Scrub old context caches
        Get-ChildItem -Path "testdata\qdq-with-context-cache" -Recurse -Filter "*_ctx.onnx" | Remove-Item -Force

        $Failed = $Failed -or (& $RunModelTests -Backend $QdqBackend -Suite qdq-with-context-cache)
    } else {
        Write-Host "Not running onnx/models qdq tests with context cache enabled on CPU backend."
    }
}

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# QDC Logs
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# All done
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if ($Failed) {
    throw "Tests failed"
}
