# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Runs OpenCppCoverage for the Lotus unit tests and ONNX tests, and merges the coverage from all test runs.
Param(
    [Parameter(Mandatory=$true, HelpMessage="OpenCppCoverage exe.")][string]$OpenCppCoverageExe,
    [Parameter(Mandatory=$true, HelpMessage="Lotus enlistment root.")][string]$SourceRoot,
    [Parameter(Mandatory=$true, HelpMessage="Build root.")][string]$BuildRoot
)

Get-Item WSMan:\localhost\Shell\MaxMemoryPerShellMB

$coreSources = Join-Path $SourceRoot "onnxruntime" 
$headerSources = Join-Path $SourceRoot "include" 
$buildDir = Join-Path $BuildRoot "Debug\Debug" -Resolve

function RunTest([string]$test_cmd, [string[]]$test_cmd_args, [string[]]$export_types, [string[]]$inputs)
{
    $cmd = "$OpenCppCoverageExe"
    $cmdParams = @("--sources=$headerSources","--sources=$coreSources","--modules=$buildDir\*","--working_dir=$buildDir")

    foreach($input in $inputs)
    {
        $inputPath = Join-Path $buildDir $input
        $cmdParams += "--input_coverage=$inputPath"
    }

    foreach($export_type in $export_types)
    {
        $cmdParams += "--export_type=$export_type"
    }

    $cmdParams += @("--","$test_cmd")
    $cmdParams += $test_cmd_args

    & $cmd $cmdParams
}

# generate cobertura xml output and html report
$outputXml = Join-Path $buildDir "cobertura.xml"
$outputDir = Join-Path $buildDir "OpenCppCoverageResults"
$modelDir = Join-Path $BuildRoot "models" 


# Lotus unit tests
#$onnxruntime_test_all = Join-Path $buildDir "onnxruntime_test_all.exe"
#Write-Host "Just try running the test_all.exe"
#& $OpenCppCoverageExe --sources="$coreSources" --modules="$buildDir\*" --working_dir=$buildDir -- $onnxruntime_test_all 

#RunTest $onnxruntime_test_all @() ("binary:" + (Join-Path $buildDir "onnxruntime_test_all.cov"))


# ONNX test runner tests. 
$onnx_test_runner = Join-Path $buildDir "onnx_test_runner.exe" -Resolve
$otr = "$onnx_test_runner " + $modelDir 
# TODO disabling due to long running time
# RunTest $onnx_test_runner ($modelDir) ("binary:"  + (Join-Path $buildDir "onnx_test_runner.cov"))


# C-API/Shared-lib test
$shared_lib_test = Join-Path $buildDir "onnxruntime_shared_lib_test.exe"
RunTest $shared_lib_test @() ("binary:" + (Join-Path $buildDir "onnxruntime_shared_lib_test.cov"))


# MLAS test
$mlas_test = Join-Path $buildDir "onnxruntime_mlas_test.exe"
# TODO: Disabling due to long time taken
# RunTest $mlas_test @() ("binary:" + (Join-Path $buildDir "onnxruntime_mlas_test.cov"))

# Session Without Environment test
#$session_test = Join-Path $buildDir "onnxruntime_test_framework_session_without_environment_standalone.exe"
#RunTest $session_test @() ("binary:" + (Join-Path $buildDir "onnxruntime_session_without_environment_test.cov"))

# Lotus unit tests
$onnxruntime_test_all = Join-Path $buildDir "onnxruntime_test_all.exe"
RunTest $onnxruntime_test_all @() ("cobertura:$outputXml","html:$outputDir") ("onnxruntime_shared_lib_test.cov")
#,"onnx_test_runner.cov"
#"onnxruntime_mlas_test.cov",
#,"onnxruntime_session_without_environment_test.cov"