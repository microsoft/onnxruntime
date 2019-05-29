# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Runs OpenCppCoverage for the Lotus unit tests and ONNX tests, and merges the coverage from all test runs.
Param(
    [Parameter(Mandatory=$true, HelpMessage="OpenCppCoverage exe.")][string]$OpenCppCoverageExe,
    [Parameter(Mandatory=$true, HelpMessage="Lotus enlistment root.")][string]$SourceRoot,
    [Parameter(Mandatory=$true, HelpMessage="Build root.")][string]$BuildRoot
)

#"OpenCppCoverageExe=$OpenCppCoverageExe"
#"SourceRoot=$SourceRoot"
#"BuildRoot=$BuildRoot"

$coreSources = Join-Path $SourceRoot "onnxruntime" 
$headerSources = Join-Path $SourceRoot "include" 
$buildDir = Join-Path $BuildRoot "Debug"

Write-Host "sources = " $coreSources $headerSources
Write-Host "buildDir = " $buildDir
Write-Host "OpenCppCoverageExe = " $OpenCppCoverageExe
Write-Host "SourceRoot = " $SourceRoot
Write-Host "BuildRoot = " $BuildRoot



function RunTest([string]$test_cmd, [string[]]$export_types, [string[]]$inputs)
{
    $cmd = "$OpenCppCoverageExe"
    $cmdParams = @("--sources=$headerSources","--sources=$coreSources","--modules=$buildDir","--working_dir=$buildDir")
#    Write-Host "cmd = " $cmd

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
    # $cmd += $cmdParams
    #Write-Host "final cmd = " $cmd
    #Write-Host "final cmdParams = " $cmdParams

    # Invoke-Expression -Command:$cmd  
    # echoargs.exe "$cmdParams"
    #& $cmd $cmdParams
    Start-Process -Wait -FilePath $cmd -ArgumentList $cmdParams 
}

# generate cobertura xml output and html report
$outputXml = Join-Path $buildDir "cobertura.xml"
$outputDir = Join-Path $buildDir "OpenCppCoverageResults"
$modelDir = Join-Path $BuildRoot "..\models" 

# ONNX test runner tests. 
$onnx_test_runner = Join-Path $buildDir "onnx_test_runner.exe" 
$otr_1 = "$onnx_test_runner " + $modelDir #(Join-Path $sourceRoot "cmake/external/onnx/onnx/backend/test/data/pytorch-converted")
#$otr_2 = "$onnx_test_runner " + $modelDir #(Join-Path $sourceRoot "cmake/external/onnx/onnx/backend/test/data/pytorch-operator")
RunTest $otr_1 ("binary:"  + (Join-Path $buildDir "otr_1.cov"))
#RunTest $otr_2 ("cobertura:$outputXml", "html:$outputDir") ("onnxruntime_test_all.cov", "otr_1.cov")

# C-API/Shared-lib test
$shared_lib_test = Join-Path $buildDir "onnxruntime_shared_lib_test.exe"
RunTest $shared_lib_test ("binary:" + (Join-Path $buildDir "onnxruntime_shared_lib_test.cov"))

# MLAS test
$mlas_test = Join-Path $buildDir "onnxruntime_mlas_test.exe"
RunTest $mlas_test ("binary:" + (Join-Path $buildDir "onnxruntime_mlas_test.cov"))

# Session Without Environment test
$session_test = Join-Path $buildDir "onnxruntime_test_framework_session_without_environment_standalone.exe"
RunTest $session_test ("binary:" + (Join-Path $buildDir "onnxruntime_session_without_environment_test.cov"))

# Lotus unit tests
$onnxruntime_test_all = Join-Path $buildDir "onnxruntime_test_all.exe"
RunTest $onnxruntime_test_all ("cobertura:$outputXml", "html:$outputDir") ("otr_1.cov","onnxruntime_shared_lib_test.cov","onnxruntime_mlas_test.cov","onnxruntime_session_without_environment_test.cov")