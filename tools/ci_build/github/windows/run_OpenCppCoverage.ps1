# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Runs OpenCppCoverage for the Lotus unit tests and ONNX tests, and merges the coverage from all test runs.
Param(
    [Parameter(Mandatory=$true, HelpMessage="OpenCppCoverage exe.")][string]$OpenCppCoverageExe,
    [Parameter(Mandatory=$true, HelpMessage="Lotus enlistment root.")][string]$SourceRoot,
    [Parameter(Mandatory=$true, HelpMessage="Build root.")][string]$BuildRoot
)

"OpenCppCoverageExe=$OpenCppCoverageExe"
"SourceRoot=$SourceRoot"
"BuildRoot=$BuildRoot"

$sources = Join-Path $SourceRoot "onnxruntime\core" 
$buildDir = Join-Path $BuildRoot "Debug"

function RunTest([string]$test_cmd, [string[]]$export_types, [string[]]$inputs)
{
    $cmd = "$OpenCppCoverageExe --sources=$sources --modules=$buildDir --working_dir=$buildDir "

    foreach($input in $inputs)
    {
        $inputPath = Join-Path $buildDir $input
        $cmd += "--input_coverage=$inputPath "
    }

    foreach($export_type in $export_types)
    {
        $cmd += "--export_type=$export_type "
    }

    $cmd += "-- $test_cmd"
    $cmd
    Invoke-Expression -Command:$cmd
}

# generate cobertura xml output and html report
$outputXml = Join-Path $buildDir "cobertura.xml"
$outputDir = Join-Path $buildDir "OpenCppCoverageResults"

# Lotus unit tests
$onnxruntime_test_all = Join-Path $buildDir "onnxruntime_test_all.exe"
RunTest $onnxruntime_test_all ("binary:" + (Join-Path $buildDir "onnxruntime_test_all.cov"))

# Set to true to use vstest.console.exe to run the model tests. Requires a 'models' directory to have been created
# in $BuildRoot by running 'generate-data' from the ONNX installation.
# Set to false to use onnx_test_runner.exe and the models from the cmake/external directory.
$useVSTest = $false

if ($haveModelsInBuildDir)
{
    # ONNX tests via the vstest onnx_test_runner. merge output from onnxruntime_test_all.cov. 
    $onnx_test_models = "vstest.console.exe " + (Join-Path $buildDir "onnx_test_runner_vstest.dll")
    RunTest $onnx_test_models ("cobertura:$outputXml", "html:$outputDir") ("onnxruntime_test_all.cov")
}
else
{
    # ONNX test runner tests. 
    $onnx_test_runner = Join-Path $buildDir "onnx_test_runner.exe" 
    $otr_1 = "$onnx_test_runner " + (Join-Path $sourceRoot "cmake/external/onnx/onnx/backend/test/data/pytorch-converted")
    $otr_2 = "$onnx_test_runner " + (Join-Path $sourceRoot "cmake/external/onnx/onnx/backend/test/data/pytorch-operator")
    RunTest $otr_1 ("binary:"  + (Join-Path $buildDir "otr_1.cov"))
    RunTest $otr_2 ("cobertura:$outputXml", "html:$outputDir") ("onnxruntime_test_all.cov", "otr_1.cov")
}