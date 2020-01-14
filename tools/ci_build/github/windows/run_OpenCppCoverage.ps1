# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Runs OpenCppCoverage for the Lotus unit tests and ONNX tests, and merges the coverage from all test runs.
Param(
    [Parameter(Mandatory=$true, HelpMessage="OpenCppCoverage exe.")][string]$OpenCppCoverageExe,
    [Parameter(Mandatory=$true, HelpMessage="Lotus enlistment root.")][string]$SourceRoot,
    [Parameter(Mandatory=$true, HelpMessage="Build root.")][string]$BuildRoot,
    [Parameter(Mandatory=$false, HelpMessage="IsLocalBuild")][switch]$LocalBuild = $true
)

if (-not $LocalBuild) {
# This is a hack to get the target path of the junctions in the build machine, lacking a neater way to do this.
# Assumes that the junction is 2 level upper from the SourceRoot/BuildRoot
# This is needed, because apparently the OpenCppCoverage cannot load the PDB symbol files from a junction 

    $buildLeaf = Split-Path $BuildRoot -Leaf
    $buildParent = Split-Path $BuildRoot -Parent
    $buildParentLeaf = Split-Path $buildParent -Leaf
    $buildParentParent = Split-Path $buildParent -Parent
    $buildParentParentTarget = Get-Item $buildParentParent | Select-Object -ExpandProperty Target

    $BuildRoot = Join-Path $buildParentParentTarget $buildParentLeaf
    $BuildRoot = Join-Path $BuildRoot $buildLeaf
}

$coreSources = Join-Path $SourceRoot "onnxruntime\core"
$headerSources = Join-Path $SourceRoot "include"
$buildDir = Join-Path $BuildRoot "Debug\Debug" 

function RunTest([string]$test_cmd, [string[]]$test_cmd_args, [string[]]$export_types, [string[]]$inputs)
{
    $cmd = "$OpenCppCoverageExe"
    $cmdParams = @("--sources=$headerSources","--sources=$coreSources","--modules=$buildDir","--working_dir=$buildDir")

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


Write-Host "ONNX test runner tests"
$onnx_test_runner = Join-Path $buildDir "onnx_test_runner.exe" 
RunTest $onnx_test_runner ($modelDir) ("binary:"  + (Join-Path $buildDir "onnx_test_runner.cov"))


Write-Host "C-API/Shared-lib test"
$shared_lib_test = Join-Path $buildDir "onnxruntime_shared_lib_test.exe"
RunTest $shared_lib_test @() ("binary:" + (Join-Path $buildDir "onnxruntime_shared_lib_test.cov"))


Write-Host "MLAS test"
$mlas_test = Join-Path $buildDir "onnxruntime_mlas_test.exe"
RunTest $mlas_test @() ("binary:" + (Join-Path $buildDir "onnxruntime_mlas_test.cov"))

Write-Host "Lotus unit tests"
# need to copy the tvm.dll, since it is not in the buildDir path
if (Test-Path -Path $BuildRoot\Debug\external\tvm\Debug\tvm.dll -PathType Leaf) {
    Copy-Item -Path $BuildRoot\Debug\external\tvm\Debug\tvm.dll -Destination $buildDir
}

$onnxruntime_test_all = Join-Path $buildDir "onnxruntime_test_all.exe"
RunTest $onnxruntime_test_all @() ("cobertura:$outputXml","html:$outputDir") ("onnxruntime_shared_lib_test.cov","onnx_test_runner.cov","onnxruntime_mlas_test.cov")

