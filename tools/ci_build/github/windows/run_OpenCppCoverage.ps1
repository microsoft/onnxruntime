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

    $sourceLeaf = Split-Path $SourceRoot -Leaf
    $sourceParent = Split-Path $SourceRoot -Parent
    $sourceParentLeaf = Split-Path $sourceParent -Leaf
    $sourceParentParent = Split-Path $sourceParent -Parent
    $sourceParentParentTarget = Get-Item $sourceParentParent | Select-Object -ExpandProperty Target
    $buildLeaf = Split-Path $BuildRoot -Leaf
    $buildParentLeaf = Split-Path $SourceRoot -Parent | Split-Path -Leaf

    $SourceRoot = Join-Path $sourceParentParentTarget $sourceParentLeaf
    $SourceRoot = Join-Path $SourceRoot $sourceLeaf

    $BuildRoot = Join-Path $sourceParentParentTarget $buildParentLeaf
    $BuildRoot = Join-Path $BuildRoot $buildLeaf
}

$coreSources = Join-Path $SourceRoot "onnxruntime\core"
$headerSources = Join-Path $SourceRoot "include"
$buildDir = Join-Path $BuildRoot "Debug\Debug" 
#-Resolve
#$buildDir = Get-Item $buildDir | Select-Object -ExpandProperty Target  # get the target of the symlink/junction

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


# ONNX test runner tests. 
$onnx_test_runner = Join-Path $buildDir "onnx_test_runner.exe" -Resolve
RunTest $onnx_test_runner ($modelDir) ("binary:"  + (Join-Path $buildDir "onnx_test_runner.cov"))


# C-API/Shared-lib test
$shared_lib_test = Join-Path $buildDir "onnxruntime_shared_lib_test.exe"
RunTest $shared_lib_test @() ("binary:" + (Join-Path $buildDir "onnxruntime_shared_lib_test.cov"))


# MLAS test
$mlas_test = Join-Path $buildDir "onnxruntime_mlas_test.exe"
RunTest $mlas_test @() ("binary:" + (Join-Path $buildDir "onnxruntime_mlas_test.cov"))

# Lotus unit tests
# need to copy the tvm.dll, since it is not in the buildDir path
Copy-Item -Path $BuildRoot\Debug\external\tvm\Debug\tvm.dll -Destination $buildDir

$onnxruntime_test_all = Join-Path $buildDir "onnxruntime_test_all.exe"
RunTest $onnxruntime_test_all @() ("cobertura:$outputXml","html:$outputDir") ("onnxruntime_shared_lib_test.cov","onnx_test_runner.cov","onnxruntime_mlas_test.cov")

