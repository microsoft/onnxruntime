# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline

# Re-construct a build directory that contains binaries from all the different platforms we're including
# in the native ORT nuget package
$nuget_artifacts_dir = "$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts"
New-Item -Path $nuget_artifacts_dir -ItemType directory

## .zip files
# unzip directly
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.zip |
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$nuget_artifacts_dir"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

## .tgz files
# first extract the tar file from the tgz
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.tgz |
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\nuget-artifact"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

# now extract the actual folder structure from the tar file to the build dir
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.tar |
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$nuget_artifacts_dir"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

# copy android AAR.
# for full build of onnxruntime Android AAR, there should only be one .aar file
# called onnxruntime-android-x.y.z.aar or onnxruntime-training-android-x.y.z.aar but sanity check that
$aars = Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.aar
if ($aars.Count -eq 1) {
  $aar = $aars[0]
  $aar_prefix = "onnxruntime"
  if ($aar -like "onnxruntime-training*") {
    $aar_prefix = "onnxruntime-training"
  }
  $target_dir = "$nuget_artifacts_dir\$aar_prefix-android-aar"
  $target_file = "$target_dir\onnxruntime.aar"  # remove '-mobile' and version info from filename
  New-Item -Path $target_dir -ItemType directory

  Write-Output "Copy-Item $($aar.FullName) $target_file"
  Copy-Item $aar.FullName $target_file
}
elseif ($aars.Count -gt 1) {
  Write-Error "Expected at most one Android .aar file but got: [$aars]"
}

New-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\_deps\protobuf-build\RelWithDebInfo -ItemType directory

# Check whether this is a training pipeline
$is_training_pipeline = $false
if (Test-Path -Path $nuget_artifacts_dir\onnxruntime-training-win-x64-*) {
  $is_training_pipeline = $true
  Write-Output "onnxruntime-training-win-x64-* dir exists. This is a training pipeline."
}

# Copy onnxruntime and protoc binaries to the binaries dir as these are required
# by Microsoft.ML.OnnxRuntime.Tests.NetCoreApp
if ($is_training_pipeline) {
  Copy-Item -Path $nuget_artifacts_dir\onnxruntime-training-win-x64-*\lib\* -Destination $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo
}
else {
  Copy-Item -Path $nuget_artifacts_dir\onnxruntime-win-x64-*\lib\* -Destination $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo
}

Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\extra-artifact\protoc.exe $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\_deps\protobuf-build\RelWithDebInfo

"Get-ChildItem -Directory -Path $nuget_artifacts_dir\onnxruntime-*"
$ort_dirs = Get-ChildItem -Directory -Path $nuget_artifacts_dir\onnxruntime-*
foreach ($ort_dir in $ort_dirs)
{
  # remove the last '-xxx' segment from the dir name. typically that's the architecture.
  $dirname = Split-Path -Path $ort_dir -Leaf
  $dirname = $dirname.SubString(0,$dirname.LastIndexOf('-'))
  Write-Output "Renaming $ort_dir to $dirname"
  Rename-Item -Path $ort_dir -NewName $nuget_artifacts_dir\$dirname
}

# List artifacts
"Post copy artifacts"
Get-ChildItem -Recurse $nuget_artifacts_dir\
