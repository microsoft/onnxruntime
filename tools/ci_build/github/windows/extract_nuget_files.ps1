# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline

# Re-construct a build directory that contains binaries from all the different platforms we're including
# in the native ORT nuget package
New-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts -ItemType directory

## .zip files
# unzip directly
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.zip | 
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts"
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
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

# copy android AAR folder (e.g. onnxruntime-android-full-aar)
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *-aar -Directory | 
Foreach-Object {
 Write-Output "Copy-Item -Recurse $($_.FullName) $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts"
 Copy-Item -Recurse $_.FullName $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts
}

New-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\external\protobuf\cmake\RelWithDebInfo -ItemType directory

Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-win-x64-*\lib\* -Destination $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo
Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\extra-artifact\protoc.exe $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\external\protobuf\cmake\RelWithDebInfo

"Get-ChildItem -Directory -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-*"
$ort_dirs = Get-ChildItem -Directory -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-*
foreach ($ort_dir in $ort_dirs)
{
  # remove the last '-xxx' segment from the dir name. typically that's the architecture. 
  $dirname = Split-Path -Path $ort_dir -Leaf
  $dirname = $dirname.SubString(0,$dirname.LastIndexOf('-'))
  Write-Output "Renaming $ort_dir to $dirname"
  Rename-Item -Path $ort_dir -NewName $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\$dirname  
}

# List artifacts
"Post copy artifacts"
Get-ChildItem -Recurse $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\