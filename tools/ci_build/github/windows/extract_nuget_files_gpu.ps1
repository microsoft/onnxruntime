# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget-Java Packaging Pipeline
New-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts -ItemType directory

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.zip | 
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.tgz | 
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\nuget-artifact" # *.tar will be created after *.tgz is extracted 
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\nuget-artifact -Filter *.tar | 
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}


New-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\_deps\protobuf-build\RelWithDebInfo -ItemType directory

Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-win-x64-cuda-*\lib\* -Destination $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo
Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\extra-artifact\protoc.exe $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\_deps\protobuf-build\RelWithDebInfo

$ort_dirs = Get-ChildItem -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-* -Directory
foreach ($ort_dir in $ort_dirs)
{  
  $dirname = Split-Path -Path $ort_dir -Leaf
  $dirname = $dirname.SubString(0,$dirname.LastIndexOf('-'))
  Write-Output "Renaming $ort_dir to $dirname"
  Rename-Item -Path $ort_dir -NewName $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\$dirname  
}

