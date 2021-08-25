# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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


New-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\external\protobuf\cmake\RelWithDebInfo -ItemType directory

Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-win-x64-cuda-*\lib\* -Destination $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo
Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\extra-artifact\protoc.exe $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\external\protobuf\cmake\RelWithDebInfo


Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts | Where-Object { $_.Name -match 'onnxruntime-linux-x64-cuda-\d{1,}\.\d{1,}\.\d{1,}$' } | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-linux-x64-gpu
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-linux-x64-tensorrt-* | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-linux-x64-tensorrt
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-win-x64-tensorrt-* | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-win-x64-tensorrt
