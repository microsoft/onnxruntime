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
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\nuget-artifact"
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

Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-win-gpu-x64-*\lib\* -Destination $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo
Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\extra-artifact\protoc.exe $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\external\protobuf\cmake\RelWithDebInfo


Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-linux-x64-* | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\RelWithDebInfo\RelWithDebInfo\nuget-artifacts\onnxruntime-linux-x64
