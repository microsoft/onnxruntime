# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts -Filter *.zip | 
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\zip-artifacts"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts -Filter *onnxruntime-win-x64-cuda*.zip |
Foreach-Object {
 $($_.FullName) -match '.*onnxruntime-win-x64-cuda-(.*).zip'
 $version=$matches[1]
 Rename-Item -Path $($_.FullName) -NewName onnxruntime-win-x64-gpu-$version.zip
}

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts | Where-Object { $_.Name -match 'onnxruntime-win-x64-cuda-\d{1,}\.\d{1,}\.\d{1,}$' } | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\zip-artifacts\onnxruntime-win-x64-gpu

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts | Where-Object { $_.Name -match 'onnxruntime-win-x64-tensorrt-\d{1,}\.\d{1,}\.\d{1,}$' } | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\zip-artifacts\onnxruntime-win-x64-tensorrt
