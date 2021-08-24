# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts -Filter *.zip | 
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\zip-artifacts"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts | Where-Object { $_.Name -match 'onnxruntime-win-gpu-x64-\d{1,}\.\d{1,}\.\d{1,}$' } | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\zip-artifacts\onnxruntime-win-gpu-x64

Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts | Where-Object { $_.Name -match 'onnxruntime-win-tensorrt-x64-\d{1,}\.\d{1,}\.\d{1,}$' } | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\zip-artifacts\onnxruntime-win-tensorrt-x64
