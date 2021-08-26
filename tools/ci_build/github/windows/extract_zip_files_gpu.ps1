# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# extract *-cuda-*.zip and *-tensorrt-*.zip
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts -Filter *.zip | 
Foreach-Object {
 $cmd = "7z.exe x $($_.FullName) -y -o$Env:BUILD_BINARIESDIRECTORY\zip-artifacts"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}

# Rename tensorrt directory for later use in bundle_dlls_gpu.bat
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts | Where-Object { $_.Name -match 'onnxruntime-win-x64-tensorrt-\d{1,}\.\d{1,}\.\d{1,}$' } | Rename-Item -NewName $Env:BUILD_BINARIESDIRECTORY\zip-artifacts\onnxruntime-win-x64-tensorrt
Remove-Item $Env:BUILD_BINARIESDIRECTORY\zip-artifacts\*.zip

# Rename cuda directory to gpu directory and re-compress it for later use in bundle_dlls_gpu.bat
Get-ChildItem $Env:BUILD_BINARIESDIRECTORY\zip-artifacts -Filter *cuda* |
Foreach-Object {
 $($_.FullName) -match '.*onnxruntime-win-x64-cuda-(.*)'
 $version=$matches[1]
 Rename-Item -Path $($_.FullName) -NewName onnxruntime-win-x64-gpu-$version
 $cmd = "7z.exe a $Env:BUILD_BINARIESDIRECTORY\zip-artifacts\onnxruntime-win-x64-gpu-$version.zip $Env:BUILD_BINARIESDIRECTORY\zip-artifacts\onnxruntime-win-x64-gpu-$version"
 Write-Output $cmd
 Invoke-Expression -Command $cmd
}
