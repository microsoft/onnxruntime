# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

$onnx_runtime_release_version = $Env:ONNXRUNTIMEVERSION + $Env:RELEASEVERSIONSUFFIX
$ErrorActionPreference = "Stop"
Write-Output "Start"
dir
Copy-Item -Path $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-linux-x64\ai\onnxruntime\native\linux-x64\libonnxruntime_providers_cuda.so -Destination $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-linux-x64-tensorrt\ai\onnxruntime\native\linux-x64
pushd onnxruntime-java-linux-x64-tensorrt
Write-Output "Run 7z"
7z a $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64\testing.jar libcustom_op_library.so
Remove-Item -Path libcustom_op_library.so
7z a $Env:BUILD_BINARIESDIRECTORY\java-artifact\onnxruntime-java-win-x64\onnxruntime-$onnx_runtime_release_version.jar .
popd
pushd onnxruntime-java-win-x64
ren onnxruntime-$onnx_runtime_release_version.jar onnxruntime_gpu-$onnx_runtime_release_version.jar
ren onnxruntime-$onnx_runtime_release_version-javadoc.jar onnxruntime_gpu-$onnx_runtime_release_version-javadoc.jar
ren onnxruntime-$onnx_runtime_release_version-sources.jar onnxruntime_gpu-$onnx_runtime_release_version-sources.jar
ren onnxruntime-$onnx_runtime_release_version.pom onnxruntime_gpu-$onnx_runtime_release_version.pom
popd
