REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM for available runtime identifiers, see https://github.com/dotnet/corefx/blob/release/3.1/pkg/Microsoft.NETCore.Platforms/runtime.json
set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion
set gpu_nuget=""
set trt_nuget=""
set trt_dir=""

FOR /R %%i IN (*.nupkg) do (
    set filename=%%~ni
    IF "!filename:~25,3!"=="Gpu" (
        set gpu_nuget=%%~ni.nupkg
    )

    IF "!filename:~25,8!"=="TensorRT" (
        set trt_nuget=%%~ni.nupkg
        set trt_dir=%%~ni
        7z x !trt_nuget! -y -o!trt_dir!
     )
)

IF !gpu_nuget! == "" (
    echo "Can't find GPU nuget package"
    EXIT 1
)

IF !trt_nuget! == "" (
    echo "Can't find TensorRT nuget package"
    EXIT 1
)

mkdir runtimes\linux-x64\native
move onnxruntime-linux-x64\lib\libonnxruntime.so.1* runtimes\linux-x64\native\libonnxruntime.so
move onnxruntime-linux-x64\lib\libonnxruntime_providers_* runtimes\linux-x64\native

mkdir runtimes\win-x64\native
move !trt_dir!\runtimes\win-x64\native\onnxruntime_providers_tensorrt.dll runtimes\win-x64\native\onnxruntime_providers_tensorrt.dll
move !trt_dir!\runtimes\win-x64\native\onnxruntime.dll runtimes\win-x64\native\onnxruntime.dll
move !trt_dir!\runtimes\win-x64\native\onnxruntime.lib runtimes\win-x64\native\onnxruntime.lib
move !trt_dir!\runtimes\win-x64\native\onnxruntime.pdb runtimes\win-x64\native\onnxruntime.pdb

7z a !gpu_nuget! runtimes
