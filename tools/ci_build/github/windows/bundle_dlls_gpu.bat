REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM for available runtime identifiers, see https://github.com/dotnet/corefx/blob/release/3.1/pkg/Microsoft.NETCore.Platforms/runtime.json
powershell -Command "Invoke-WebRequest http://stahlworks.com/dev/unzip.exe -OutFile unzip.exe"
powershell -Command "Invoke-WebRequest http://stahlworks.com/dev/zip.exe -OutFile zip.exe"
set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion
set gpu_nuget=""
set gpu_zip=""
set gpu_dir=""
set trt_nuget=""
set trt_zip=""
set trt_dir=""
FOR /R %%i IN (*.nupkg) do (
    set filename=%%~ni
    IF "!filename:~25,3!"=="Gpu" (
        set gpu_nuget=%%~ni.nupkg
        set gpu_zip=%%~ni.zip
        set gpu_dir=%%~ni
        rename !gpu_nuget! !gpu_zip!
        unzip !gpu_zip! -d !gpu_dir!
        del /Q !gpu_zip!
    )
    IF "!filename:~25,8!"=="TensorRT" (
        set trt_nuget=%%~ni.nupkg
        set trt_zip=%%~ni.zip
        set trt_dir=%%~ni
        rename !trt_nuget! !trt_zip!
        unzip !trt_zip! -d !trt_dir!
        del /Q !trt_zip!
     )
)

IF !gpu_dir! == "" (
    echo "Can't find GPU nuget package to unpack/pack"
    EXIT 1
)

unzip linux-x64.zip -d linux-x64
mkdir !gpu_dir!\runtimes\linux-x64
mkdir !gpu_dir!\runtimes\linux-x64\native
move linux-x64\linux-x64\libonnxruntime.so !gpu_dir!\runtimes\linux-x64\native\libonnxruntime.so
move linux-x64\linux-x64\libonnxruntime_providers_shared.so !gpu_dir!\runtimes\linux-x64\native\libonnxruntime_providers_shared.so
move linux-x64\linux-x64\libonnxruntime_providers_cuda.so !gpu_dir!\runtimes\linux-x64\native\libonnxruntime_providers_cuda.so

IF !trt_dir! == "" (
    echo "Can't find TensorRT nuget package to unpack/pack"
    EXIT 1
)

move !trt_dir!\runtimes\win-x64\native\onnxruntime_providers_tensorrt.dll !gpu_dir!\runtimes\win-x64\native\onnxruntime_providers_tensorrt.dll
move !trt_dir!\runtimes\win-x64\native\onnxruntime.dll !gpu_dir!\runtimes\win-x64\native\onnxruntime.dll
move !trt_dir!\runtimes\win-x64\native\onnxruntime.lib !gpu_dir!\runtimes\win-x64\native\onnxruntime.lib
move !trt_dir!\runtimes\win-x64\native\onnxruntime.pdb !gpu_dir!\runtimes\win-x64\native\onnxruntime.pdb
pushd !gpu_dir! 
zip -r ..\!gpu_zip! .
popd
move !gpu_zip! !gpu_nuget!
