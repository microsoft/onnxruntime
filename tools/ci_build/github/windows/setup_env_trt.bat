REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@echo off
setlocal
REM Check if the user provided TensorRT paths as arguments
if "%~1"=="" (
    echo Please provide the TensorRT cuda 12 directory path as the first argument.
    exit /b 1
)
if "%~2"=="" (
    echo Please provide the TensorRT cuda 11 directory path as the second argument.
    exit /b 1
)

set TENSORRT_CUDA12_PATH=%~1
set TENSORRT_CUDA11_PATH=%~2

if exist PATH=%AGENT_TEMPDIRECTORY%\v12.2\ (
    set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\v12.2\bin;%AGENT_TEMPDIRECTORY%\v12.2\extras\CUPTI\lib64
) else (
    set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\CUPTI\lib64
)
set PATH=%AGENT_TEMPDIRECTORY%\%TENSORRT_CUDA12_PATH%\lib;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY

endlocal