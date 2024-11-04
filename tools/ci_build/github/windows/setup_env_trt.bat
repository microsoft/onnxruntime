REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@echo off
setlocal
REM Check if the user provided TensorRT path as arguments
if "%~1"=="" (
    echo Please provide the TensorRT cuda 12 directory path as the first argument.
    exit /b 1
)
set TENSORRT_CUDA12_PATH=%~1

if exist PATH=%AGENT_TEMPDIRECTORY%\v12.2\ (
    set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\v12.2\bin;%AGENT_TEMPDIRECTORY%\v12.2\extras\CUPTI\lib64
) else (
    set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\CUPTI\lib64
)
set PATH=%AGENT_TEMPDIRECTORY%\%TENSORRT_CUDA12_PATH%\lib;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY

endlocal