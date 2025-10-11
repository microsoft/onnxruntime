REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

if exist PATH=%AGENT_TEMPDIRECTORY%\v12.8\ (
    set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\v12.8\bin;%AGENT_TEMPDIRECTORY%\v12.8\extras\CUPTI\lib64
) else (
    set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\CUPTI\lib64
)
set PATH=%AGENT_TEMPDIRECTORY%\TensorRT-10.9.0.34.Windows10.x86_64.cuda-12.8\lib;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY
