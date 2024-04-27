REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

if exist PATH=%AGENT_TEMPDIRECTORY%\v11.8\ (
    set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\v11.8\bin;%AGENT_TEMPDIRECTORY%\v11.8\extras\CUPTI\lib64
) else (
    set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
)
set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\TensorRT-10.0.1.6.Windows10.x86_64.cuda-11.8\lib

@REM The default version is cuda v11.8, because set cuda v12.3 after it
set PATH=%AGENT_TEMPDIRECTORY%\TensorRT-10.0.1.6.Windows10.x86_64.cuda-12.4\lib;%PATH%
if exist PATH=%AGENT_TEMPDIRECTORY%\v12.3\ (
    set PATH=%AGENT_TEMPDIRECTORY%\v12.3\bin;%AGENT_TEMPDIRECTORY%\v12.3\extras\CUPTI\lib64;%PATH%
) else (
    set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\extras\CUPTI\lib64;%PATH%
)

set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY