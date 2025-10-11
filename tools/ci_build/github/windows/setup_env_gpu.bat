REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

if exist PATH=%AGENT_TEMPDIRECTORY%\v12.8\ (
    set PATH=%AGENT_TEMPDIRECTORY%\v12.8\bin;%AGENT_TEMPDIRECTORY%\v12.8\extras\CUPTI\lib64;%PATH%
) else (
    set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\CUPTI\lib64;%PATH%
)
set PATH=%AGENT_TEMPDIRECTORY%\TensorRT-10.13.3.9.Windows10.x86_64.cuda-12.9\lib;%PATH%

@REM The default version is still cuda v12.8, because set cuda v13.0 after it
set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\TensorRT-10.13.3.9.Windows10.x86_64.cuda-13.0\lib
if exist PATH=%AGENT_TEMPDIRECTORY%\v13.0\ (
    set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\v13.0\bin;%AGENT_TEMPDIRECTORY%\v13.0\extras\CUPTI\lib64
) else (
    set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\extras\CUPTI\lib64
)


set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY
