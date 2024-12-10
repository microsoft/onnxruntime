REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

if exist PATH=%AGENT_TEMPDIRECTORY%\v12.2\ (
    set PATH=%AGENT_TEMPDIRECTORY%\v12.2\bin;%AGENT_TEMPDIRECTORY%\v12.2\extras\CUPTI\lib64;%PATH%
) else (
    set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\extras\CUPTI\lib64;%PATH%
)
set PATH=%AGENT_TEMPDIRECTORY%\TensorRT-10.6.0.26.Windows10.x86_64.cuda-12.6\lib;%PATH%

@REM The default version is still cuda v12.2, because set cuda v11.8 after it
set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\TensorRT-10.6.0.26.Windows10.x86_64.cuda-11.8\lib
if exist PATH=%AGENT_TEMPDIRECTORY%\v11.8\ (
    set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\v11.8\bin;%AGENT_TEMPDIRECTORY%\v11.8\extras\CUPTI\lib64
) else (
    set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\\extras\CUPTI\lib64
)


set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY
