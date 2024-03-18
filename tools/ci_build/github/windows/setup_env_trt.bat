REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

if exist PATH=%AGENT_TEMPDIRECTORY%\v12.3\ {
    set PATH=%AGENT_TEMPDIRECTORY%\v12.3\bin;%AGENT_TEMPDIRECTORY%\v12.3\extras\CUPTI\lib64;%PATH%
} else {
    set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\extras\CUPTI\lib64;%PATH%
}
set PATH=C:\local\TensorRT-10.0.0.6.Windows10.x86_64.cuda-12.4\lib;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY