REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@REM --- Setup CUDA 12.9 ---
@REM Check if a local/agent-specific version exists
if exist "%AGENT_TEMPDIRECTORY%\v12.9\" (
    echo "Using CUDA 12.9 from AGENT_TEMPDIRECTORY."
    set "PATH=%AGENT_TEMPDIRECTORY%\v12.9\bin;%AGENT_TEMPDIRECTORY%\v12.9\extras\CUPTI\lib64;%PATH%"
) else (
    echo "Using system default CUDA 12.9."
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\CUPTI\lib64;%PATH%"
)

@REM --- Setup TensorRT for CUDA 12.9 ---
set "TRT_12_9_PATH=%AGENT_TEMPDIRECTORY%\TensorRT-10.14.1.48.Windows.win10.x86_64.cuda-12.9\lib"
if exist "%TRT_12_9_PATH%\" (
    echo "Adding TensorRT 10.14.1.48 for CUDA 12.9 to PATH."
    set "PATH=%TRT_12_9_PATH%;%PATH%"
) else (
    echo "Warning: TensorRT 10.14.1.48 directory not found at %TRT_12_9_PATH%"
)


set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY
