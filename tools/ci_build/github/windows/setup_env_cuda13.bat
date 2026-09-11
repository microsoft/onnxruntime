REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@REM --- Setup for CUDA 13.0 ---
if exist "%AGENT_TEMPDIRECTORY%\v13.0\" (
    echo "Using CUDA 13.0 from AGENT_TEMPDIRECTORY."
    set "PATH=%AGENT_TEMPDIRECTORY%\v13.0\bin;%AGENT_TEMPDIRECTORY%\v13.0\extras\CUPTI\lib64;%PATH%"
) else (
    echo "Using system default CUDA 13.0."
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\extras\CUPTI\lib64;%PATH%"
)

@REM --- Setup TensorRT for CUDA 13.0 ---
set "TRT_13_0_PATH=%AGENT_TEMPDIRECTORY%\TensorRT-10.13.3.9.Windows.win10.cuda-13.0\lib"
if exist "%TRT_13_0_PATH%\" (
    echo "Adding TensorRT 10.13.3.9 for CUDA 13.0 to PATH."
    set "PATH=%TRT_13_0_PATH%;%PATH%"
) else (
    echo "Warning: TensorRT 10.13.3.9 directory not found at %TRT_13_0_PATH%"
)

set GRADLE_OPTS=-Dorg.gradle.daemon=false
set CUDA_MODULE_LOADING=LAZY
