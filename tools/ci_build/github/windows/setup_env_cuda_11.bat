REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM This file is used by DML Nuget Pipeline,Nuget GPU TensorRT Pipeline,Zip-Nuget-Java Packaging Pipeline
set PATH=C:\azcopy;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.1\bin;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
