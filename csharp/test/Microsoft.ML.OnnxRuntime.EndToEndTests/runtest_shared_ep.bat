REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@ECHO ON
SETLOCAL EnableDelayedExpansion

SET TargetFramework=netcoreapp2.1
SET TargetArch=x64
SET dn="C:\Program Files\dotnet\dotnet"
SET CurrentOnnxRuntimeVersion=""
SET ORTEP=""
SET DefineConstants=""

SET LocalNuGetRepo=%1
IF NOT "%2"=="" (SET TargetFramework=%2)
IF NOT "%3"=="" (SET TargetArch=%3)
IF NOT "%4"=="" (SET CurrentOnnxRuntimeVersion=%4)
IF NOT "%5"=="" (
    SET ORTEP=%5
) ELSE (
    echo "Usage: runtest_shared_ep.bat LocalNuGetRepoPath TargetFramework TargetArch NuGetPackageVersion ORTExecutionProvier"
)

IF "%TargetArch%"=="x64" (
  SET RuntimeIdentifier=win-x64
  SET PlatformTarget=x64
)

IF "%TargetArch%"=="x86" (
  SET dn="C:\Program Files (x86)\dotnet\dotnet"
  SET RuntimeIdentifier=win-x86
  SET PlatformTarget=x86
)

ECHO Target Framework is %TargetFramework%

IF "%ORTEP%"=="TensorRT" (
  SET DefineConstants="USE_TENSORRT"
)

@echo %DefineConstants
SET DefineConstants="USE_TENSORRT"

REM Update if CUDA lib paths if set
SET PATH=%CUDA_PATH%\bin;%CUDNN_PATH%\bin;%PATH%

IF EXIST test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages RMDIR /S /Q test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages
IF EXIST test\Microsoft.ML.OnnxRuntime.EndToEndTests\bin RMDIR /S /Q test\Microsoft.ML.OnnxRuntime.EndToEndTests\bin
IF EXIST test\Microsoft.ML.OnnxRuntime.EndToEndTests\obj RMDIR /S /Q test\Microsoft.ML.OnnxRuntime.EndToEndTests\obj

@echo %PackageName%
@echo %CurrentOnnxRuntimeVersion%
%dn% clean test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj

%dn% add test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj package Microsoft.ML.OnnxRuntime.Managed --no-restore -v %CurrentOnnxRuntimeVersion%
%dn% add test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj package Microsoft.ML.OnnxRuntime.TensorRT --no-restore -v %CurrentOnnxRuntimeVersion%

%dn% restore test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --configfile .\Nuget.CSharp.config --no-cache --packages test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages --source https://api.nuget.org/v3/index.json --source  %LocalNuGetRepo%

IF NOT errorlevel 0 (
    @echo "Failed to restore nuget packages for the test project"
    EXIT 1
)

dir test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages\
dir test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages\microsoft.ml.onnxruntime.tensorrt\%CurrentOnnxRuntimeVersion%\runtimes\win-x64\native\

%dn% list test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj package

%dn% test test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore
%dn% test test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore /p:DefineConstants=%DefineConstants% --filter "FullyQualifiedName=Microsoft.ML.OnnxRuntime.Tests.InferenceTest.CanRunInferenceOnAModelWithTensorRT"
%dn% test test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore /p:DefineConstants=%DefineConstants% --filter "FullyQualifiedName=Microsoft.ML.OnnxRuntime.Tests.InferenceTest.TestTensorRTProviderOptions" 

IF NOT errorlevel 0 (
    @echo "Failed to restore nuget packages for the test project"
    EXIT 1
)
IF NOT errorlevel 0 (
    @echo "Failed to build or execute the end-to-end test"
    EXIT 1
)
