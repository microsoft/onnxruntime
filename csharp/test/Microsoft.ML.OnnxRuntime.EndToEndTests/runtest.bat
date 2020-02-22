@ECHO ON
SETLOCAL EnableDelayedExpansion

SET TargetFramework=netcoreapp2.1
SET TargetArch=x64
SET dn="C:\Program Files\dotnet\dotnet"

SET LocalNuGetRepo=%1
IF NOT "%2"=="" (SET TargetFramework=%2)
IF NOT "%3"=="" (SET TargetArch=%3)
IF NOT "%4"=="" (
    SET CurrentOnnxRuntimeVersion=%4
) ELSE (
    echo "Usage: runtest.bat LocalNuGetRepoPath TargetFramework TargetArch NuGetPackageVersion"
)

IF "%TargetArch%"=="AnyCPU" (
    IF "%TargetFramework%"=="net461" (
        @echo "Can't build AnyCPU app for .NET Framework"
        EXIT 1
    )
)

IF "%TargetArch%"=="x64" (
  SET RuntimeIdentifier=win-x64
)

IF "%TargetArch%"=="x86" (
  SET RuntimeIdentifier=win-x86
)

ECHO Target Framework is %TargetFramework%

REM Update if CUDA lib paths if set
SET PATH=%CUDA_PATH%\bin;%CUDNN_PATH%\bin;%PATH%

@echo %CurrentOnnxRuntimeVersion%
%dn% restore test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --configfile .\Nuget.CSharp.config --no-cache --packages test\Microsoft.ML.OnnxRuntime.EndToEndTests --source https://api.nuget.org/v3/index.json --source  %LocalNuGetRepo%

IF NOT errorlevel 0 (
    @echo "Failed to restore nuget packages for the test project"
    EXIT 1
)

%dn% test test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore
IF NOT errorlevel 0 (
    @echo "Failed to build or execute the end-to-end test"
    EXIT 1
)
