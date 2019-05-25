REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@ECHO ON
SETLOCAL EnableDelayedExpansion

SET TargetFramework=netcoreapp2.1
SET TargetArch=x64
SET dn="C:\Program Files\dotnet\dotnet"

SET LocalNuGetRepo=%1
IF NOT "%2"=="" (SET TargetFramework=%2)
IF NOT "%3"=="" (SET TargetArch=%3)

IF "%TargetArch%"=="x86" (
  SET dn="C:\Program Files (x86)\dotnet\dotnet"
  SET RuntimeIdentifier=win-x86
  SET PlatformTarget=x86
)

ECHO Target Framework is %TargetFramework%
REM WorkingDirectory is Build.SourcesDirectory\csharp
SET /p MajorVersionNumber=<..\VERSION_NUMBER
SET VersionSuffix=
IF NOT DEFINED IsReleaseBuild (
    FOR /F "tokens=* USEBACKQ" %%F IN (`git rev-parse --short HEAD`) DO ( 
        set VersionSuffix=-dev-%%F 
    )
)

SET CurrentOnnxRuntimeVersion=%MajorVersionNumber%%VersionSuffix%

@echo %CurrentOnnxRuntimeVersion%
%dn% restore test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj -s %LocalNuGetRepo% --configfile .\Nuget.CSharp.config
IF NOT errorlevel 0 (
    @echo "Failed to restore nuget packages for the test project"
    EXIT 1
)

%dn% test test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore
IF NOT errorlevel 0 (
    @echo "Failed to build or execute the end-to-end test"
    EXIT 1
)
