REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@echo off
SETLOCAL EnableDelayedExpansion

set LocalNuGetRepo=%1
IF "%2"=="" (SET TargetFramework=netcoreapp2.1) ELSE (SET TargetFramework=%2)
ECHO Target Framework is %TargetFramework%
REM WorkingDirectory is Build.SourcesDirectory\csharp
set /p MajorVersionNumber=<..\VERSION_NUMBER
set VersionSuffix=
IF NOT DEFINED IsReleaseBuild (
    FOR /F "tokens=* USEBACKQ" %%F IN (`git rev-parse --short HEAD`) DO ( 
        set VersionSuffix=-dev-%%F 
    )
)

set CurrentOnnxRuntimeVersion=%MajorVersionNumber%%VersionSuffix%
@echo %CurrentOnnxRuntimeVersion%
dotnet restore test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj -s %LocalNuGetRepo% --configfile .\Nuget.CSharp.config
if NOT errorlevel 0 (
    @echo "Failed to restore nuget packages for the test project"
    Exit 1
)

dotnet test test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore
if NOT errorlevel 0 (
    @echo "Failed to build or execute the end-to-end test"
    Exit 1
)
