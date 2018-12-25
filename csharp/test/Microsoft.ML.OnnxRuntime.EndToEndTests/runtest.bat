@echo off
SETLOCAL EnableDelayedExpansion
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
dotnet restore test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj -s %buildDirectory%\NuGetRepo --configfile .\Nuget.CSharp.config
dotnet test test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore