REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
echo on

set LocalNuGetRepo=%1
setlocal enableextensions disabledelayedexpansion

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

pushd test\Microsoft.ML.OnnxRuntime.EndToEndTests.Capi

REM Set up VS envvars
REM call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Generate packages.config with version
echo off
set "token=CurrentOnnxRuntimeVersion"
set "replace=%CurrentOnnxRuntimeVersion%"
set "templateFile=packages.conf"
for /f "delims=" %%i in ('type "%templateFile%" ^& break ^> "packages.config" ') do (
    set "line=%%i"
    setlocal enabledelayedexpansion
    >>"packages.config" echo(!line:%token%=%replace%!
    endlocal
)
echo on

REM Restore NuGet Packages
nuget restore -PackagesDirectory ..\packages -Source %LocalNuGetRepo% Microsoft.ML.OnnxRuntime.EndToEndTests.Capi.vcxproj
if NOT %ERRORLEVEL% EQU 0 (
	echo "Error:Nuget restore failed"
	EXIT /B 1
)

REM Build Native project
msbuild Microsoft.ML.OnnxRuntime.EndToEndTests.Capi.vcxproj
if NOT %ERRORLEVEL% EQU 0 (
	echo "Error:MSBuild failed to compile project"
	EXIT /B 1
)


REM Run Unit Tests
vstest.console.exe /platform:x64 x64\debug\Microsoft.ML.OnnxRuntime.EndToEndTests.Capi.dll
if NOT %ERRORLEVEL% EQU 0 (
    echo "Unit test failure: %ERRORLEVEL%"
)

popd
EXIT /B 0
