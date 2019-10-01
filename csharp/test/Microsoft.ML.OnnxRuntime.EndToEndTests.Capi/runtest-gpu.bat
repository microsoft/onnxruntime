REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
echo on

set LocalNuGetRepo=%1
SET CurrentOnnxRuntimeVersion=%2
setlocal enableextensions disabledelayedexpansion

@echo %CurrentOnnxRuntimeVersion%

pushd test\Microsoft.ML.OnnxRuntime.EndToEndTests.Capi

REM Set up VS envvars
REM call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Generate packages.config with version
echo off
set "token=CurrentOnnxRuntimeVersion"
set "replace=%CurrentOnnxRuntimeVersion%"
set "templateFile=packages-gpu.conf"
for /f "delims=" %%i in ('type "%templateFile%" ^& break ^> "packages.config" ') do (
    set "line=%%i"
    setlocal enabledelayedexpansion
    >>"packages.config" echo(!line:%token%=%replace%!
    endlocal
)
echo on

REM Restore NuGet Packages
nuget restore -PackagesDirectory ..\packages -Source %LocalNuGetRepo% Microsoft.ML.OnnxRuntime.Gpu.EndToEndTests.RunCapi.vcxproj
if NOT %ERRORLEVEL% EQU 0 (
    echo "Error:Nuget restore failed"
    popd
    EXIT /B 1
)

REM Build Native project
msbuild  Microsoft.ML.OnnxRuntime.Gpu.EndToEndTests.RunCapi.vcxproj
if NOT %ERRORLEVEL% EQU 0 (
    echo "Error:MSBuild failed to compile project"
    popd
    EXIT /B 1
)


REM Run Unit Tests
pushd x64\Debug
REM vstest.console.exe /platform:x64 Microsoft.ML.OnnxRuntime.EndToEndTests.Capi.dll
.\Microsoft.ML.OnnxRuntime.Gpu.EndToEndTests.RunCapi.exe
if NOT %ERRORLEVEL% EQU 0 (
    echo "Unit test failure: %ERRORLEVEL%"
    popd
    popd
    EXIT /B 1
)

popd
popd

EXIT /B 0
