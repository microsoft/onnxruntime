REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
echo on

set LocalNuGetRepo=%1
set CurrentOnnxRuntimeVersion=%2
setlocal enableextensions disabledelayedexpansion

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
nuget restore -NonInteractive -PackagesDirectory ..\packages -Source %LocalNuGetRepo% Microsoft.ML.OnnxRuntime.EndToEndTests.Capi.vcxproj

REM Build Native project
msbuild Microsoft.ML.OnnxRuntime.EndToEndTests.Capi.vcxproj

REM Run Unit Tests
vstest.console.exe /platform:x64 x64\debug\Microsoft.ML.OnnxRuntime.EndToEndTests.Capi.dll