REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
ECHO on

SET LocalNuGetRepo=%1
SET TargetArch=x64
IF NOT "%2"=="" (SET TargetArch=%2)
SET CurrentOnnxRuntimeVersion=%3

SETLOCAL enableextensions disabledelayedexpansion

@ECHO %CurrentOnnxRuntimeVersion%

PUSHD test\Microsoft.ML.OnnxRuntime.EndToEndTests.Capi

REM SET up VS envvars
REM call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvars64.bat"

REM Generate packages.config with version
ECHO off
SET "token=CurrentOnnxRuntimeVersion"
SET "replace=%CurrentOnnxRuntimeVersion%"
SET "templateFile=packages.conf"
for /f "delims=" %%i in ('type "%templateFile%" ^& break ^> "packages.config" ') do (
    SET "line=%%i"
    SETLOCAL enabledelayedexpansion
    >>"packages.config" ECHO(!line:%token%=%replace%!
    ENDLOCAL
)
ECHO on

REM Update project file with package name (e.g. Microsoft.ML.OnnxRuntime.Gpu)
IF "%PackageName%"==""  goto :skip
@ECHO off
SETLOCAL EnableExtensions DisableDelayedExpansion
SET "search="Microsoft.ML.OnnxRuntime""
SET "replace="%PackageName%""
SET "projfile="packages.config""
FOR /f "delims=" %%i in ('type "packages.config" ^& break ^> "packages.config" ') do (
        SET "line=%%i"
        SETLOCAL enabledelayedexpansion
        >>"packages.config" ECHO(!line:%search%=%replace%!
        ENDLOCAL
  )
:skip


REM Restore NuGet Packages
nuget restore -PackagesDirectory ..\packages -Source %LocalNuGetRepo% Microsoft.ML.OnnxRuntime.EndToEndTests.RunCapi.vcxproj
if NOT %ERRORLEVEL% EQU 0 (
    ECHO "Error:Nuget restore failed"
    POPD
    EXIT /B 1
)


IF "%TargetArch%"=="x86" (
   SET OutputDir="Debug"
) ELSE (
   SET OutputDir="x64\Debug"
)

REM Build Native project
msbuild  /p:Platform=%TargetArch%  Microsoft.ML.OnnxRuntime.EndToEndTests.RunCapi.vcxproj
if NOT %ERRORLEVEL% EQU 0 (
    ECHO "Error:MSBuild failed to compile project"
    POPD
    EXIT /B 1
)

REM Run Unit Tests
PUSHD %OutputDir%
REM vstest.console.exe /platform:x64 Microsoft.ML.OnnxRuntime.EndToEndTests.Capi.dll
.\Microsoft.ML.OnnxRuntime.EndToEndTests.RunCapi.exe
if NOT %ERRORLEVEL% EQU 0 (
    ECHO "Unit test failure: %ERRORLEVEL%"
    POPD
    POPD
    EXIT /B 1
)

POPD
POPD

EXIT /B 0
