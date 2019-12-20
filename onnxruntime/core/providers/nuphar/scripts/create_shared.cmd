:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off
setlocal EnableDelayedExpansion

if "%1"=="" goto Usage

set CACHE_DIR=%~f1
set MODEL_FILE=%~f2

if "%3"=="" (
set OUTPUT_DLL=jit.so
) else (
set OUTPUT_DLL=%3
)

REM check required tools
if not "%2"=="" (
REM need fciv when provided model file
where /q fciv.exe || echo Could not find fciv.exe, please make sure it is in PATH, or download from https://support.microsoft.com/en-us/help/841290 && exit /b -1
)
where /q cl.exe || echo Could not find cl.exe, please make sure it is in PATH, or install Visual Studio 2017 && exit /b -1
where /q link.exe || echo Could not find link.exe, please make sure it is in path, or install Visual Studio 2017 && exit /b -1

REM generate dllmain.cc
set DLLMAIN_CC=%CACHE_DIR%\dllmain.cc
echo Generating %DLLMAIN_CC%...
echo #include ^<windows.h^> >%DLLMAIN_CC%
echo BOOL APIENTRY DllMain(HMODULE hModule, >>%DLLMAIN_CC%
echo                       DWORD  ul_reason_for_call, >>%DLLMAIN_CC%
echo                       LPVOID lpReserved) >>%DLLMAIN_CC%
echo {return TRUE;} >>%DLLMAIN_CC%

REM skip checksum if no model file specified
if NOT EXIST "%MODEL_FILE%" goto Compile

REM get checksum from the model file
set CHECKSUM_CC=%CACHE_DIR%\checksum.cc
echo Generating %CHECKSUM_CC%...
for /f %%i in ('fciv %MODEL_FILE%') do (set MD5SUM=%%i)
echo #include ^<stdlib.h^> >%CHECKSUM_CC%
echo static const char model_checksum[] = "%MD5SUM%"; >>%CHECKSUM_CC%
echo extern "C" >>%CHECKSUM_CC%
echo __declspec(dllexport) >>%CHECKSUM_CC%
echo void _ORTInternal_GetCheckSum(const char*^& cs, size_t^& len) { >> %CHECKSUM_CC%
echo   cs = model_checksum; len = sizeof(model_checksum)/sizeof(model_checksum[0]) - 1;} >>%CHECKSUM_CC%

:Compile
cd /d %CACHE_DIR%
for /f %%i in ('dir /b *.cc') do (
  cl /Fo:%%i.o /c %%i
)

echo Linking %CACHE_DIR%\%OUTPUT_DLL%...
link -dll -FORCE:MULTIPLE *.o -EXPORT:__tvm_main__ -out:%CACHE_DIR%\%OUTPUT_DLL%
del *.o *.cc

exit /b

:Usage
echo Usage: %0 cache_dir [model_file] [output_dll]
echo The generated file would be cache_dir\output_dll
exit /b