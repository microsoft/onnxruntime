:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off

set argC=0
for %%x in (%*) do Set /A argC+=1

if %argC%==0 goto Default
if %argC%==1 goto Usage
@echo Unexpected argument was passed - please provide build configuration {Release,Debug,RelWithDebInfo}
goto :eof

:default
rem Requires a python 3.6 or higher install to be available in your PATH
@echo Building ORT Web default configuration - Release
python %~dp0\scripts\build_web.py --config Release
goto :eof

:usage
rem Requires a python 3.6 or higher install to be available in your PATH
@echo Building ORT Web configuration - %1
python %~dp0\scripts\build_web.py --config %1
exit /B 1
