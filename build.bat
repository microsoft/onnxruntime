:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off

setlocal
set PATH=C:\Program Files\Git\usr\bin;%PATH%

rem Requires a Python install to be available in your PATH
python "%~dp0\tools\ci_build\build.py" --build_dir "%~dp0\build\Windows" %*
