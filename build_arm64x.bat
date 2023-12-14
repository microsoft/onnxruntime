:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off

setlocal
set PATH=C:\Program Files\Git\usr\bin;%PATH%
set LINK_REPRO_NAME=/mylink.rsp

rem Requires a Python install to be available in your PATH
python "%~dp0\tools\ci_build\build.py" --arm64 --buildasx  --build_dir "%~dp0\build\arm64-x" %*
python "%~dp0\tools\ci_build\build.py" --arm64ec --buildasx --build_dir "%~dp0\build\arm64ec-x" %*
