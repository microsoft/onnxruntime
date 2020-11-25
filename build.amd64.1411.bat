:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

rem This will setup the VC env vars to use the 14.11 (VS2017 ver15.3) toolchain which is supported by CUDA 9.2 prior to running build.py.
rem It currently defaults to amd64 but that could be made configurable if that would be useful to developers running this locally.
@echo off

rem Use 14.11 toolset
call "%VCINSTALLDIR%\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.11

rem Requires a python 3.6 or higher install to be available in your PATH
python %~dp0\tools\ci_build\build.py --build_dir %~dp0\build\Windows %*