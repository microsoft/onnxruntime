:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

@echo off
rem Requires a python 3.6 or higher install to be available in your PATH
c:\Users\jeffbloo\AppData\Local\Continuum\anaconda3\python.exe %~dp0\tools\ci_build\build.py --build_dir %~dp0\build\Windows %*