:: Locates and calls VS 2017's vcvarsall.bat

@echo off

for /f "usebackq delims=" %%i in (`powershell -Command "%~dp0\get_vcvarsall_path.ps1" -VsVersion 2017`) do (
  echo %%i
  set "VcVarsAllPath=%%i"
)

if "%VcVarsAllPath%"=="" (
  echo Unable to get path to vcvarsall.bat.
  exit /b 1
)

call "%VcVarsAllPath%" %*
