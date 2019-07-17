:: Locates and calls VS 2017's vcvarsall.bat

@echo off

for /f "usebackq delims=" %%i in (`"%~dp0\vswhere.bat" -version 15.0 -property installationPath`) do (
  if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
    set "VcVarsAllPath=%%i\VC\Auxiliary\Build\vcvarsall.bat"
  )
)

if "%VcVarsAllPath%"=="" (
  echo Unable to get path to vcvarsall.bat.
  exit /b 1
)

call "%VcVarsAllPath%" %*
