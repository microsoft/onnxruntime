@echo off
setlocal

set vswherepath="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set vcvarsall_arch=%1
if "%vcvarsall_arch%" == "x86" (
  set vcvarsall_arch=x86
) else (
  set vcvarsall_arch=x64
)

for /f "usebackq delims=" %%i in (`%vswherepath% -latest -property installationPath`) do (
  if exist "%%i\VC\Auxiliary\Build\vcvars%vcvarsall_arch%.bat" (
    set "vcvarsall=%%i\VC\Auxiliary\Build\vcvars%vcvarsall_arch%.bat"
  )
)

echo "vcvarsall will be used as the VC compiler: %vcvarsall%"

REM Get initial environment variables
set > initial_env.txt

REM Call vcvarsall.bat
call "%vcvarsall%"

REM Get environment variables after calling vcvarsall.bat
set > final_env.txt

REM Call the Python script to update the GitHub Actions environment
python "%~dp0\update_environment.py"

endlocal