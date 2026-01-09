@echo off
SETLOCAL

SET BASEDIR=%~dp0

IF "%~1"=="" goto :PrintHelp
IF "%~2"=="" goto :PrintHelp

SET TARGET=%~1
SET CONFIG=%~2

IF /I NOT "%CONFIG%"=="Release" IF /I NOT "%CONFIG%"=="Debug" (
    echo Invalid configuration
    goto :PrintHelp
)

IF /I "%TARGET%"=="qnn" (
    SET PLATFORM=ARM64
) ELSE IF /I "%TARGET%"=="ov" (
    SET PLATFORM=x64
) ELSE IF /I "%TARGET%"=="vitisai" (
    SET PLATFORM=x64
) ELSE (
    echo Invalid target
    goto :PrintHelp
)

REM ---- Locate MSBuild ----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -find MSBuild\**\Bin\MSBuild.exe`) do (
    set "MSBUILD=%%i"
)

IF "%MSBUILD%"=="" (
    echo MSBuild not found
    exit /b 1
)

echo Using MSBuild: %MSBUILD%
SET MSBUILD_PROJECT_DIR=%BASEDIR%winml\EpValidationToolWinML
SET MSBUILD_PROJECT_PATH=%MSBUILD_PROJECT_DIR%\EpValidationToolWinML.vcxproj
SET MSBUILD_SOLUTION_PATH=%MSBUILD_PROJECT_DIR%\EpValidationToolWinML.sln
nuget restore "%MSBUILD_SOLUTION_PATH%"

"%MSBUILD%" "%MSBUILD_PROJECT_PATH%" ^
 /t:Clean;Build ^
 /p:Configuration=%CONFIG% ^
 /p:Platform=%PLATFORM% ^
 /p:WinMLEnableDefaultOrtHeaderIncludePath=true

exit /b 0

:PrintHelp
echo Usage: build.bat ^<qnn|ov|vitisai^> ^<Release|Debug^>
exit /b 1
