@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: --- Base directory = where this bat file is located ---
SET BASEDIR=%~dp0

:: --- Check for required arguments ---
IF "%~1"=="" goto :PrintHelp
IF "%~2"=="" goto :PrintHelp

SET BUILD_TYPE=%~1
SET TARGET=%~2

:: --- Set CONFIGURATION (default to Release) ---
IF "%~3"=="" (
    SET CONFIG=Release
) ELSE IF /I "%~3"=="Release" (
    SET CONFIG=Release
) ELSE IF /I "%~3"=="Debug" (
    SET CONFIG=Debug
) ELSE (
    echo Invalid configuration %~3
    goto :PrintHelp
)

:: --- Determine PLATFORM based on TARGET ---
IF /I "%TARGET%"=="qnn" (
    SET PLATFORM=ARM64
) ELSE IF /I "%TARGET%"=="ov" (
    SET PLATFORM=x64
) ELSE IF /I "%TARGET%"=="vitisai" (
    SET PLATFORM=x64
) ELSE (
    echo Invalid target %TARGET%
    goto :PrintHelp
)


:: --- Decide which build system to use ---
IF /I "%BUILD_TYPE%"=="winml" (
    CALL :BuildMSBuild
	goto :EOF
) ELSE IF /I "%BUILD_TYPE%"=="psort" (
    CALL :BuildCMake
	goto :EOF
) ELSE (
    echo Invalid build type %BUILD_TYPE%
    goto :PrintHelp
)
:: --- Build with MSBuild ---
:BuildMSBuild
rem --- Locate vswhere.exe ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found at "%VSWHERE%"
    exit /b 1
)

rem --- Use CALL to handle spaces correctly ---
for /f "usebackq tokens=*" %%i in (`call "%VSWHERE%" -latest -products * -requires Microsoft.Component.MSBuild -find "MSBuild\**\Bin\MSBuild.exe"`) do (
    set "MSBUILD=%%i"
)

if "%MSBUILD%"=="" (
    echo ERROR: Could not locate MSBuild.exe
    exit /b 1
)

echo Using MSBuild: "%MSBUILD%"
echo Building using MSBuild for %BUILD_TYPE% on %TARGET%...
echo   - Building WinML version (USE_WINML_FEATURES=ON in vcxproj)
:: --- Paths ---
SET MSBUILD_PROJECT_DIR=%BASEDIR%winml\EpValidationToolWinML
SET MSBUILD_PROJECT_PATH=%MSBUILD_PROJECT_DIR%\EpValidationToolWinML.vcxproj
SET MSBUILD_SOLUTION_PATH=%MSBUILD_PROJECT_DIR%\EpValidationToolWinML.sln
nuget restore "%MSBUILD_SOLUTION_PATH%"
"%MSBUILD%" "%MSBUILD_PROJECT_PATH%" /t:Clean;Build /p:Configuration=%CONFIG% /p:Platform=%PLATFORM% /p:WinMLEnableDefaultOrtHeaderIncludePath=true
goto :EOF

:: --- Build with CMake ---
:BuildCMake
echo Building using CMake for %BUILD_TYPE% on %TARGET%...
echo   - Building PSORT version (USE_WINML_FEATURES=OFF)
set CMAKE_WORKING_DIR=%BASEDIR%
SET CMAKE_BUILD_DIR=%BASEDIR%\build_%TARGET%
SET CMAKE_INSTALL_DIR=%BASEDIR%\install_%TARGET%_%CONFIG%
set CMAKE_EXTRA_ARGS=-DUSE_WINML_FEATURES=OFF
cmake -S %CMAKE_WORKING_DIR% -A %PLATFORM% -B %CMAKE_BUILD_DIR% -DEP=%TARGET% %CMAKE_EXTRA_ARGS%
cmake --build %CMAKE_BUILD_DIR% --config %CONFIG%
cmake --install %CMAKE_BUILD_DIR% --config %CONFIG% --prefix %CMAKE_INSTALL_DIR%

goto :EOF



:: --- Help function ---
:PrintHelp
echo Usage: build.bat ^<winml|psort^> ^<qnn|ov|vitisai^> ^<Release|Debug^>
echo Example: build.bat winml qnn Release
exit /b 1
