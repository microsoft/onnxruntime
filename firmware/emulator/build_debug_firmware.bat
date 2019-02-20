:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.
@echo on

:: parameter 1: the full path of folder containing firmware bond files and .c/.h files.
:: parameter 2: the full path of nuget_root folder
if "%~1"=="" goto:invalid_parameter
if "%~2"=="" goto:invalid_parameter
set FIRMWARE_PATH=%~1
set NUGET_ROOT=%~2

:: use filename parsed from full file path
set FirmwareName=%~nx1

set DEBUG_PATH=%FIRMWARE_PATH%\%FirmwareName%_debug
if not exist %DEBUG_PATH% mkdir %DEBUG_PATH%

set CODE_PATH=%DEBUG_PATH%\%FirmwareName%
if not exist %CODE_PATH% mkdir %CODE_PATH%
copy %FIRMWARE_PATH%\*.c %CODE_PATH%
copy %FIRMWARE_PATH%\*.h %CODE_PATH%
copy %FIRMWARE_PATH%\*.bond %CODE_PATH%


echo 'Generating formware property file'
copy ".\emulator_template.vcxproj" %DEBUG_PATH%\emulator.vcxproj
copy ".\firmware-template.props" %DEBUG_PATH%\firmware-template.props

call fill_file_list.py --firmware_name %FirmwareName% --target_folder %CODE_PATH% --template_prop_file %DEBUG_PATH%\firmware-template.props --target_prop_file %DEBUG_PATH%\firmware.props

echo 'Setting environment variables'
call ..\set_env.bat %NUGET_ROOT%

pushd %DEBUG_PATH%
echo 'Building emulator'
msbuild /p:Platform=x64 emulator.vcxproj
popd

echo 'Copy related files to Giano'
set GIANO_TARGET_PATH=%Giano%\build\native\tests\nios\bs3.0\
copy %DEBUG_PATH%\x64\Debug\%FirmwareName%.dll %GIANO_TARGET_PATH%
copy %DEBUG_PATH%\x64\Debug\%FirmwareName%.pdb %GIANO_TARGET_PATH%
copy %DEBUG_PATH%\x64\Debug\schema.bin %GIANO_TARGET_PATH%

copy %BrainSliceImage%\content\BrainSlice.sku.json %GIANO_TARGET_PATH%

set DEVKIT_DEPENDENCY_PATH=%DevKit%\lib\native\python36
copy %DEVKIT_DEPENDENCY_PATH%\bs.plx %GIANO_TARGET_PATH%
copy %DEVKIT_DEPENDENCY_PATH%\bs_hex.dll %GIANO_TARGET_PATH%
copy %DEVKIT_DEPENDENCY_PATH%\bs_haas.dll %GIANO_TARGET_PATH%
copy %DEVKIT_DEPENDENCY_PATH%\bs_omp.dll %GIANO_TARGET_PATH%
copy %DEVKIT_DEPENDENCY_PATH%\bs_telemetry.dll %GIANO_TARGET_PATH%
copy %DEVKIT_DEPENDENCY_PATH%\bs_imp.dll %GIANO_TARGET_PATH%
copy %DEVKIT_DEPENDENCY_PATH%\bs_nfu_bfp.dll %GIANO_TARGET_PATH%
copy %DEVKIT_DEPENDENCY_PATH%\bs_nfu_float32.dll %GIANO_TARGET_PATH%

echo 'Complete successfully.'
echo Please run "start_giano.bat %FirmwareName% <OnnxRuntimeRootPath>" to start giano process.

exit /b 0

:invalid_parameter
echo invalid_parameter
exit /b 1
