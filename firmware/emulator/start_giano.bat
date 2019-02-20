:: Copyright (c) Microsoft Corporation. All rights reserved.
:: Licensed under the MIT License.

:: parameter 1: firmware name
:: parameter 2: lotus root path
if "%~1"=="" goto:invalid_parameter
if "%~2"=="" goto:invalid_parameter
:: use filename parsed from full file path
set FirmwareName=%~1
set LotusRoot=%~2

echo 'Starting giano.exe.'
set GIANO_TARGET_PATH=%Giano%\build\native\tests\nios\bs3.0
echo 'Replace FPGACoreLib.dll to redirect all FPGA requests from OnnxRuntime to giano.'
copy %GIANO_TARGET_PATH%\FPGACoreLib.pdb %LotusRoot%\build\Windows\Debug\Debug\
copy %GIANO_TARGET_PATH%\FPGACoreLib.dll %LotusRoot%\build\Windows\Debug\Debug\
pushd %GIANO_TARGET_PATH%
call ..\..\..\Release\giano.exe -Platform bs.plx BrainSlice::Implementation %FirmwareName% SchemaMemory::PermanentStorage schema.bin VectorProcessor::SkuFile BrainSlice.sku.json
popd
exit /b 0

