REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM for available runtime identifiers, see https://github.com/dotnet/corefx/blob/release/3.1/pkg/Microsoft.NETCore.Platforms/runtime.json
set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion
FOR /R %%i IN (*.zip) do (
   set filename=%%~ni
   IF "!filename:~16,3!"=="gpu" (
       mkdir !filename!\lib
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime_providers_tensorrt.dll !filename!\lib\onnxruntime_providers_tensorrt.dll
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime_providers_tensorrt.lib !filename!\lib\onnxruntime_providers_tensorrt.lib
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime_providers_tensorrt.pdb !filename!\lib\onnxruntime_providers_tensorrt.pdb
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime_providers_shared.dll !filename!\lib\onnxruntime_providers_shared.dll
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime_providers_shared.lib !filename!\lib\onnxruntime_providers_shared.lib
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime_providers_shared.pdb !filename!\lib\onnxruntime_providers_shared.pdb
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime.dll !filename!\lib\onnxruntime.dll
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime.lib !filename!\lib\onnxruntime.lib
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime.pdb !filename!\lib\onnxruntime.pdb
       C:\ProgramData\chocolatey\tools\7z.exe a  %%~ni.zip !filename!\lib
   )
)
