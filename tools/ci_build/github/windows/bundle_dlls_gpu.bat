REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion
FOR /R %%i IN (*.zip) do (
   set filename=%%~ni
   IF "!filename:~20,3!"=="gpu" (
       mkdir !filename!\lib
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime_providers_tensorrt.dll !filename!\lib\onnxruntime_providers_tensorrt.dll
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime_providers_tensorrt.lib !filename!\lib\onnxruntime_providers_tensorrt.lib
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime_providers_tensorrt.pdb !filename!\lib\onnxruntime_providers_tensorrt.pdb
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime_providers_shared.dll !filename!\lib\onnxruntime_providers_shared.dll
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime_providers_shared.lib !filename!\lib\onnxruntime_providers_shared.lib
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime_providers_shared.pdb !filename!\lib\onnxruntime_providers_shared.pdb
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime.dll !filename!\lib\onnxruntime.dll
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime.lib !filename!\lib\onnxruntime.lib
       move /y onnxruntime-win-x64-tensorrt\lib\onnxruntime.pdb !filename!\lib\onnxruntime.pdb
       7z a  %%~ni.zip !filename!\lib
   )
)
