REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM for available runtime identifiers, see https://github.com/dotnet/corefx/blob/release/3.1/pkg/Microsoft.NETCore.Platforms/runtime.json
set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion 
FOR /R %%i IN (*.zip) do (
   set filename=%%~ni
   IF !filename!=="onnxruntime-win-tensorrt-x64-*" (
       mkdir onnxruntime-win-gpu-x64\lib 
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime_providers_tensorrt.dll onnxruntime-win-gpu-x64\lib\onnxruntime_providers_tensorrt.dll
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime_providers_shared.dll onnxruntime-win-gpu-x64\lib\onnxruntime_providers_shared.dll
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime.dll onnxruntime-win-gpu-x64\lib\onnxruntime.dll
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime.lib onnxruntime-win-gpu-x64\lib\onnxruntime.lib
       move onnxruntime-win-tensorrt-x64\lib\onnxruntime.pdb onnxruntime-win-gpu-x64\lib\onnxruntime.pdb
       7z a  %%~ni.zip onnxruntime-win-gpu-x64 
   )
) 
