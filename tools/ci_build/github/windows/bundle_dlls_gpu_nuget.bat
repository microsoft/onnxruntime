REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM for available runtime identifiers, see https://github.com/dotnet/corefx/blob/release/3.1/pkg/Microsoft.NETCore.Platforms/runtime.json
set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion 
FOR /R %%i IN (*.nupkg) do (
   set filename=%%~ni
   IF NOT "!filename:~25,7!"=="Managed" (
       mkdir runtimes\linux-x64\native
       move onnxruntime-linux-x64-gpu\lib\libonnxruntime_providers_* runtimes\linux-x64\native
       move onnxruntime-linux-x64-tensorrt\lib\libonnxruntime.so.1* runtimes\linux-x64\native\libonnxruntime.so
       move onnxruntime-linux-x64-tensorrt\lib\libonnxruntime_providers_shared.so runtimes\linux-x64\native\libonnxruntime_providers_shared.so
       move onnxruntime-linux-x64-tensorrt\lib\libonnxruntime_providers_tensorrt.so runtimes\linux-x64\native\libonnxruntime_providers_tensorrt.so
       mkdir runtimes\win-x64\native
       move onnxruntime-win-x64-tensorrt\lib\onnxruntime_providers_tensorrt.dll runtimes\win-x64\native\onnxruntime_providers_tensorrt.dll
       move onnxruntime-win-x64-tensorrt\lib\onnxruntime_providers_shared.dll runtimes\win-x64\native\onnxruntime_providers_shared.dll
       move onnxruntime-win-x64-tensorrt\lib\onnxruntime.dll runtimes\win-x64\native\onnxruntime.dll
       move onnxruntime-win-x64-tensorrt\lib\onnxruntime.lib runtimes\win-x64\native\onnxruntime.lib
       move onnxruntime-win-x64-tensorrt\lib\onnxruntime.pdb runtimes\win-x64\native\onnxruntime.pdb
       7z a  %%~ni.nupkg runtimes
   )
) 
