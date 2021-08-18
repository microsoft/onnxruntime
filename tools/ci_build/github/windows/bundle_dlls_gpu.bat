REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

REM for available runtime identifiers, see https://github.com/dotnet/corefx/blob/release/3.1/pkg/Microsoft.NETCore.Platforms/runtime.json
set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion 
FOR /R %%i IN (*.nupkg) do (
   set filename=%%~ni
   IF NOT "!filename:~25,7!"=="Managed" (
       mkdir runtimes\linux-x64\native
       move onnxruntime-linux-x64\lib\libonnxruntime.so.1* runtimes\linux-x64\native\libonnxruntime.so
       move onnxruntime-linux-x64\lib\libonnxruntime_providers_* runtimes\linux-x64\native
       7z a  %%~ni.nupkg runtimes
   )
)