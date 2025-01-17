REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

set PATH=%CD%;%PATH%
SETLOCAL EnableDelayedExpansion 
FOR /R %%i IN (*.nupkg) do (
   set filename=%%~ni
   IF NOT "!filename:~25,7!"=="Managed" (
       mkdir build\native\include
       7z a  %%~ni.nupkg build 
   )
) 
