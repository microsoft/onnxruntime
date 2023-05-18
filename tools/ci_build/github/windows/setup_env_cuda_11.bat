REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

set PATH=%AGENT_TEMPDIRECTORY%\v11.8\bin;%AGENT_TEMPDIRECTORY%\v11.8\extras\CUPTI\lib64;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
