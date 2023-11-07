REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@REM set PATH=%AGENT_TEMPDIRECTORY%\v11.8\bin;%AGENT_TEMPDIRECTORY%\v11.8\extras\CUPTI\lib64;%PATH%
@REM The default version is still cuda v11.8, because set cuda v12.2 after it
set PATH=%PATH%;%AGENT_TEMPDIRECTORY%\v12.2\bin;%AGENT_TEMPDIRECTORY%\v12.2\extras\CUPTI\lib64

set GRADLE_OPTS=-Dorg.gradle.daemon=false
