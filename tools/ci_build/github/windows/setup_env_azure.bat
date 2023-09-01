REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.
set PATH=%cd%\RelWithDebInfo\_deps\vcpkg-src\installed\x64-windows\bin;%cd%\RelWithDebInfo\_deps\vcpkg-src\installed\x86-windows\bin;%PATH%
set GRADLE_OPTS=-Dorg.gradle.daemon=false
