@echo off

rem build_jsep.bat --- build onnxruntime-web with JSEP
rem
rem Usage:
rem   build_jsep.bat  config  threaded  [clean]
rem
rem Options:
rem   config      Build configuration, "d" or "r"
rem   threaded    Build with threading support, "st" or "mt"
rem   clean       Perform a clean build, "clean" or empty

setlocal enabledelayedexpansion

set ROOT=%~dp0..\

:arg1
if ["%~1"]==["d"] (
    set CONFIG=Debug
    set CONFIG_EXTRA_FLAG=--enable_wasm_debug_info
    goto :arg2
)
if ["%~1"]==["r"] (
    set CONFIG=Release
    set CONFIG_EXTRA_FLAG=--enable_wasm_api_exception_catching --disable_rtti
    goto :arg2
)
echo Invalid configuration "%~1", must be "d"(Debug) or "r"(Release)
exit /b 1

:arg2
if ["%~2"]==["st"] (
    set BUILD_DIR=%ROOT%build_jsep_st
    set THREADED_EXTRA_FLAG=
    set TARGET_FILE_PREFIX=ort-wasm-simd
    goto :arg3
)
if ["%~2"]==["mt"] (
    set BUILD_DIR=%ROOT%build_jsep_mt
    set THREADED_EXTRA_FLAG=--enable_wasm_threads
    set TARGET_FILE_PREFIX=ort-wasm-simd-threaded
    goto :arg3
)
echo Invalid threading option "%~2", must be "st" or "mt"
exit /b 1

:arg3
if ["%~3"]==["clean"] (
    goto :clean
)
if not exist "%ROOT%js\web\dist" (
    goto :npm_ci
)

goto :build_wasm

:clean
if exist "%BUILD_DIR%" (
    rd /s /q %BUILD_DIR%
)

pushd %ROOT%
git submodule sync --recursive
git submodule update --init --recursive
popd

:npm_ci
pushd %ROOT%js
call npm ci
popd
pushd %ROOT%js\common
call npm ci
popd
pushd %ROOT%js\web
call npm ci
call npm run pull:wasm
popd

:build_wasm

set PATH=C:\Program Files\Git\usr\bin;%PATH%

call %ROOT%build.bat --config %CONFIG% %CONFIG_EXTRA_FLAG% %THREADED_EXTRA_FLAG%^
 --skip_submodule_sync --build_wasm --skip_tests --enable_wasm_simd --use_jsep --target onnxruntime_webassembly --build_dir %BUILD_DIR%

IF NOT "%ERRORLEVEL%" == "0" (
  exit /b %ERRORLEVEL%
)

copy /Y %BUILD_DIR%\%CONFIG%\%TARGET_FILE_PREFIX%.js %ROOT%js\web\lib\wasm\binding\%TARGET_FILE_PREFIX%.jsep.js
copy /Y %BUILD_DIR%\%CONFIG%\%TARGET_FILE_PREFIX%.wasm %ROOT%js\web\dist\%TARGET_FILE_PREFIX%.jsep.wasm
