@echo off

rem build_jsep.bat --- build onnxruntime-web with JSEP
rem
rem Usage:
rem   build_jsep.bat  config  [clean]
rem
rem Options:
rem   config      Build configuration, "d" or "r"
rem   clean       Perform a clean build, "clean" or empty

setlocal enabledelayedexpansion

set ROOT=%~dp0..\
set BUILD_DIR=%ROOT%build_jsep

:arg1
if ["%~1"]==["d"] (
    set CONFIG=Debug
    set CONFIG_EXTRA_FLAG=--enable_wasm_debug_info --enable_wasm_profiling --cmake_extra_defines onnxruntime_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL=1
    goto :arg2
)
if ["%~1"]==["r"] (
    set CONFIG=Release
    set CONFIG_EXTRA_FLAG=--enable_wasm_api_exception_catching --disable_rtti --enable_wasm_profiling
    goto :arg2
)
echo Invalid configuration "%~1", must be "d"(Debug) or "r"(Release)
exit /b 1

:arg2
if ["%~2"]==["clean"] (
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

call %ROOT%build.bat --config %CONFIG% %CONFIG_EXTRA_FLAG% --skip_submodule_sync --build_wasm --skip_tests^
 --enable_wasm_simd --enable_wasm_threads --use_jsep --use_webnn --target onnxruntime_webassembly --build_dir %BUILD_DIR%

IF NOT "%ERRORLEVEL%" == "0" (
  exit /b %ERRORLEVEL%
)

copy /Y %BUILD_DIR%\%CONFIG%\ort-wasm-simd-threaded.jsep.wasm %ROOT%js\web\dist\
copy /Y %BUILD_DIR%\%CONFIG%\ort-wasm-simd-threaded.jsep.mjs %ROOT%js\web\dist\
