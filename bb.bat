@echo off

if ["%~1"]==["--clean"] (
    if exist "%~dp0build\Windows\Debug" (
        rd /s /q %~dp0build\Windows\Debug
    )
)

setlocal
set PATH=C:\Program Files\Git\usr\bin;%PATH%

if exist "%~dp0build\Windows\host_protoc\Release\protoc.exe" (
    set protoc_path_flag=--path_to_protoc_exe %~dp0build\Windows\host_protoc\Release\protoc.exe
) else (
    set protoc_path_flag=
)

call .\build.bat --config Debug --skip_submodule_sync --skip_tests --build_wasm --use_xnnpack --enable_wasm_simd --use_js --cmake_generator "Visual Studio 17 2022" %protoc_path_flag% --target onnxruntime_webassembly --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1

IF %ERRORLEVEL% == 0 (
copy /Y .\build\Windows\Debug\ort-wasm-simd.js .\js\web\lib\wasm\binding\ort-wasm.js
copy /Y .\build\Windows\Debug\ort-wasm-simd.wasm .\js\web\dist\
)

endlocal
