@echo off

if ["%~1"]==["--clean"] (
    if exist "%~dp0build\Windows\Release" (
        rd /s /q %~dp0build\Windows\Release
    )
)

setlocal

if exist "%~dp0build\Windows\host_protoc\Release\protoc.exe" (
    set protoc_path_flag=--path_to_protoc_exe %~dp0build\Windows\host_protoc\Release\protoc.exe
) else (
    set protoc_path_flag=
)

call .\build.bat --config Release --skip_submodule_sync --skip_tests --disable_wasm_exception_catching --disable_rtti --build_wasm --use_xnnpack --enable_wasm_simd --use_js --cmake_generator "Visual Studio 17 2022" %protoc_path_flag% --target onnxruntime_webassembly

IF %ERRORLEVEL% == 0 (
copy /Y .\build\Windows\Release\ort-wasm-simd.js .\js\web\lib\wasm\binding\
copy /Y .\build\Windows\Release\ort-wasm-simd.wasm .\js\web\dist\
)

endlocal
