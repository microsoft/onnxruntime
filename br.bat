call .\build.bat --config Release --skip_submodule_sync --skip_tests --disable_wasm_exception_catching --disable_rtti --build_wasm --use_js --cmake_generator "Visual Studio 17 2022" --target onnxruntime_webassembly

IF %ERRORLEVEL% == 0 (
copy /Y .\build\Windows\Release\ort-wasm.js .\js\web\lib\wasm\binding\
copy /Y .\build\Windows\Release\ort-wasm.wasm .\js\web\dist\
)
