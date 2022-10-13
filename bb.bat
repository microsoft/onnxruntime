call .\build.bat --config Debug --skip_submodule_sync --skip_tests --build_wasm --use_js --cmake_generator "Visual Studio 17 2022" --target onnxruntime_webassembly

IF %ERRORLEVEL% == 0 (
copy /Y .\build\Windows\Debug\ort-wasm.js .\js\web\lib\wasm\binding\
copy /Y .\build\Windows\Debug\ort-wasm.wasm .\js\web\dist\
)
