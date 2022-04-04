mkdir js\web\dist
cmd /c build.bat --config Debug --build_wasm --skip_tests --emsdk_version releases-upstream-823d37b15d1ab61bc9ac0665ceef6951d3703842-64bit --build_dir=build\wasm
copy /Y build\wasm\Debug\ort-wasm.wasm js\web\dist\
copy /Y build\wasm\Debug\ort-wasm.js js\web\dist\
copy /Y build\wasm\Debug\ort-wasm.js js\web\lib\wasm\binding\
cmd /c build.bat --config Debug --build_wasm --skip_tests --enable_wasm_simd --emsdk_version releases-upstream-823d37b15d1ab61bc9ac0665ceef6951d3703842-64bit --build_dir=build\wasmsimd
copy /Y build\wasmsimd\Debug\ort-wasm-simd.wasm js\web\dist\
cmd /c build.bat --config Debug --build_wasm --skip_tests --enable_wasm_threads --emsdk_version releases-upstream-823d37b15d1ab61bc9ac0665ceef6951d3703842-64bit --build_dir=build\wasmthreads
copy /Y build\wasmthreads\Debug\ort-wasm-threaded.wasm js\web\dist\
copy /Y build\wasmthreads\Debug\ort-wasm-threaded.js js\web\dist\
copy /Y build\wasmthreads\Debug\ort-wasm-threaded.js js\web\lib\wasm\binding\
copy /Y build\wasmthreads\Debug\ort-wasm-threaded.worker.js js\web\dist\
copy /Y build\wasmthreads\Debug\ort-wasm-threaded.worker.js js\web\lib\wasm\binding\
cmd /c build.bat --config Debug --build_wasm --skip_tests --enable_wasm_threads --enable_wasm_simd --emsdk_version releases-upstream-823d37b15d1ab61bc9ac0665ceef6951d3703842-64bit --build_dir=build\wasmsimdthreads
copy /Y build\wasmsimdthreads\Debug\ort-wasm-simd-threaded.wasm js\web\dist\