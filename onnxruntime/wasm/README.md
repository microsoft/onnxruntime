# ONNX Runtime for WebAssembly

## HOW TO BUILD:

### Before build

1. Install Node.js 14.x
2. Syncup git submodules (cmake/external/emsdk)
3. Perform one-time setup (This will be implicit called by build.cmd. It takes some time to download.)
    - `emsdk install latest`
    - `emsdk activate latest`

### Building

1. Build ONNXRuntime.

    Building ONNXRuntime helps to generate `onnx-ml.pb.h` and `onnx-operators-ml.pb.h` under folder `build\Windows\{BUILD_TYPE}\external\onnx\onnx`. This file is required for building WebAssembly.

    call `build --config {BUILD_TYPE}' in root folder. Supported BUILD_TYPE are Debug and Release. Don't build as a shared lib to compile onnx protobuf properly.

2. Build WebAssembly

    mkdir build
    cd build

    Add cmake\external\emsdk\upstream\emscripten into path

    emcmake cmake -DCMAKE_BUILD_TYPE={BUILD_TYPE} -G Ninja ..
    cmake --build . --verbose

3. Build native for debugging

    mkdir build
    cd build

    cmake -DCMAKE_BUILD_TYPE={BUILD_TYPE} -DBUILD_NATIVE=1 -G "Visual Studio 16 2019" ..
    cmake --build . --verbose --config {BUILD_TYPE}

### Output

Files `onnxruntime_wasm.*` will be outputted.
