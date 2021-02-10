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

    Building ONNXRuntime helps to generate `onnx-ml.pb.h` and `onnx-operators-ml.pb.h` under folder `build\Windows\{BUILD_TYPE}\onnx`. This file is required for building WebAssembly.

    call `build --config {BUILD_TYPE} --minimal_build` in root folder. Supported BUILD_TYPE are Debug and Release.

2. Build WebAssembly

    mkdir build
    cd build

    Add cmake\external\emsdk\upstream\emscripten into path

    emcmake cmake -DCMAKE_BUILD_TYPE={BUILD_TYPE} -G Ninja ..
    cmake --build . --verbose

### Output

Files `onnxruntime_wasm.*` will be outputted.
