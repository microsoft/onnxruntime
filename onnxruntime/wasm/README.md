# ONNX Runtime for WebAssembly

## HOW TO BUILD:

### Before build

1. Install Node.js 14.x
2. Syncup git submodules (cmake/external/emsdk)
3. Perform one-time setup (This will be implicit called by build.cmd. It takes some time to download.)

    ```cmd
    emsdk install latest
    emsdk activate latest
    ```

### Building

1. Build ONNXRuntime.

    Building ONNXRuntime helps to generate `onnx-ml.pb.h` and `onnx-operators-ml.pb.h` under folder `build/{BUILD_PLATFORM}/{BUILD_TYPE}/external/onnx/onnx`. This file is required for building WebAssembly.

    Call `build --config {BUILD_TYPE}' in root folder. Supported BUILD_TYPE are Debug and Release. Don't build as a shared lib to compile onnx protobuf properly.

2. Build WebAssembly

    ```cmd
    mkdir build
    cd build

    Add cmake/external/emsdk/upstream/emscripten into path

    emcmake cmake -DCMAKE_BUILD_TYPE={BUILD_TYPE} -G Ninja ..
    cmake --build . --verbose
    ```

3. Build native for debugging

    ```cmd
    mkdir build
    cd build

    cmake -DCMAKE_BUILD_TYPE={BUILD_TYPE} -DBUILD_NATIVE=1 -G "Visual Studio 16 2019" ..
    cmake --build . --verbose --config {BUILD_TYPE}
    ```

4. Reduce WebAssembly binary size and run with ort format

    Refer to 'https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_for_Mobile_Platforms.md' to build an ORT model and a configuration file to reduce operator kernels. This command creates a ORT model and 'required_operators.config'

    ```cmd
    python tools/python/convert_onnx_models_to_ort.py model_path
    ```

    To comment out all unnecessary operator kernels from repository, run

    ```cmd
    python ../../tools/ci_build/exclude_unused_ops_and_types.py required_operators.config
    ```

    Build WebAssembly with 'MinSizeRel' build type and '-DBUILD_MINIMAL' macro

    ```cmd
    emcmake cmake -DCMAKE_BUILD_TYPE=MinSizeRel -DBUILD_MINIMAL -G Ninja ..
    cmake --build . --verbose
    ```

### Output

Files `onnxruntime_wasm.*` will be outputted.
