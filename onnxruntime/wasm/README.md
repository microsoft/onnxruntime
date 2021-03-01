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

1. Build for onnx model

    Call `build` command in root folder. There are some restrictions to build WebAssembly.
    
    - Add '--wasm'.
    - Don't build as a shared lib to compile onnx protobuf properly.
    - Skip unit test.

    In example,

    ```cmd
    build.bat --config=Release --wasm --skip_tests
    ```

2. Reduce WebAssembly binary size and run with ort model format

    Refer to 'https://github.com/microsoft/onnxruntime/blob/master/docs/ONNX_Runtime_for_Mobile_Platforms.md' to build an ORT model and a configuration file to reduce operator kernels. This command creates a ORT model and 'required_operators.config'

    Keep the same restrictions listed at #1 above. In example,

    ```cmd
    build.bat --config=MinSizeRel --wasm --skip_tests --include_ops_by_config required_operators.config --enable_reduced_operator_type_support --minimal_build --disable_exceptions --disable_ml_ops
    ```

### Output

Files `onnxruntime_wasm.*` will be outputted.
