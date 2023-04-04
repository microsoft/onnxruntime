<div align=center>
	<h1><code>ort</code> - ONNX Runtime Rust bindings</h1>
    <a href="https://app.codecov.io/gh/pykeio/ort" target="_blank"><img alt="Coverage Results" src="https://img.shields.io/codecov/c/gh/pykeio/ort?style=for-the-badge"></a> <a href="https://github.com/pykeio/ort/actions/workflows/test.yml"><img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/pykeio/ort/test.yml?branch=main&style=for-the-badge"></a><a href="https://crates.io/crates/ort" target="_blank"><img alt="Crates.io" src="https://img.shields.io/crates/d/ort?style=for-the-badge"></a>
</div>

`ort` is an ONNX Runtime wrapper for Rust based on [`onnxruntime-rs`](https://github.com/nbigaouette/onnxruntime-rs). `ort` is updated for ONNX Runtime 1.13.1 and contains many API improvements & fixes.

See [the docs](https://docs.rs/ort) and [`examples/`](https://github.com/pykeio/ort/tree/main/examples) for more detailed information.

## Cargo features
- **`fetch-models`**: Enables fetching models from the ONNX Model Zoo.
- **`generate-bindings`**: Update/generate ONNX Runtime bindings with `bindgen`. Requires [libclang](https://clang.llvm.org/doxygen/group__CINDEX.html).
- **`copy-dylibs`**: Copy dynamic libraries to the Cargo `target` folder.
- **`half`**: Builds support for `float16`/`bfloat16` ONNX tensors.
- **Execution providers**: These are required to use some execution providers. If you are using an execution provider not provided for your platform by the `download` strategy, you must use the `compile` or `system` strategies with binaries that support those execution providers, otherwise you'll run into linking errors.
    - Some EPs are not currently implemented due to a lack of hardware for testing. Please open an issue if your desired EP has a ⚠️
    - **`cuda`**: Enables the CUDA execution provider for Maxwell (7xx) NVIDIA GPUs and above. Requires CUDA v11.6+.
    - **`tensorrt`**: Enables the TensorRT execution provider for GeForce 9xx series NVIDIA GPUs and above; requires CUDA v11.4+ and TensorRT v8.4+.
    - ⚠️ **`openvino`**: Enables the OpenVINO execution provider for 6th+ generation Intel Core CPUs.
    - **`onednn`**: Enables the Intel oneDNN execution provider for x86/x64 targets.
    - **`directml`**: Enables the DirectML execution provider for Windows x86/x64 targets with dedicated GPUs supporting DirectX 12.
    - ⚠️ **`snpe`**: Enables the SNPE execution provider for Qualcomm Snapdragon CPUs & Adreno GPUs.
    - ⚠️ **`nnapi`**: Enables the Android Neural Networks API (NNAPI) execution provider.
    - **`coreml`**: Enables the CoreML execution provider for macOS/iOS targets.
    - ⚠️ **`xnnpack`**: Enables the [XNNPACK](https://github.com/google/XNNPACK) backend for WebAssembly and Android.
    - ⚠️ **`rocm`**: Enables the ROCm execution provider for AMD ROCm-enabled GPUs.
    - **`acl`**: Enables the ARM Compute Library execution provider for multi-core ARM v8 processors.
    - ⚠️ **`armnn`**: Enables the ArmNN execution provider for ARM v8 targets.
    - ⚠️ **`tvm`**: Enables the **preview** Apache TVM execution provider.
    - ⚠️ **`migraphx`**: Enables the MIGraphX execution provider for Windows x86/x64 targets with dedicated AMD GPUs.
    - ⚠️ **`rknpu`**: Enables the RKNPU execution provider for Rockchip NPUs.
    - ⚠️ **`vitis`**: Enables Xilinx's Vitis-AI execution provider for U200/U250 accelerators.
    - ⚠️ **`cann`**: Enables the Huawei Compute Architecture for Neural Networks (CANN) execution provider.
- **Compile strategy features** - *These features only apply when using the compile [strategy](#strategies).*
    - **`compile-static`**: Compiles ONNX Runtime as a static library.
    - **`mimalloc`**: Uses the (usually) faster mimalloc memory allocation library instead of the platform default.
    - **`experimental`**: Compiles Microsoft experimental operators.
    - **`minimal-build`**: Builds ONNX Runtime without RTTI, `.onnx` model format support, runtime optimizations, or dynamically-registered EP kernels. Drastically reduces binary size, recommended for release builds (if possible).

## Strategies
There are 3 'strategies' for obtaining and linking ONNX Runtime binaries. The strategy can be set with the `ORT_STRATEGY` environment variable.
- **`download` (default)**: Downloads prebuilt ONNX Runtime from Microsoft. These binaries [may collect telemetry](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md).
- **`system`**: Links to ONNX Runtime binaries provided by the system or a path pointed to by the `ORT_LIB_LOCATION` environment variable. `ort` will automatically link to static or dynamic libraries depending on what is available in the `ORT_LIB_LOCATION` folder.
- **`compile`**: Clones & compiles ONNX Runtime from source. This is **extremely slow**! It's recommended to use `system` instead.

## Execution providers
To use other execution providers, you must explicitly enable them via their Cargo features. Using the `compile` [strategy](#strategies), everything should just work™️. If using the `system` strategy, ensure that the binaries you are linking to have been built with the execution providers you want to use, otherwise you may get linking errors. Configuring & enabling execution providers can be done through `SessionBuilder::execution_providers()`.

Execution providers will attempt to be registered in the order they are passed, silently falling back to the CPU provider if none of the requested providers are available. If you must know whether an EP is available, you can use `ExecutionProvider::cuda().is_available()`.

For prebuilt Microsoft binaries, you can enable the CUDA or TensorRT execution providers for Windows and Linux via the `cuda` and `tensorrt` Cargo features respectively. Microsoft does not provide prebuilt binaries for other execution providers, and thus enabling other EP features will fail when `ORT_STRATEGY=download`. To use other execution providers, you must build ONNX Runtime from source.

## Shared library hell
If using shared libraries (as is the default with `ORT_STRATEGY=download`), you may need to make some changes to avoid issues with library paths and load orders.

### Windows
Some versions of Windows come bundled with an older vesrion of `onnxruntime.dll` in the System32 folder, which will cause an assertion error at runtime:
```
The given version [13] is not supported, only version 1 to 10 is supported in this build.
thread 'main' panicked at 'assertion failed: `(left != right)`
  left: `0x0`,
 right: `0x0`', src\lib.rs:50:5
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

The fix is to copy the ONNX Runtime DLLs into the same directory as the binary. `ort` can automatically copy the DLLs to the Cargo target folder when the `copy-dylibs` feature is enabled, though this only fixes *binary* Cargo targets. When running tests/benchmarks/examples for the first time, you'll have to manually copy the `target/debug/onnxruntime*.dll` files to `target/debug/deps/` for tests & benchmarks or `target/debug/examples/` for examples.

### Linux
Running a binary via `cargo run` should work without `copy-dylibs`. If you'd like to use the produced binaries outside of Cargo, you'll either have to copy `libonnxruntime.so` to a known lib location (e.g. `/usr/lib`) or enable rpath to load libraries from the same folder as the binary and place `libonnxruntime.so` alongside your binary.

In `Cargo.toml`:
```toml
[profile.dev]
rpath = true

[profile.release]
rpath = true

# do this for all profiles
```

In `.cargo/config.toml`:
```toml
[target.x86_64-unknown-linux-gnu]
rustflags = [ "-Clink-args=-Wl,-rpath,\\$ORIGIN" ]

# do this for all Linux targets as well
```

### macOS
macOS has the same limitations as Linux. If enabling rpath, note that the rpath should point to `@loader_path` rather than `$ORIGIN`:

```toml
# .cargo/config.toml
[target.x86_64-apple-darwin]
rustflags = [ "-Clink-args=-Wl,-rpath,@loader_path" ]
```
