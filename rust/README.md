# ONNX Runtime

These are Rust bindings to
[Microsoft's ONNX Runtime](https://github.com/microsoft/onnxruntime).

This project consists of two crates:

* [`onnxruntime-sys`](onnxruntime-sys): Low-level binding to the C API;
* [`onnxruntime`](onnxruntime): High-level and safe API.

The `build.rs` script supports downloading pre-built versions of the Microsoft ONNX Runtime,
which provides the following targets:

CPU:

* Linux x86_64
* macOS x86_64
* macOS aarch64
* Windows i686
* Windows x86_64

GPU:

* Linux x86_64
* Windows x86_64

---

**WARNING**:

* This is an experiment and work in progress; it is _not_ complete/working/safe. Help welcome!
* Basic inference works, see [`onnxruntime/examples/sample.rs`](onnxruntime/examples/sample.rs) or [`onnxruntime/tests/integration_tests.rs`](onnxruntime/tests/integration_tests.rs)
* ONNX Runtime has many options to control the inference process but those options are not yet exposed.

---

## Setup

Three different strategy to obtain the ONNX Runtime are supported by the `build.rs` script:

1. Download a pre-built binary from upstream;
2. Point to a local version already installed;
3. Compile from source.

To select which strategy to use, set the `ORT_RUST_STRATEGY` environment variable to:

1. `download`: Download prebuilt onnxruntime;
2. `system`: To use a locally installed version (use `ORT_RUST_LIB_LOCATION` environment variable to point to the install path)
3. `compile`: To compile the library. This is the default.

The `download` strategy supports downloading a version of ONNXRuntime that supports CUDA. To use this, set the
environment variable `ORT_RUST_USE_CUDA=1` (only supports Linux or Windows).

### Note on 'ORT_RUST_STRATEGY=system'

When using `ORT_RUST_STRATEGY=system`, executing a built crate binary (for example the tests) might fail, at least on macOS,
if the library is not installed in a system path. An error similar to the following happens:

```text
dyld: Library not loaded: @rpath/libonnxruntime.1.7.1.dylib
  Referenced from: onnxruntime-rs.git/target/debug/deps/onnxruntime_sys-22eb0e3e89a0278c
  Reason: image not found
```

To fix, one can either:

* Set the `LD_LIBRARY_PATH` environment variable to point to the path where the library can be found.
* Adapt the `.cargo/config` file to contain a linker flag to provide the **full** path:

  ```toml
  [target.aarch64-apple-darwin]
  rustflags = ["-C", "link-args=-Wl,-rpath,/full/path/to/onnxruntime/lib"]
  ```

See [rust-lang/cargo #5077](https://github.com/rust-lang/cargo/issues/5077) for more information.

## Example

The C++ example that uses the C API
([`C_Api_Sample.cpp`](https://github.com/microsoft/onnxruntime/blob/v1.3.1/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp))
was ported to both the low level crate (`onnxruntime-sys`) and the high level on (`onnxruntime`).

### onnxruntime-sys

To run this example ([`onnxruntime-sys/examples/c_api_sample.rs`](onnxruntime-sys/examples/c_api_sample.rs)):

```sh
# Download the model (SqueezeNet 1.0, ONNX version: 1.3, Opset version: 8)
❯ curl -LO "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
❯ cargo run --example c_api_sample
[...]
    Finished dev [unoptimized + debuginfo] target(s) in 1.88s
     Running `target/debug/examples/c_api_sample`
Using Onnxruntime C API
2020-08-09 09:37:41.554922 [I:onnxruntime:, inference_session.cc:174 ConstructorCommon] Creating and using per session threadpools since use_per_session_threads_ is true
2020-08-09 09:37:41.556650 [I:onnxruntime:, inference_session.cc:830 Initialize] Initializing session.
2020-08-09 09:37:41.556665 [I:onnxruntime:, inference_session.cc:848 Initialize] Adding default CPU execution provider.
2020-08-09 09:37:41.556678 [I:onnxruntime:test, bfc_arena.cc:15 BFCArena] Creating BFCArena for Cpu
2020-08-09 09:37:41.556687 [V:onnxruntime:test, bfc_arena.cc:32 BFCArena] Creating 21 bins of max chunk size 256 to 268435456
2020-08-09 09:37:41.558313 [I:onnxruntime:, reshape_fusion.cc:37 ApplyImpl] Total fused reshape node count: 0
2020-08-09 09:37:41.559327 [I:onnxruntime:, reshape_fusion.cc:37 ApplyImpl] Total fused reshape node count: 0
2020-08-09 09:37:41.559476 [I:onnxruntime:, reshape_fusion.cc:37 ApplyImpl] Total fused reshape node count: 0
2020-08-09 09:37:41.559607 [V:onnxruntime:, inference_session.cc:671 TransformGraph] Node placements
2020-08-09 09:37:41.559615 [V:onnxruntime:, inference_session.cc:673 TransformGraph] All nodes have been placed on [CPUExecutionProvider].
2020-08-09 09:37:41.559639 [I:onnxruntime:, session_state.cc:25 SetGraph] SaveMLValueNameIndexMapping
2020-08-09 09:37:41.559787 [I:onnxruntime:, session_state.cc:70 SetGraph] Done saving OrtValue mappings.
2020-08-09 09:37:41.560252 [I:onnxruntime:, session_state_initializer.cc:178 SaveInitializedTensors] Saving initialized tensors.
2020-08-09 09:37:41.563467 [I:onnxruntime:, session_state_initializer.cc:223 SaveInitializedTensors] Done saving initialized tensors
2020-08-09 09:37:41.563979 [I:onnxruntime:, inference_session.cc:919 Initialize] Session successfully initialized.
Number of inputs = 1
Input 0 : name=data_0
Input 0 : type=1
Input 0 : num_dims=4
Input 0 : dim 0=1
Input 0 : dim 1=3
Input 0 : dim 2=224
Input 0 : dim 3=224
2020-08-09 09:37:41.573127 [I:onnxruntime:, sequential_executor.cc:145 Execute] Begin execution
2020-08-09 09:37:41.573183 [I:onnxruntime:test, bfc_arena.cc:259 AllocateRawInternal] Extending BFCArena for Cpu. bin_num:13 rounded_bytes:3154176
2020-08-09 09:37:41.573197 [I:onnxruntime:test, bfc_arena.cc:143 Extend] Extended allocation by 4194304 bytes.
2020-08-09 09:37:41.573203 [I:onnxruntime:test, bfc_arena.cc:147 Extend] Total allocated bytes: 9137152
2020-08-09 09:37:41.573212 [I:onnxruntime:test, bfc_arena.cc:150 Extend] Allocated memory at 0x7fb7d6cb7000 to 0x7fb7d70b7000
2020-08-09 09:37:41.573248 [I:onnxruntime:test, bfc_arena.cc:259 AllocateRawInternal] Extending BFCArena for Cpu. bin_num:8 rounded_bytes:65536
2020-08-09 09:37:41.573256 [I:onnxruntime:test, bfc_arena.cc:143 Extend] Extended allocation by 4194304 bytes.
2020-08-09 09:37:41.573262 [I:onnxruntime:test, bfc_arena.cc:147 Extend] Total allocated bytes: 13331456
2020-08-09 09:37:41.573268 [I:onnxruntime:test, bfc_arena.cc:150 Extend] Allocated memory at 0x7fb7d70b7000 to 0x7fb7d74b7000
Score for class [0] =  0.000045440644
Score for class [1] =  0.0038458651
Score for class [2] =  0.00012494653
Score for class [3] =  0.0011804523
Score for class [4] =  0.0013169361
Done!
```

### onnxruntime

To run this example ([`onnxruntime/examples/sample.rs`](onnxruntime/examples/sample.rs)):

```sh
# Download the model (SqueezeNet 1.0, ONNX version: 1.3, Opset version: 8)
❯ curl -LO "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-8.onnx"
❯ cargo run --example sample
[...]
    Finished dev [unoptimized + debuginfo] target(s) in 13.62s
     Running `target/debug/examples/sample`
Uninitialized environment found, initializing it with name "test".
2020-08-09 09:34:37.395577 [I:onnxruntime:, inference_session.cc:174 ConstructorCommon] Creating and using per session threadpools since use_per_session_threads_ is true
2020-08-09 09:34:37.399253 [I:onnxruntime:, inference_session.cc:830 Initialize] Initializing session.
2020-08-09 09:34:37.399284 [I:onnxruntime:, inference_session.cc:848 Initialize] Adding default CPU execution provider.
2020-08-09 09:34:37.399313 [I:onnxruntime:test, bfc_arena.cc:15 BFCArena] Creating BFCArena for Cpu
2020-08-09 09:34:37.399335 [V:onnxruntime:test, bfc_arena.cc:32 BFCArena] Creating 21 bins of max chunk size 256 to 268435456
2020-08-09 09:34:37.410516 [I:onnxruntime:, reshape_fusion.cc:37 ApplyImpl] Total fused reshape node count: 0
2020-08-09 09:34:37.417478 [I:onnxruntime:, reshape_fusion.cc:37 ApplyImpl] Total fused reshape node count: 0
2020-08-09 09:34:37.420131 [I:onnxruntime:, reshape_fusion.cc:37 ApplyImpl] Total fused reshape node count: 0
2020-08-09 09:34:37.422623 [V:onnxruntime:, inference_session.cc:671 TransformGraph] Node placements
2020-08-09 09:34:37.428863 [V:onnxruntime:, inference_session.cc:673 TransformGraph] All nodes have been placed on [CPUExecutionProvider].
2020-08-09 09:34:37.428954 [I:onnxruntime:, session_state.cc:25 SetGraph] SaveMLValueNameIndexMapping
2020-08-09 09:34:37.429079 [I:onnxruntime:, session_state.cc:70 SetGraph] Done saving OrtValue mappings.
2020-08-09 09:34:37.429925 [I:onnxruntime:, session_state_initializer.cc:178 SaveInitializedTensors] Saving initialized tensors.
2020-08-09 09:34:37.436300 [I:onnxruntime:, session_state_initializer.cc:223 SaveInitializedTensors] Done saving initialized tensors
2020-08-09 09:34:37.437255 [I:onnxruntime:, inference_session.cc:919 Initialize] Session successfully initialized.
Dropping the session options.
2020-08-09 09:34:37.448956 [I:onnxruntime:, sequential_executor.cc:145 Execute] Begin execution
2020-08-09 09:34:37.449041 [I:onnxruntime:test, bfc_arena.cc:259 AllocateRawInternal] Extending BFCArena for Cpu. bin_num:13 rounded_bytes:3154176
2020-08-09 09:34:37.449072 [I:onnxruntime:test, bfc_arena.cc:143 Extend] Extended allocation by 4194304 bytes.
2020-08-09 09:34:37.449087 [I:onnxruntime:test, bfc_arena.cc:147 Extend] Total allocated bytes: 9137152
2020-08-09 09:34:37.449104 [I:onnxruntime:test, bfc_arena.cc:150 Extend] Allocated memory at 0x7fb3b9585000 to 0x7fb3b9985000
2020-08-09 09:34:37.449176 [I:onnxruntime:test, bfc_arena.cc:259 AllocateRawInternal] Extending BFCArena for Cpu. bin_num:8 rounded_bytes:65536
2020-08-09 09:34:37.449196 [I:onnxruntime:test, bfc_arena.cc:143 Extend] Extended allocation by 4194304 bytes.
2020-08-09 09:34:37.449209 [I:onnxruntime:test, bfc_arena.cc:147 Extend] Total allocated bytes: 13331456
2020-08-09 09:34:37.449222 [I:onnxruntime:test, bfc_arena.cc:150 Extend] Allocated memory at 0x7fb3b9985000 to 0x7fb3b9d85000
Dropping Tensor.
Score for class [0] =  0.000045440578
Score for class [1] =  0.0038458686
Score for class [2] =  0.0001249467
Score for class [3] =  0.0011804511
Score for class [4] =  0.00131694
Dropping TensorFromOrt.
Dropping the session.
Dropping the memory information.
Dropping the environment.
```

See also the integration tests ([`onnxruntime/tests/integration_tests.rs`](onnxruntime/tests/integration_tests.rs))
that performs simple model download and inference, validating the results.

## License

The Rust bindings are licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or
  http://opensource.org/licenses/MIT)

at your option.
