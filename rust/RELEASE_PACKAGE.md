# Rust release bundle

This platform-specific ZIP is a local Cargo source distribution, not a crates.io
package. It contains the two first-party crates under `source/` and the matching
CPU ONNX Runtime SDK built from this source tree under `native/`.

Use the high-level crate from an ordinary Cargo project:

```toml
[dependencies]
onnxruntime = { path = "../onnxruntime-rust/source/onnxruntime" }
```

Before `cargo build`, set:

```text
ORT_RUST_STRATEGY=system
ORT_RUST_LIB_LOCATION=<absolute path>/onnxruntime-rust/native
ORT_RUST_HEADER_LOCATION=<absolute path>/onnxruntime-rust/native/include
```

At execution time, make `native/lib` discoverable (`PATH` on Windows,
`LD_LIBRARY_PATH` on Linux, or `DYLD_LIBRARY_PATH` on macOS). The native SDK is
a build and runtime dependency; Cargo does not install or download it.

For both the local bundle and the registry-format crates, set
`ORT_RUST_STRATEGY=system` and point `ORT_RUST_LIB_LOCATION` and
`ORT_RUST_HEADER_LOCATION` at the exact externally supplied SDK. With that
strategy, the release consumer never downloads a prebuilt ONNX Runtime and
never invokes the historical source-build wrapper.

`onnxruntime-sys` does not package a host-generated `bindings.rs`. Its build
script generates C bindings for the consumer target from the headers in that
exact external SDK. This avoids presenting one build machine's ABI as a
universal crate.

The archive is intentionally CPU-only and single-platform. It does not imply
crates.io publication, signing, or compatibility with a different ONNX Runtime
binary.

The portable `//rust:crates_io_artifacts` target separately produces genuine
Cargo package archives named from each manifest:
`onnxruntime-0.0.14.crate` and `onnxruntime-sys-0.0.14.crate`. Their normalized
manifests use package-local README and license files and remove the workspace
path from the inter-crate dependency. They intentionally contain no native
runtime.

These files are format-correct registry inputs, but this repository does not
publish them. In particular, `onnxruntime-sys` still supports its historical
download/compile strategies. A release publisher must decide whether those
network and legacy-build behaviors meet the registry's policy; the release
packager does not disguise them or claim that crates.io publication succeeded.

The Bazel release targets are CPU-only and become incompatible under
`--config=cuda`. CUDA publication requires a separately defined provider
payload and release contract; these targets do not imply GPU support.
