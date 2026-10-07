# External Dawn API packages

Native WebGPU hosts can supply their own Dawn implementation while ONNX Runtime
compiles only the Dawn API dispatch layer. Set `onnxruntime_DAWN_PREBUILT_DIR` to
a directory containing matching public/generated headers and standalone proc
sources:

```text
dawn-api/
  include/
    dawn/          # Dawn public and generated headers
    webgpu/        # WebGPU C and C++ headers
  src/
    dawn_proc.cpp
    dawn_thread_dispatch_proc.cpp
```

The proc sources must compile using only these headers and the C++ standard
library. In particular, `dawn_proc.cpp` must not require private Dawn build
headers. Retain its proc-table version check. Compile the proc sources with
ONNX Runtime's toolchain rather than supplying a precompiled proc library.

```text
python tools/ci_build/build.py --config Release --use_webgpu --use_external_dawn \
  --build_shared_lib --skip_tests --target onnxruntime \
  --cmake_extra_defines onnxruntime_DAWN_PREBUILT_DIR=/absolute/path/to/dawn-api
```

This option requires `onnxruntime_USE_WEBGPU` and
`onnxruntime_USE_EXTERNAL_DAWN`. It cannot be combined with a custom Dawn source
path, shared Dawn, Emscripten, PIX, or the Agility SDK. It skips Dawn source
configuration and does not build or package native Dawn or DXC binaries.
The host owns its Dawn backend's runtime dependencies.

Pass the host's matching `DawnProcTable` address through the WebGPU
`dawnProcTable` provider option before creating a session. Headers, proc sources,
and the host implementation must come from the same Dawn version; mismatched
proc tables are not ABI-compatible.

Windows WebGPU CI exports a matching API package from the source-based external
Dawn build, then builds a separate package-backed ONNX Runtime DLL. It runs the
same native host against that DLL, verifies Abs inference with CPU fallback
disabled, and checks missing proc-table and invalid-package failures.
