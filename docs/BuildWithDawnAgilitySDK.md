# Build WebGPU with the D3D12 Agility SDK

Use `--use_dawn_agility_sdk` to build Dawn's D3D12 backend with ONNX Runtime's pinned preview D3D12 Agility SDK:

```powershell
python tools\ci_build\build.py `
  --build_dir build\Windows `
  --no_telemetry `
  --config Release `
  --update --build `
  --use_webgpu `
  --use_dawn_agility_sdk `
  --build_shared_lib `
  --skip_tests
```

This option is intended for local development and supports Windows desktop x86, x64, and ARM64 targets. Windows ARM32
and WindowsStore/UWP targets are not supported. Python wheels, C#, NuGet, Java, and Node.js packages are also not
supported because they do not deploy the required D3D12 runtime DLLs.

The pinned SDK is a preview release, need enable Windows Developer Mode, and install a
GPU driver that supports the experimental shader models required by the pinned Dawn revision. Feature availability
depends on the GPU and driver. At runtime, Dawn loads the SDK DLLs from the `D3D12` directory next to the executable.
Look for `[AgilitySDK] active` in the Dawn log to confirm successful activation.

For direct CMake configuration, enable `onnxruntime_USE_WEBGPU`, `onnxruntime_ENABLE_DAWN_BACKEND_D3D12`, and
`DAWN_USE_AGILITY_SDK`. The SDK is downloaded to `third_party/agility-sdk` in the Dawn source tree. When
`onnxruntime_CUSTOM_DAWN_SRC_PATH` is set, this directory is created in the specified Dawn checkout.
