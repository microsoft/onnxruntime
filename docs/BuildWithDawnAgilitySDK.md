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

This option is intended for local development and supports Windows desktop x86, x64, and ARM64 targets. Windows ARM32,
ARM64EC, and WindowsStore/UWP targets are not supported. Python wheels, C#, NuGet, Java, and Node.js packages are also
not supported because they do not deploy the required D3D12 runtime DLLs. Custom Dawn checkouts selected with
`onnxruntime_CUSTOM_DAWN_SRC_PATH` are not supported.

The pinned SDK requires Windows 10 version 1909 or newer. For versions 1909, 2004, and 20H2, the minimum OS build
revisions are:

- Version 1909: revision 1350 or newer (build 18363.1350).
- Versions 2004 and 20H2: revision 789 or newer.

Later Windows versions do not require a separate revision check. The presence of `%SystemRoot%\System32\D3D12Core.dll`
also confirms that the system has the Agility SDK loader update. See Microsoft's
[Agility SDK setup guide](https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#setup) for details.

Because the pinned SDK is a preview release, Windows Developer Mode must be enabled. Install a compatible GPU driver
and consult Microsoft's
[AgilitySDK 721 Preview announcement](https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/)
for the current per-vendor driver links and feature-support table. Feature availability depends on the GPU and driver.
At runtime, Dawn loads the SDK DLLs from the `D3D12` directory next to the executable. Look for `[AgilitySDK] active`
in the Dawn log to confirm successful activation.

For direct CMake configuration, enable `onnxruntime_USE_WEBGPU`, `onnxruntime_ENABLE_DAWN_BACKEND_D3D12`, and
`DAWN_USE_AGILITY_SDK`. The SDK is downloaded to `third_party/agility-sdk` in the Dawn source tree managed by ONNX
Runtime.
