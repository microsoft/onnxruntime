## Microsoft.ML.OnnxRuntime.EP.WebGpu

WebGPU plugin Execution Provider for [ONNX Runtime](https://github.com/microsoft/onnxruntime).
Provides GPU acceleration via WebGPU (Dawn) with D3D12 and Vulkan backends.

### Usage

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.EP.WebGpu;

// Register the WebGPU EP plugin library
var env = OrtEnv.Instance();
env.RegisterExecutionProviderLibrary("webgpu_ep", WebGpuEp.GetLibraryPath());

// Find the WebGPU EP device
OrtEpDevice? webGpuDevice = null;
foreach (var device in env.GetEpDevices())
{
    if (string.Equals(WebGpuEp.GetEpName(), device.EpName, StringComparison.Ordinal))
    {
        webGpuDevice = device;
        break;
    }
}

// Create a session with the WebGPU EP
using var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider(env, new[] { webGpuDevice }, new Dictionary<string, string>());

using var session = new InferenceSession("model.onnx", sessionOptions);
```

### Supported Platforms

| Runtime Identifier | Native Library |
|---|---|
| win-x64 | `onnxruntime_providers_webgpu.dll`, `dxil.dll`, `dxcompiler.dll` |
| win-arm64 | `onnxruntime_providers_webgpu.dll`, `dxil.dll`, `dxcompiler.dll` |
| linux-x64 | `libonnxruntime_providers_webgpu.so` |
| osx-arm64 | `libonnxruntime_providers_webgpu.dylib` |
