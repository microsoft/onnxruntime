## Microsoft.ML.OnnxRuntime.EP.WebGpu

WebGPU plugin Execution Provider for [ONNX Runtime](https://github.com/microsoft/onnxruntime).

### Prerequisites

This package provides the WebGPU plugin EP only. Your project must separately reference an ONNX Runtime
core package (e.g. `Microsoft.ML.OnnxRuntime`) of version `@min_onnxruntime_version@` or later.

If the referenced ONNX Runtime is incompatible, the plugin EP will report an error when its library is
registered.

### Usage

```csharp
// Note: Error handling is omitted for brevity.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.EP.WebGpu;

// Register the WebGPU EP plugin library
var env = OrtEnv.Instance();
env.RegisterExecutionProviderLibrary("webgpu_ep", WebGpuEp.GetLibraryPath());

// Find the WebGPU EP device
OrtEpDevice? webGpuDevice = null;
foreach (var d in env.GetEpDevices())
{
    if (d.EpName == WebGpuEp.GetEpName())
    {
        webGpuDevice = d;
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
