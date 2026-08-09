## Microsoft.ML.OnnxRuntime.EP.Cuda

CUDA plugin Execution Provider for [ONNX Runtime](https://github.com/microsoft/onnxruntime).

### Prerequisites

This package provides the CUDA plugin EP only. Your project must separately reference an ONNX Runtime
core package (e.g. `Microsoft.ML.OnnxRuntime`) of version `@min_onnxruntime_version@` or later.

If the referenced ONNX Runtime is incompatible, the plugin EP will report an error when its library is
registered.

### Usage

```csharp
// Note: Error handling is omitted for brevity.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.EP.Cuda;

// Register the CUDA EP plugin library
var env = OrtEnv.Instance();
env.RegisterExecutionProviderLibrary("cuda_ep", CudaEp.GetLibraryPath());

// Find the CUDA EP device
OrtEpDevice? cudaDevice = null;
foreach (var d in env.GetEpDevices())
{
    if (d.EpName == CudaEp.GetEpName())
    {
        cudaDevice = d;
        break;
    }
}

// Create a session with the CUDA EP
using var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider(env, new[] { cudaDevice }, new Dictionary<string, string>());

using var session = new InferenceSession("model.onnx", sessionOptions);
// ... run inference ...

// Unregister when done
env.UnregisterExecutionProviderLibrary("cuda_ep");
```

### Supported Platforms

| Platform | Runtime Identifier |
|---|---|
| Windows x64 | `win-x64` |
| Linux x64 | `linux-x64` |
| Linux ARM64 | `linux-arm64` |

### Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit and cuDNN installed on the system
- ONNX Runtime `@min_onnxruntime_version@` or later
