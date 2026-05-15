## Microsoft.ML.OnnxRuntime.EP.Cuda

CUDA plugin Execution Provider for [ONNX Runtime](https://github.com/microsoft/onnxruntime).

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
- ONNX Runtime 1.26.0 or later
