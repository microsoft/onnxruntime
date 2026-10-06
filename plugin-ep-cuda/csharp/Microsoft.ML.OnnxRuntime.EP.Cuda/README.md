## ONNX Runtime CUDA Plugin EP

CUDA plugin Execution Provider for [ONNX Runtime](https://github.com/microsoft/onnxruntime).

The plugin EP ships as one package per RID and per CUDA major version, for example
`Microsoft.ML.OnnxRuntime.EP.Cuda13.win-x64` and `Microsoft.ML.OnnxRuntime.EP.Cuda13.linux-arm64`.

### Prerequisites

This package provides the CUDA plugin EP only. Your project must separately reference an ONNX Runtime
core package (e.g. `Microsoft.ML.OnnxRuntime`) of version `@min_onnxruntime_version@` or later.

The minimum version does not guarantee compatibility for contributed operators. If the CUDA plugin implements
contributed operators used by your model, the core and plugin must use the same contributed-operator schemas. Building
both from the same ONNX Runtime revision is recommended; mismatched schemas may cause incorrect execution or a crash,
and may not be detected when registering the plugin.

Referencing an ONNX Runtime version at or above the minimum satisfies only the core-version requirement. For models
using contributed operators implemented by the CUDA plugin, use an ONNX Runtime core built from the same revision as
the plugin.

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

| Platform | Runtime Identifier | Package |
|---|---|---|
| Windows x64 | `win-x64` | `Microsoft.ML.OnnxRuntime.EP.Cuda<major>.win-x64` |
| Windows ARM64 | `win-arm64` | `Microsoft.ML.OnnxRuntime.EP.Cuda<major>.win-arm64` |
| Linux x64 | `linux-x64` | `Microsoft.ML.OnnxRuntime.EP.Cuda<major>.linux-x64` |
| Linux ARM64 | `linux-arm64` | `Microsoft.ML.OnnxRuntime.EP.Cuda<major>.linux-arm64` |

### Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit and cuDNN installed on the system
- ONNX Runtime `@min_onnxruntime_version@` or later
