## Microsoft.ML.OnnxRuntime.EP.WebGpu

WebGPU plugin Execution Provider for [ONNX Runtime](https://github.com/microsoft/onnxruntime).

### Prerequisites

This package provides the WebGPU plugin EP only. Your project must separately reference an ONNX Runtime
core package (e.g. `Microsoft.ML.OnnxRuntime`) of version `@min_onnxruntime_version@` or later.

If the referenced ONNX Runtime is incompatible, the plugin EP will report an error when its library is
registered.

On Linux, a system Vulkan loader (`libvulkan.so.1`) must be installed and available at runtime.

On Windows, Agility-enabled builds of this package include the pinned preview D3D12 Agility SDK runtime. The
`buildTransitive` target copies the SDK DLLs into a `D3D12` directory next to the WebGPU provider after build and
publish. Activation requires a supported Windows version with the Agility SDK loader update, Windows Developer Mode
because the SDK is a preview release, and a compatible GPU driver. The pinned SDK requires Windows 10 version 1909 or
newer. For Windows 10 versions 1909, 2004, and 20H2, version 1909 requires build 18363.1350 or newer, while versions
2004 and 20H2 require revision 789 or newer. Later Windows versions do not require a separate revision check. The
presence of `%SystemRoot%\System32\D3D12Core.dll` also confirms that the system has the Agility SDK loader update. For
setup details and current driver guidance, see Microsoft's
[DirectX 12 Agility SDK setup guide](https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#setup)
and the
[Agility SDK 721 Preview announcement](https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/).

### Supported Platforms

| Runtime Identifier (RID) |
|---|
| win-x64 |
| win-arm64 |
| linux-x64 |
| linux-arm64 |
| osx-arm64 |

### Installation

```bash
dotnet add package Microsoft.ML.OnnxRuntime --version @min_onnxruntime_version@
dotnet add package Microsoft.ML.OnnxRuntime.EP.WebGpu
```

### Usage

```csharp
// Note: Error handling is omitted for brevity, except for the device-discovery check below.

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
if (webGpuDevice is null)
{
    throw new InvalidOperationException("No WebGPU device found.");
}

// Create a session with the WebGPU EP
using var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider(env, new[] { webGpuDevice }, new Dictionary<string, string>());

using var session = new InferenceSession("model.onnx", sessionOptions);
```

### Troubleshooting

- **`No WebGPU device found`** — the plugin EP loaded but no compatible adapter was discovered. On Linux this
  usually means the Vulkan loader (`libvulkan.so.1`) is not installed; install it via your distribution's package
  manager. On Windows it may indicate a missing or outdated GPU driver.
- **`ORT runtime version "..." is below the minimum required version "@min_onnxruntime_version@"`** — the
  referenced `Microsoft.ML.OnnxRuntime` package is older than `@min_onnxruntime_version@`. Upgrade with
  `dotnet add package Microsoft.ML.OnnxRuntime --version @min_onnxruntime_version@`.
