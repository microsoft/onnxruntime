# ONNX Runtime WebGPU Plugin Execution Provider

WebGPU Plugin Execution Provider for [ONNX Runtime](https://github.com/microsoft/onnxruntime).

## Prerequisites

This package provides the WebGPU plugin EP only. You must separately install an ONNX Runtime package
(e.g. `onnxruntime`) of version `@min_onnxruntime_version@` or later.

If the installed ONNX Runtime is incompatible, the plugin EP will report an error when its library is
registered.

On Linux, a system Vulkan loader (`libvulkan.so.1`) must be installed and available at runtime.

On Windows, Agility-enabled builds of this package include the pinned preview D3D12 Agility SDK runtime in a `D3D12`
directory next to the WebGPU provider. Locally built wheels may use a plugin built without Agility and omit the SDK
runtime. Activation requires a supported Windows version with the Agility SDK loader update, Windows Developer Mode
because the SDK is a preview release, and a compatible GPU driver. The pinned SDK requires Windows 10 version 1909 or
newer. For Windows 10 versions 1909, 2004, and 20H2, version 1909 requires build 18363.1350 or newer, while versions
2004 and 20H2 require revision 789 or newer. Later Windows versions do not require a separate revision check. The
presence of `%SystemRoot%\System32\D3D12Core.dll` also confirms that the system has the Agility SDK loader update. For
setup details and current driver guidance, see Microsoft's
[DirectX 12 Agility SDK setup guide](https://devblogs.microsoft.com/directx/gettingstarted-dx12agility/#setup)
and the
[Agility SDK 721 Preview announcement](https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/).

## Supported Platforms

| Platform |
|---|
| Windows x64 |
| Windows arm64 |
| Linux x64 (manylinux) |
| Linux aarch64 (manylinux) |
| macOS arm64 |

## Installation

```bash
pip install "onnxruntime>=@min_onnxruntime_version@"
pip install onnxruntime-ep-webgpu
```

## Usage

```python
import numpy as np
import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu_ep

# Register the plugin EP library with ONNX Runtime
ort.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())

# Discover WebGPU EP devices
# The WebGPU EP currently accepts one EP device. Select a physical Dawn adapter independently.
webgpu_ep_device = next((d for d in ort.get_ep_devices() if d.ep_name == webgpu_ep.get_ep_name()), None)
if webgpu_ep_device is None:
    raise RuntimeError("No WebGPU EP device found.")

# Create a session using the WebGPU EP
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(
  [webgpu_ep_device],
  {"adapterIndex": "0"},
)
session = ort.InferenceSession("model.onnx", sess_options=sess_options)

# Run inference (replace shape/dtype/name to match your model)
input_data = np.zeros((1, 3, 224, 224), dtype=np.float32)
output = session.run(None, {"input": input_data})
```

`adapterIndex` is a zero-based index into Dawn's adapter enumeration for the requested backend, using the power
preference as an ordering and selection hint.
It selects a physical adapter; `deviceId` remains the ORT WebGPU context identifier and does not select a GPU.
The default context owns one device per process, so use a separate process for each adapter, or supply a custom
`webgpuInstance`/`webgpuDevice` pair for multiple devices in one process. Omit `adapterIndex` to retain Dawn's
automatic adapter selection.

## Troubleshooting

- **`No WebGPU EP device found`** — the plugin EP loaded but no compatible adapter was discovered. On Linux this
  usually means the Vulkan loader (`libvulkan.so.1`) is not installed; install it via your distribution's package
  manager. On Windows it may indicate a missing or outdated GPU driver.
- **`ORT runtime version "..." is below the minimum required version "@min_onnxruntime_version@"`** — the
  installed `onnxruntime` package is older than `@min_onnxruntime_version@`. Upgrade with
  `pip install --upgrade "onnxruntime>=@min_onnxruntime_version@"`.
