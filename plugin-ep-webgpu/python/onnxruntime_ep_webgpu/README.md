# ONNX Runtime WebGPU Plugin Execution Provider

WebGPU Plugin Execution Provider for [ONNX Runtime](https://github.com/microsoft/onnxruntime).

## Prerequisites

This package provides the WebGPU plugin EP only. You must separately install an ONNX Runtime package
(e.g. `onnxruntime`) of version `@min_onnxruntime_version@` or later.

If the installed ONNX Runtime is incompatible, the plugin EP will report an error when its library is
registered.

On Linux, a system Vulkan loader (`libvulkan.so.1`) must be installed and available at runtime.

## Supported Platforms

| Platform |
|---|
| Windows x64 |
| Windows arm64 |
| Linux x64 (manylinux) |
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

# Discover WebGPU devices
all_devices = ort.get_ep_devices()
webgpu_devices = [d for d in all_devices if d.ep_name == webgpu_ep.get_ep_name()]
if not webgpu_devices:
    raise RuntimeError("No WebGPU device found.")

# Create a session using the WebGPU EP
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(webgpu_devices, {})
session = ort.InferenceSession("model.onnx", sess_options=sess_options)

# Run inference (replace shape/dtype/name to match your model)
input_data = np.zeros((1, 3, 224, 224), dtype=np.float32)
output = session.run(None, {"input": input_data})
```

## Troubleshooting

- **`No WebGPU device found`** — the plugin EP loaded but no compatible adapter was discovered. On Linux this
  usually means the Vulkan loader (`libvulkan.so.1`) is not installed; install it via your distribution's package
  manager. On Windows it may indicate a missing or outdated GPU driver.
- **`ORT runtime version "..." is below the minimum required version "@min_onnxruntime_version@"`** — the
  installed `onnxruntime` package is older than `@min_onnxruntime_version@`. Upgrade with
  `pip install --upgrade "onnxruntime>=@min_onnxruntime_version@"`.
