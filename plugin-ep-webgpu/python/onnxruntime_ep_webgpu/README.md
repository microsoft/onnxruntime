# ONNX Runtime WebGPU Plugin Execution Provider

WebGPU Execution Provider plugin for ONNX Runtime. Install alongside `onnxruntime` to enable WebGPU
acceleration.

## Prerequisites

This package provides the WebGPU plugin EP only. You must separately install an ONNX Runtime package
(e.g. `onnxruntime`) of version `@min_onnxruntime_version@` or later.

If the installed ONNX Runtime is incompatible, the plugin EP will report an error when its library is
registered.

## Installation

```bash
pip install "onnxruntime>=@min_onnxruntime_version@"
pip install onnxruntime-ep-webgpu
```

## Usage

```python
import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu_ep

# Register the plugin EP library with ONNX Runtime
ort.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())

# Discover WebGPU devices
all_devices = ort.get_ep_devices()
webgpu_devices = [d for d in all_devices if d.ep_name == webgpu_ep.get_ep_name()]

# Create a session using the WebGPU EP
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(webgpu_devices, {})
session = ort.InferenceSession("model.onnx", sess_options=sess_options)

# Run inference
output = session.run(None, {"input": input_data})
```
