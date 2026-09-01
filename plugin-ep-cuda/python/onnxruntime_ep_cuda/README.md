# ONNX Runtime CUDA Plugin Execution Provider

CUDA Execution Provider plugin for ONNX Runtime. Install alongside `onnxruntime` to enable the CUDA plugin EP.

## Prerequisites

This package provides the CUDA plugin EP only. You must separately install an ONNX Runtime package
(e.g. `onnxruntime`) of version `@min_onnxruntime_version@` or later.

If the installed ONNX Runtime is incompatible, the plugin EP will report an error when its library is
registered.

## Installation

```bash
pip install "onnxruntime>=@min_onnxruntime_version@"
pip install onnxruntime-ep-cuda12  # or onnxruntime-ep-cuda13 for CUDA 13.x
```

## Usage

```python
import onnxruntime as ort
import onnxruntime_ep_cuda as cuda_ep

ort.register_execution_provider_library(cuda_ep.get_ep_name(), cuda_ep.get_library_path())

devices = [d for d in ort.get_ep_devices() if d.ep_name == cuda_ep.get_ep_name()]
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(devices, {})
session = ort.InferenceSession("model.onnx", sess_options=sess_options)
```
