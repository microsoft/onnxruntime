# ONNX Runtime CUDA Plugin Execution Provider

CUDA Execution Provider plugin for ONNX Runtime. Install alongside `onnxruntime` to enable the CUDA plugin EP.

## Prerequisites

This package provides the CUDA plugin EP only. You must separately install an ONNX Runtime package
(e.g. `onnxruntime`) of version `@min_onnxruntime_version@` or later.

The minimum version does not guarantee compatibility for contributed operators. If the CUDA plugin implements
contributed operators used by your model, the core and plugin must use the same contributed-operator schemas. Building
both from the same ONNX Runtime revision is recommended; mismatched schemas may cause incorrect execution or a crash,
and may not be detected when registering the plugin.

## Installation

Installing an ONNX Runtime version at or above the minimum satisfies only the core-version requirement. For models using
contributed operators implemented by the CUDA plugin, use an ONNX Runtime core built from the same revision as the
plugin.

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
