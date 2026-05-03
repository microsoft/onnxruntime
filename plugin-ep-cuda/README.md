# CUDA Plugin Execution Provider

Packaging sources for the ONNX Runtime CUDA plugin Execution Provider (EP), distributed as a standalone artifact that
plugs into an existing ONNX Runtime installation rather than being built into the main `onnxruntime` binary.

For more information about plugin EPs, see the documentation
[here](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/).

## Contents

- [`MIN_ONNXRUNTIME_VERSION`](MIN_ONNXRUNTIME_VERSION) - Minimum compatible ONNX Runtime version for the Python package.
- [`python/`](python/) - Sources and build script for the `onnxruntime-ep-cuda12`/`onnxruntime-ep-cuda13` Python wheels.

## Usage

Install the CUDA-family-specific Python distribution, then register the plugin EP at runtime. The package names are
`onnxruntime-ep-cuda12` for CUDA 12.x builds and `onnxruntime-ep-cuda13` for CUDA 13.x builds. Both distributions expose
the same Python import module, `onnxruntime_ep_cuda`.

```python
import onnxruntime as ort
import onnxruntime_ep_cuda as cuda_ep

ort.register_execution_provider_library(cuda_ep.get_ep_name(), cuda_ep.get_library_path())

devices = [d for d in ort.get_ep_devices() if d.ep_name == cuda_ep.get_ep_name()]
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(devices, {})
session = ort.InferenceSession("model.onnx", sess_options=sess_options)
```
