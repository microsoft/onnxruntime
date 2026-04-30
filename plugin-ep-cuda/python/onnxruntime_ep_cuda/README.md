# ONNX Runtime CUDA Plugin Execution Provider

CUDA Execution Provider plugin for ONNX Runtime. Install alongside `onnxruntime` to enable the CUDA plugin EP.

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