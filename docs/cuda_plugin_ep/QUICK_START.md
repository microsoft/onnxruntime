# CUDA Plugin EP Quick Start

## Build Instructions

To build ONNX Runtime with the CUDA Plugin Execution Provider instead of the statically linked CUDA EP, use the `--build_cuda_ep_as_plugin` flag with the build script.

```bash
# Build the core framework and the CUDA Plugin EP
./build.sh --config RelWithDebInfo --build_shared_lib --use_cuda --build_cuda_ep_as_plugin
```

## Running

When the plugin is built, it will produce `libonnxruntime_providers_cuda_plugin.so` (or `.dll` on Windows) in the build output directory alongside `libonnxruntime.so`.

The plugin EP is registered under the name **`CudaPluginExecutionProvider`** and uses the EP Plugin API (`RegisterExecutionProviderLibrary` / `GetEpDevices` / `SessionOptionsAppendExecutionProvider_V2`). It is **not** a drop-in replacement for the in-tree `CUDAExecutionProvider` — you must register the plugin library, enumerate its devices, and add them to the session.

### C++ API

Use `Env::RegisterExecutionProviderLibrary` to load the plugin, `Env::GetEpDevices` to discover the CUDA devices it exposes, and `SessionOptions::AppendExecutionProvider_V2` to add the selected device to the session.

```cpp
#include "onnxruntime_cxx_api.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PluginTest");

// 1. Register the plugin library.
env.RegisterExecutionProviderLibrary("CudaPluginExecutionProvider",
                                     ORT_TSTR("libonnxruntime_providers_cuda_plugin.so"));

// 2. Enumerate available EP devices and pick the CUDA plugin device.
auto ep_devices = env.GetEpDevices();
std::vector<Ort::ConstEpDevice> plugin_devices;
for (const auto& dev : ep_devices) {
  if (std::string(dev.EpName()) == "CudaPluginExecutionProvider") {
    plugin_devices.push_back(dev);
    break;  // use the first CUDA plugin device
  }
}

// 3. Add the plugin device to session options.
Ort::SessionOptions session_options;
session_options.AppendExecutionProvider_V2(env, plugin_devices, {});

Ort::Session session(env, "model.onnx", session_options);
```

### Python API

Use `onnxruntime.register_execution_provider_library` to load the plugin, `onnxruntime.get_ep_devices` to discover devices, and `SessionOptions.add_provider_for_devices` to add the selected device.

**Device-based approach (recommended):**

```python
import onnxruntime as ort

# 1. Register the plugin library.
ort.register_execution_provider_library(
    "CudaPluginExecutionProvider",
    "libonnxruntime_providers_cuda_plugin.so",
)

# 2. Enumerate devices and pick the CUDA plugin device.
devices = ort.get_ep_devices()
plugin_device = next(d for d in devices if d.ep_name == "CudaPluginExecutionProvider")

# 3. Create session with the plugin device.
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices([plugin_device], {})

sess = ort.InferenceSession("model.onnx", sess_options=sess_options)
```

**Provider-name approach:**

You can also pass `CudaPluginExecutionProvider` by name in the `providers` list
(the plugin library must already be registered):

```python
import onnxruntime as ort

ort.register_execution_provider_library(
    "CudaPluginExecutionProvider",
    "libonnxruntime_providers_cuda_plugin.so",
)

sess = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("CudaPluginExecutionProvider", {"device_id": "0"}),
        "CPUExecutionProvider",
    ],
)
```

## Known Limitations
* The plugin does not currently support CUDA Graphs.
* The plugin direct-allocates memory using `cudaMalloc` resulting in a potential performance penalty compared to the integrated Memory Arena.

## Verification
You can generate a parity report comparing the kernels available in the plugin EP versus the statically linked CUDA EP.
```bash
# Check static source registration parity:
python tools/ci_build/cuda_plugin_parity_report.py

# Check runtime registry parity:
python tools/ci_build/cuda_plugin_parity_report.py --runtime --plugin-ep-lib build/Linux/RelWithDebInfo/libonnxruntime_providers_cuda_plugin.so
```
