# CUDA Plugin EP Quick Start

## Build Instructions

To build ONNX Runtime with the CUDA Plugin Execution Provider instead of the statically linked CUDA EP, use the `--cmake_extra_defines "onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON"` flag with the build script.

Example command to build the CUDA Plugin EP in Windows:
```
build.bat --cmake_generator "Visual Studio 17 2022" --config Release --build_wheel ^
          --parallel --nvcc_threads 1 --build_shared_lib ^
          --use_cuda --cuda_version "12.8" --cuda_home "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" ^
          --cudnn_home "D:\path\to\cudnn-installation-root" ^
          --use_vcpkg --use_binskim_compliant_compile_flags ^
          --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=native" ^
          --cmake_extra_defines "onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON"
```

## Minimum ONNX Runtime Version

The plugin is compiled against the ONNX Runtime headers in this repository, but it is designed to load into an **older** ONNX Runtime runtime as well. The minimum compatible version is declared in [`plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION`](../../plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION) (currently **1.24.4**) and is the single source of truth:

- It is baked into the plugin library at build time (`ORT_PLUGIN_EP_MIN_ORT_VERSION`) and enforced at load time.
- It is documented as the minimum core ONNX Runtime version in the `onnxruntime-ep-cuda12`/`onnxruntime-ep-cuda13` Python wheel and NuGet package READMEs. The plugin packages do **not** declare a hard dependency on the core `onnxruntime` package, so you install a compatible `onnxruntime` (version `>=<min>`) separately.

At load time, `CreateEpFactories()` negotiates the API version with the runtime: it reads the runtime's reported version, enforces the minimum, and requests the `OrtApi` that matches the **runtime** (not the build). If the runtime is older than the minimum, registration fails with a descriptive error instead of crashing. As a result, the same plugin binary works with any ONNX Runtime from the minimum version onward, including runtimes newer than the version it was built against (API versions are backward compatible because they are only appended to).

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

## Running Tests

The focused validation script for the CUDA Plugin EP is `onnxruntime/test/python/transformers/test_cuda_plugin_ep.py`.

### Test prerequisites

- Build ONNX Runtime with `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`.
- Install the built ONNX Runtime wheel.
- Install Python test dependencies. `test_cuda_plugin_ep.py` uses PyTorch for CPU-side reference computations, so CPU-only PyTorch is sufficient.

Example dependency install:

```bash
python -m pip install numpy onnx
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Point the test to the plugin library

The test helper tries to auto-detect the plugin library from the installed wheel or a local build tree. If you have multiple builds or want to be explicit, set `ORT_CUDA_PLUGIN_PATH` to the plugin library produced by your build.

Linux example:

```bash
export ORT_CUDA_PLUGIN_PATH=/path/to/build/Release/libonnxruntime_providers_cuda_plugin.so
```

Windows example:

```cmd
set ORT_CUDA_PLUGIN_PATH=E:\path\to\build\Release\Release\onnxruntime_providers_cuda_plugin.dll
```

### Run the test script

Run the script from a directory outside the repository checkout to avoid Python module shadowing.

```bash
cd onnxruntime/test/python/transformers
python test_cuda_plugin_ep.py
```

On Windows:

```cmd
cd /d onnxruntime\test\python\transformers
python test_cuda_plugin_ep.py
```

The script validates plugin registration, device enumeration, provider options, operator coverage, and that key nodes are actually assigned to `CudaPluginExecutionProvider`.

### Test against the minimum supported ORT version

The plugin must keep working on the oldest supported ONNX Runtime (see [Minimum ONNX Runtime Version](#minimum-onnx-runtime-version)), not just the version it was built against. To validate this locally, install the minimum base runtime and run the same test against the freshly built plugin library:

```bash
# 1. Build the plugin (see "Build Instructions"). This produces the plugin .so and installs the
#    matching, freshly built onnxruntime wheel.

# 2. Replace the base runtime with the minimum supported version.
pip install "onnxruntime==$(cat plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION)" --force-reinstall

# 3. Point the test at the freshly built plugin library and run it.
export ORT_CUDA_PLUGIN_PATH=/path/to/build/Release/libonnxruntime_providers_cuda_plugin.so
cd onnxruntime/test/python/transformers
python test_cuda_plugin_ep.py
```

This loads the plugin (compiled against the latest headers) into the minimum supported ONNX Runtime, exercising the API-version negotiation in `CreateEpFactories()`. If the plugin accidentally depends on an API newer than the declared minimum, the test fails here. The same check runs automatically in the `Test Linux CUDA Plugin EP` CI stage.


## Verification
You can generate a parity report comparing the kernels available in the plugin EP versus the statically linked CUDA EP.
```bash
# Check runtime registry parity:
python tools/ci_build/cuda_plugin_parity_report.py --runtime --plugin-ep-lib build/Linux/RelWithDebInfo/libonnxruntime_providers_cuda_plugin.so
```
