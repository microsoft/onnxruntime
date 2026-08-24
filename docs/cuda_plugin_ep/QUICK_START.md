# CUDA Plugin EP Quick Start

## Build Instructions

To build ONNX Runtime with the CUDA Plugin Execution Provider, pass `--cmake_extra_defines "onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON"` with the build script.

If the flag is omitted, the default build uses the legacy source-built CUDA EP (`onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=OFF`).

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

### Building and testing without cuDNN at runtime

The CUDA Plugin EP build still requires cuDNN headers, but the plugin library must not have a hard runtime dependency on cuDNN. When cuDNN is not present, non-cuDNN kernels can still run. Kernels that still require cuDNN fail with `NOT_IMPLEMENTED` unless they have a native CUDA fallback.

For local Linux CUDA 13 validation, use the no-cuDNN helper script. It keeps `CUDNN_HOME` available for headers, excludes cuDNN directories from `PATH` and `LD_LIBRARY_PATH`, verifies the plugin has no direct cuDNN dependency, and runs plugin tests in no-cuDNN mode:

```bash
bash .env/cuda_130_plugin_no_cudnn.sh --build --test_plugin
```

The test mode sets `ORT_TEST_CUDA_PLUGIN_EP=1` and `ORT_TEST_CUDA_PLUGIN_NO_CUDNN=1`, which passes `enable_cudnn=0` to plugin sessions and skips plugin tests for operators that still require cuDNN, such as Conv, ConvTranspose, BatchNormalization, InstanceNormalization, LRN, Einsum, and cuDNN-backed pooling paths.

## Minimum ONNX Runtime Version

The plugin is compiled against the ONNX Runtime headers in this repository, but it is designed to load into an **older** ONNX Runtime runtime as well. The minimum compatible version is declared in [`plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION`](../../plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION) (currently **1.24.4**) and is the single source of truth:

- It is baked into the plugin library at build time (`ORT_PLUGIN_EP_MIN_ORT_VERSION`) and enforced at load time.
- It is documented as the minimum core ONNX Runtime version in the `onnxruntime-ep-cuda12`/`onnxruntime-ep-cuda13` Python wheel and NuGet package READMEs. The plugin packages do **not** declare a hard dependency on the core `onnxruntime` package, so you install a compatible `onnxruntime` (version `>=<min>`) separately.

At load time, `CreateEpFactories()` negotiates the API version with the runtime: it reads the runtime's reported version, enforces the minimum, and requests the `OrtApi` that matches the **runtime** (not the build). If the runtime is older than the minimum, registration fails with a descriptive error instead of crashing. As a result, the same plugin binary works with any ONNX Runtime from the minimum version onward, including runtimes newer than the version it was built against (API versions are backward compatible because they are only appended to).

## Running

When the plugin is built, it will produce `libonnxruntime_providers_cuda.so` on Linux, `onnxruntime_providers_cuda.dll` on Windows, or `libonnxruntime_providers_cuda.dylib` on macOS in the build output directory alongside the ONNX Runtime library.

The plugin EP is registered under the name **`CUDAExecutionProvider`** and intentionally uses the same native provider library filename as the legacy CUDA EP. The selected build mode determines what that filename contains:

- `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`: `onnxruntime_providers_cuda` is the CUDA plugin EP.
- `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=OFF`: `onnxruntime_providers_cuda` is the legacy source-built CUDA EP.

The plugin uses the EP Plugin API (`RegisterExecutionProviderLibrary` / `GetEpDevices` / `SessionOptionsAppendExecutionProvider_V2`). The bundled ONNX Runtime Python package auto-registers the plugin from its `onnxruntime/capi/` directory when the build info contains `cuda-plugin-ep=1`. Direct/native use, C++ use, and the standalone `onnxruntime-ep-cuda12` / `onnxruntime-ep-cuda13` plugin packages still register the plugin library explicitly before creating sessions that request `CUDAExecutionProvider`.

### C++ API

Use `Env::RegisterExecutionProviderLibrary` to load the plugin, `Env::GetEpDevices` to discover the CUDA devices it exposes, and `SessionOptions::AppendExecutionProvider_V2` to add the selected device to the session.

```cpp
#include "onnxruntime_cxx_api.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PluginTest");

// 1. Register the plugin library.
env.RegisterExecutionProviderLibrary("CUDAExecutionProvider",
                                     ORT_TSTR("libonnxruntime_providers_cuda.so"));

// 2. Enumerate available EP devices and pick the CUDA plugin device.
auto ep_devices = env.GetEpDevices();
std::vector<Ort::ConstEpDevice> plugin_devices;
for (const auto& dev : ep_devices) {
  if (std::string(dev.EpName()) == "CUDAExecutionProvider") {
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

Bundled CUDA plugin wheels auto-register the plugin at `import onnxruntime` time when `cuda-plugin-ep=1` appears in `onnxruntime.get_build_info()` and the provider library is present in `onnxruntime/capi/`. After import, `CUDAExecutionProvider` can be used like the legacy CUDA EP name.

If CUDA/cuDNN DLL discovery must be prepared first (commonly on Windows), call `onnxruntime.preload_dlls(...)`; it retries bundled plugin registration after loading DLLs and warns if registration still fails.

When using a standalone plugin package or a manually built plugin library outside the bundled wheel, use `onnxruntime.register_execution_provider_library` to load the plugin, `onnxruntime.get_ep_devices` to discover devices, and `SessionOptions.add_provider_for_devices` to add the selected device.

**Bundled wheel approach:**

```python
import onnxruntime as ort

sess = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("CUDAExecutionProvider", {"device_id": "0"}),
        "CPUExecutionProvider",
    ],
)
```

**Standalone or manually registered plugin approach (recommended when using EP devices):**

```python
import onnxruntime as ort

# 1. Register the plugin library.
ort.register_execution_provider_library(
    "CUDAExecutionProvider",
    "libonnxruntime_providers_cuda.so",
)

# 2. Enumerate devices and pick the CUDA plugin device.
devices = ort.get_ep_devices()
plugin_device = next(d for d in devices if d.ep_name == "CUDAExecutionProvider")

# 3. Create session with the plugin device.
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices([plugin_device], {})

sess = ort.InferenceSession("model.onnx", sess_options=sess_options)
```

**Provider-name approach with explicit registration:**

You can also pass `CUDAExecutionProvider` by name in the `providers` list after explicit registration:

```python
import onnxruntime as ort

ort.register_execution_provider_library(
    "CUDAExecutionProvider",
    "libonnxruntime_providers_cuda.so",
)

sess = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("CUDAExecutionProvider", {"device_id": "0"}),
        "CPUExecutionProvider",
    ],
)
```

**Python `OrtValue` host/device copies:**

`OrtValue.update_inplace()` and `OrtValue.numpy()` work with CUDA plugin tensors after the plugin has been registered. The Python binding cannot call CUDA runtime APIs directly; host/device copies must use the data-transfer implementation registered by the CUDA plugin library. If `OrtValue.update_inplace()` fails with a message about the CUDA provider interface or an unsupported GPU device, verify that the plugin library is registered before creating or updating CUDA `OrtValue` objects.

### External GPU Allocator Options

The CUDA plugin EP supports the same external GPU allocator provider options as the legacy CUDA EP: `gpu_external_alloc`, `gpu_external_free`, and `gpu_external_empty_cache` (also accepted with the canonical `ep.cuda.*` session config prefix). External allocator callbacks are session-scoped. A session that provides external allocator callbacks creates a per-session CUDA device allocator from that EP instance's options; a later session on the same GPU without those options continues to use the plugin factory's internal arena or CUDA mempool allocator.

`user_compute_stream` and an external allocator cannot be used together in the same session. If both are configured, session creation fails with `ORT_INVALID_ARGUMENT`.

## Running Tests

The focused validation script for the CUDA Plugin EP is `onnxruntime/test/python/transformers/test_cuda_plugin_ep.py`.

### Test prerequisites

- Build ONNX Runtime with CUDA plugin EP enabled by setting `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`.
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
export ORT_CUDA_PLUGIN_PATH=/path/to/build/Release/libonnxruntime_providers_cuda.so
```

Windows example:

```cmd
set ORT_CUDA_PLUGIN_PATH=E:\path\to\build\Release\Release\onnxruntime_providers_cuda.dll
```

### Run the test script

Run the script from a directory outside the repository checkout to avoid Python module shadowing.

```bash
cd onnxruntime/test/python/transformers
export ORT_TEST_CUDA_PLUGIN_EP=1
python test_cuda_plugin_ep.py
```

On Windows:

```cmd
cd /d onnxruntime\test\python\transformers
set ORT_TEST_CUDA_PLUGIN_EP=1
python test_cuda_plugin_ep.py
```

The script validates plugin registration, device enumeration, provider options, operator coverage, and that key nodes are actually assigned to `CUDAExecutionProvider`.

To run the same focused test against a plugin build without cuDNN in the runtime search path:

```bash
export ORT_TEST_CUDA_PLUGIN_NO_CUDNN=1
export ORT_TEST_CUDA_PLUGIN_EP=1
export ORT_CUDA_PLUGIN_PATH=/path/to/build/Release/libonnxruntime_providers_cuda.so
python test_cuda_plugin_ep.py
```

### Test against the minimum supported ORT version

The plugin must keep working on the oldest supported ONNX Runtime (see [Minimum ONNX Runtime Version](#minimum-onnx-runtime-version)), not just the version it was built against. To validate this locally, install the minimum base runtime and run the same test against the freshly built plugin library:

```bash
# 1. Build the plugin (see "Build Instructions"). This produces the plugin .so and installs the
#    matching, freshly built onnxruntime wheel.

# 2. Replace the base runtime with the minimum supported version.
pip install "onnxruntime==$(cat plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION)" --force-reinstall

# 3. Point the test at the freshly built plugin library and run it.
export ORT_CUDA_PLUGIN_PATH=/path/to/build/Release/libonnxruntime_providers_cuda.so
cd onnxruntime/test/python/transformers
python test_cuda_plugin_ep.py
```

This loads the plugin (compiled against the latest headers) into the minimum supported ONNX Runtime, exercising the API-version negotiation in `CreateEpFactories()`. If the plugin accidentally depends on an API newer than the declared minimum, the test fails here. The same check runs automatically in the `Test Linux CUDA Plugin EP` CI stage.


## Verification
You can generate a parity report comparing the kernels available in the plugin EP versus the statically linked CUDA EP.
```bash
# Check runtime registry parity:
python tools/ci_build/cuda_plugin_parity_report.py --runtime --plugin-ep-lib build/Linux/RelWithDebInfo/libonnxruntime_providers_cuda.so
```
