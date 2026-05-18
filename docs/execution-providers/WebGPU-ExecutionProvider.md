---
title: WebGPU
description: Instructions to execute ONNX Runtime with the WebGPU execution provider
parent: Execution Providers
nav_order: 13
redirect_from: /docs/reference/execution-providers/WebGPU-ExecutionProvider
---

# WebGPU Execution Provider
{: .no_toc }

The WebGPU Execution Provider enables hardware-accelerated inference of ONNX models on a wide range of GPUs by targeting the [WebGPU](https://www.w3.org/TR/webgpu/) API. On native platforms it uses [Dawn](https://dawn.googlesource.com/dawn) (Google's WebGPU implementation), which in turn dispatches to D3D12 on Windows, Vulkan on Linux/Android, and Metal on macOS/iOS. The same execution provider also powers the `webgpu` backend of [ONNX Runtime Web](../get-started/with-javascript/web.md) inside the browser.

Compared to other GPU execution providers, WebGPU EP aims to be cross-vendor and cross-platform: a single build can run on NVIDIA, AMD, Intel, Apple, and Qualcomm GPUs through the OS's native graphics API, without requiring a vendor-specific SDK on the end user's machine.

{: .note }
> If you are using ONNX Runtime Web (JavaScript/TypeScript in the browser), see the dedicated tutorial [Using the WebGPU Execution Provider](../tutorials/web/ep-webgpu.md) for browser-specific topics such as `onnxruntime-web/webgpu` imports, `Tensor.fromGpuBuffer`, and `env.webgpu` flags. The rest of this page focuses on the native (C/C++/Python/C#) WebGPU EP.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Install

Two flavors of native WebGPU EP are available:

- **Plugin EP packages (recommended).** Distributed separately and registered at runtime alongside any compatible core ONNX Runtime install.
  - Python: `pip install onnxruntime onnxruntime-ep-webgpu`
  - .NET: add a reference to `Microsoft.ML.OnnxRuntime.EP.WebGpu` together with your existing `Microsoft.ML.OnnxRuntime` package.
- **Static-library Python wheel.** A single wheel with the WebGPU EP built into ONNX Runtime: `pip install onnxruntime-webgpu` (built from source with `--use_webgpu` / `--wheel_name_suffix=webgpu`). Use this when you need a single artifact and only require basic EP registration.

For ONNX Runtime Web (browser), no separate install is required beyond the `onnxruntime-web` package — see the [JavaScript quickstart](../get-started/with-javascript/web.md).

## Requirements

The WebGPU EP requires a GPU and driver that support one of Dawn's native backends:

| Platform | Backend used by Dawn |
|----------|----------------------|
| Windows  | Direct3D 12 (default), Vulkan (opt-in via `dawnBackendType`) |
| Linux    | Vulkan |
| Android  | Vulkan |
| macOS    | Metal |
| iOS      | Metal |

Most discrete and integrated GPUs released in the last several years meet these requirements. On Windows the GPU/driver must support D3D12 (the same baseline as the DirectML EP). In the browser, the underlying browser must implement WebGPU; see the [WebGPU support matrix](../get-started/with-javascript/web.md#supported-versions) for ORT Web specifics and [webgpu.io/status](https://webgpu.io/status/) for browser availability.

## Build from source

Use `tools/ci_build/build.py` with the `--use_webgpu` flag. See the [BUILD page](../build/eps.md) for the general build workflow.

```bash
# Minimal build with WebGPU EP (static-library link of Dawn, the default)
python tools/ci_build/build.py --build_dir build/RelWithDebInfo --config RelWithDebInfo \
    --build_shared_lib --parallel --use_webgpu
```

### Build flags

| Flag | Description |
|------|-------------|
| `--use_webgpu [static_lib\|shared_lib]` | Enable the WebGPU EP. `static_lib` (the default if no value is given) builds the WebGPU EP into the main `onnxruntime` library. `shared_lib` builds the WebGPU EP as a separate [plugin EP library](./plugin-ep-libraries/index.md) (`onnxruntime_USE_EP_API_ADAPTERS=ON`) and is what produces the `onnxruntime_providers_webgpu.{dll,so,dylib}` shipped in the `onnxruntime-ep-webgpu` / `Microsoft.ML.OnnxRuntime.EP.WebGpu` packages. `shared_lib` is not supported for WebAssembly builds. Sets the CMake flag `onnxruntime_USE_WEBGPU=ON`. Dawn linkage is controlled separately (see `--use_external_dawn`). |
| `--use_external_dawn` | Link against an externally-provided Dawn instead of building Dawn from source. Requires `--use_webgpu`. Sets `onnxruntime_USE_EXTERNAL_DAWN=ON`. |
| `--wgsl_template {static,dynamic}` | Select the WGSL shader-template generator. `static` (default) bakes shader sources in at build time; `dynamic` generates them at runtime. |
| `--enable_pix_capture` | Windows only. Build with PIX GPU debugger support. Requires `--use_webgpu`. |
| `--use_jsep` | Enables the JavaScript Execution Provider used by ORT Web's pre-existing WebGPU/WebNN bridge. This is a different EP from `--use_webgpu`; see [Notes](#notes) below. |

{: .note }
> Building from a fully isolated network is not currently supported for native WebGPU because Dawn must be fetched during the build. See [Dependencies](../build/dependencies.md) for context.

## Usage

There are two ways to add the WebGPU EP to a session:

1. **Plugin EP packages (recommended).** The plugin EP shared library is registered at runtime via `register_execution_provider_library` / `RegisterExecutionProviderLibrary`. The `onnxruntime-ep-webgpu` (Python) and `Microsoft.ML.OnnxRuntime.EP.WebGpu` (.NET) packages bundle the shared library and provide helpers that return its path and the EP name to use.
2. **Static-library build (short-name API).** When ONNX Runtime is built with `--use_webgpu` (`static_lib`) the WebGPU EP is built into the main `onnxruntime` library and can be added through the generic `AppendExecutionProvider("WebGPU", { ... })` call. Provider options are passed as a string→string map (see [Configuration options](#configuration-options) for the full list of keys).

In both cases the underlying EP and option keys are the same. The plugin EP path is the recommended approach going forward; the short-name API is most useful for C/C++/C# applications that build ONNX Runtime themselves.

### Plugin EP (recommended)

#### Python

```python
import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu_ep

# Register the plugin EP library with ONNX Runtime
ort.register_execution_provider_library("webgpu", webgpu_ep.get_library_path())

# Discover WebGPU devices
webgpu_devices = [d for d in ort.get_ep_devices() if d.ep_name == webgpu_ep.get_ep_name()]

# Create a session using the WebGPU EP
sess_options = ort.SessionOptions()
sess_options.add_provider_for_devices(webgpu_devices, {
    "preferredLayout": "NHWC",
    "enableGraphCapture": "1",
})
session = ort.InferenceSession("model.onnx", sess_options=sess_options)
```

#### C# / .NET

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.EP.WebGpu;

var env = OrtEnv.Instance();
env.RegisterExecutionProviderLibrary("webgpu_ep", WebGpuEp.GetLibraryPath());

OrtEpDevice? webGpuDevice = null;
foreach (var d in env.GetEpDevices())
{
    if (d.EpName == WebGpuEp.GetEpName())
    {
        webGpuDevice = d;
        break;
    }
}

using var sessionOptions = new SessionOptions();
sessionOptions.AppendExecutionProvider(env, new[] { webGpuDevice },
    new Dictionary<string, string>
    {
        ["preferredLayout"] = "NHWC",
        ["enableGraphCapture"] = "1",
    });

using var session = new InferenceSession("model.onnx", sessionOptions);
```

#### C++

The C++ pattern is the generic plugin EP idiom — the host application is responsible for locating the `onnxruntime_providers_webgpu.{dll,so,dylib}` shared library (from the NuGet package's runtime files, a manual build, etc.):

```cpp
#include "onnxruntime_cxx_api.h"

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "webgpu_sample");
env.RegisterExecutionProviderLibrary("webgpu_ep",
    ORT_TSTR("onnxruntime_providers_webgpu.dll"));

std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();
std::array<Ort::ConstEpDevice, 1> selected{nullptr};
for (auto d : ep_devices) {
    if (std::strcmp(d.GetName(), "WebGpuExecutionProvider") == 0) {
        selected[0] = d;
        break;
    }
}

Ort::KeyValuePairs ep_options;
ep_options.Add("preferredLayout",    "NHWC");
ep_options.Add("enableGraphCapture", "1");

Ort::SessionOptions session_options;
session_options.AppendExecutionProvider_V2(env, selected, ep_options);

Ort::Session session(env, ORT_TSTR("model.onnx"), session_options);
```

See [Using a Plugin Execution Provider Library](./plugin-ep-libraries/usage.md) for the full cross-EP idiom, including library lifetime and per-language API references.

### Static-library build (short-name API)

For applications using a `--use_webgpu` (static-lib) build of ONNX Runtime, the WebGPU EP can also be registered through the generic `AppendExecutionProvider` API. There is no dedicated `OrtSessionOptionsAppendExecutionProvider_WebGPU` function.

#### C++

```cpp
#include "onnxruntime_cxx_api.h"

Ort::SessionOptions session_options;

std::unordered_map<std::string, std::string> provider_options;
provider_options["preferredLayout"] = "NHWC";          // optional
provider_options["enableGraphCapture"] = "1";          // optional

session_options.AppendExecutionProvider("WebGPU", provider_options);

Ort::Session session(env, model_path, session_options);
```

#### C

```c
const char* keys[]   = {"preferredLayout", "enableGraphCapture"};
const char* values[] = {"NHWC", "1"};
OrtStatus* status = g_ort->SessionOptionsAppendExecutionProvider(
    session_options, "WebGPU", keys, values, 2);
```

#### C#

```csharp
using var sessionOptions = new SessionOptions();
var providerOptions = new Dictionary<string, string>
{
    ["preferredLayout"] = "NHWC",
    ["enableGraphCapture"] = "1",
};
sessionOptions.AppendExecutionProvider("WebGPU", providerOptions);

using var session = new InferenceSession("model.onnx", sessionOptions);
```

For Python, use the plugin EP path shown above.

### JavaScript / TypeScript (ONNX Runtime Web)

For browser usage, import `onnxruntime-web/webgpu` and list `'webgpu'` in `executionProviders`. See [Using the WebGPU Execution Provider](../tutorials/web/ep-webgpu.md) for the complete guide, including IO binding with GPU buffers.

## Configuration options

Provider options are read from the session's config entries with the key prefix `ep.webgpuexecutionprovider.*`. When set via `AppendExecutionProvider("WebGPU", { ... })`, supply the short names below (the `ep.webgpuexecutionprovider.` prefix is added internally).

### Execution-provider options

| Option | Allowed values | Description |
|--------|----------------|-------------|
| `preferredLayout` | `NCHW`, `NHWC` | Preferred data layout for layout-sensitive kernels. |
| `enableGraphCapture` | `0`, `1` | Enable [graph capture](#graph-capture) for models with static shapes that run entirely on WebGPU. |
| `enableInt64` | `0`, `1` | Enable native `int64` support in WGSL kernels (requires a device that exposes the matching feature). Forced on when `enableGraphCapture` is `1`, regardless of this setting. |
| `multiRotaryCacheConcatOffset` | non-negative integer | Offset used by multi-rotary cache concatenation kernels (advanced tuning option). |
| `forceCpuNodeNames` | newline-separated list | Force the listed node names to run on the CPU EP fallback instead of WebGPU. Each line is one node name; empty lines are ignored. |
| `enablePIXCapture` | `0`, `1` | Enable per-run PIX captures. Only meaningful in a Windows build configured with `--enable_pix_capture`. |

### WebGPU context options

These options configure the underlying Dawn/WebGPU instance, adapter, and device shared across sessions with the same `deviceId`.

| Option | Allowed values | Description |
|--------|----------------|-------------|
| `deviceId` | integer | Selects a WebGPU context to use. Sessions sharing the same `deviceId` share an underlying device and buffer cache. Defaults to `0`. |
| `dawnBackendType` | `D3D12`, `Vulkan` | Override the native backend Dawn requests. Defaults to the OS's preferred backend. |
| `powerPreference` | `high-performance`, `low-power` | Hint passed to `requestAdapter`. Defaults to the implementation default. |
| `webgpuInstance` | pointer (as integer) | Bring-your-own `WGPUInstance`. Encoded as the decimal representation of the pointer value. Use when sharing a Dawn instance with the host application. |
| `webgpuDevice` | pointer (as integer) | Bring-your-own `WGPUDevice`. Encoded as the decimal representation of the pointer value. Use together with `webgpuInstance` to share an existing device. |
| `dawnProcTable` | pointer (as integer) | Address of an external `DawnProcTable` to use when the host application loads Dawn itself (typical with `--use_external_dawn`). |
| `preserveDevice` | `0`, `1` | When `1`, keep the underlying WebGPU device alive after the last session that owns the context is released. Useful when sessions are created and destroyed repeatedly. |
| `validationMode` | `disabled`, `wgpuOnly`, `basic`, `full` | Controls WGSL/runtime validation. `disabled` skips ONNX-side validation, `wgpuOnly` relies on Dawn's validation, `basic` enables lightweight ONNX checks, `full` enables all available checks. Defaults to a build-dependent value. |
| `maxStorageBufferBindingSize` | integer (bytes) | Override the requested `maxStorageBufferBindingSize` device limit. Cannot exceed the adapter's reported limit. |

{: .note }
> Pointer-valued options (`webgpuInstance`, `webgpuDevice`, `dawnProcTable`) are parsed as base-10 integers — pass the decimal representation of the pointer value. Hex literals (e.g. `0x...`) are not accepted.

### Buffer cache modes

The EP maintains four pooled-buffer caches. Each accepts the same set of modes:

| Option | Allowed values |
|--------|----------------|
| `storageBufferCacheMode` | `disabled`, `lazyRelease`, `simple`, `bucket` |
| `uniformBufferCacheMode` | `disabled`, `lazyRelease`, `simple`, `bucket` |
| `queryResolveBufferCacheMode` | `disabled`, `lazyRelease`, `simple`, `bucket` |
| `defaultBufferCacheMode` | `disabled`, `lazyRelease`, `simple`, `bucket` |

Modes, in increasing order of caching aggressiveness:

- `disabled` — buffers are destroyed immediately on release; lowest memory footprint, highest allocation overhead.
- `lazyRelease` — buffers are released at the end of the next run.
- `simple` — single free-list keyed on exact size.
- `bucket` — buffers are bucketed by power-of-two size for fast reuse with bounded waste. Recommended for inference workloads with stable shapes.

## Graph capture

When a model has fully static shapes and all kernels run on WebGPU, setting `enableGraphCapture` to `1` records the WebGPU command sequence during an early run (after a configurable warm-up period) and replays the recorded commands on subsequent runs, significantly reducing per-run CPU overhead.

If any kernel falls back to CPU, or any input shape changes between runs, graph capture may fail or fall back to regular execution. In that case, leave `enableGraphCapture` unset or set it to `0`.

The same feature is exposed in ORT Web via the `enableGraphCapture` session option — see [`enableGraphCapture`](../tutorials/web/env-flags-and-session-options.md#enablegraphcapture).

## Profiling and debugging

- **Generic ORT profiling.** ORT's built-in profiler (`SessionOptions::EnableProfiling`) works with the WebGPU EP and produces per-op timing.
- **Native PIX capture (Windows).** Build with `--use_webgpu --enable_pix_capture`, attach PIX to the host process, and set `enablePIXCapture` to `1` on the session to capture WebGPU/D3D12 work for individual runs. PIX only attaches to D3D12, so this requires `dawnBackendType` to be `D3D12` (the default on Windows); it is a no-op when `dawnBackendType` is `Vulkan`.
- **WebGPU validation.** Tune `validationMode` to surface device-side issues during development; turn it down to `disabled` for production benchmarking to remove validation overhead.
- **Browser profiling.** For ORT Web, see [WebGPU profiling](../tutorials/web/performance-diagnosis.md#webgpu-profiling).

## Notes

### WebGPU EP vs. JSEP

ONNX Runtime Web historically powered its `webgpu` backend through the **JavaScript Execution Provider (JSEP)**, which is enabled at build time with `--use_jsep`. The native WebGPU EP (`--use_webgpu`) is a separate, C++-side implementation built on Dawn. Today both flags can be enabled in the same build (a transitional state); a future change is expected to make them mutually exclusive. Use `--use_webgpu` for native and WebAssembly+JSEP builds for the browser path that ORT Web currently ships.

### Relationship to DirectML

On Windows the WebGPU EP dispatches to D3D12 through Dawn, while the [DirectML EP](./DirectML-ExecutionProvider.md) dispatches to D3D12 through DirectML (DML). The two providers have different op coverage, fusion strategies, and packaging stories; neither is a drop-in replacement for the other.

## Additional resources

- [WebGPU specification (W3C)](https://www.w3.org/TR/webgpu/)
- [WGSL — WebGPU Shading Language](https://www.w3.org/TR/WGSL/)
- [Dawn — Google's WebGPU implementation](https://dawn.googlesource.com/dawn)
- [Using the WebGPU Execution Provider (ORT Web tutorial)](../tutorials/web/ep-webgpu.md)
- [Build ONNX Runtime Web](../build/web.md)
- [WebGPU browser support status](https://webgpu.io/status/)
