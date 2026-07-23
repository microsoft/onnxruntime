# C++ model-package sample

`model_package_sample.cpp` loads a model package, lets ONNX Runtime **select the best
variant** for the registered execution providers, creates a session, and runs one
inference — for every component in the package.

It uses the ORT **C++ API** (`onnxruntime_cxx_api.h`) together with a small set of
model-package wrappers in [`model_package_cxx.h`](model_package_cxx.h), so consuming a
package reads like ordinary C++ ORT code:

```cpp
Ort::ModelPackage pkg{package_root};
for (const std::string& name : pkg.ComponentNames()) {
  Ort::SessionOptions so;                       // leave empty → CPU variant
  so.AppendExecutionProvider_V2(env, {device}, {});  // or an EP device → its variant
  Ort::ModelPackageComponent comp = pkg.SelectComponent(env, name, so);
  Ort::Session session = comp.CreateSession(env);
  // ... run inference ...
}
```

Under the hood the model-package functions are the experimental `OrtModelPackageApi`,
looked up **by name** through `OrtApi::GetExperimentalFunction` (all marked `_SinceV28`).
`model_package_cxx.h` hides that lookup behind RAII classes; when the API is promoted to
the stable `OrtApi`, only that header changes and this sample stays the same.

The ORT DLL is loaded with `LoadLibrary` + `OrtGetApiBase` and the C++ API is pointed at
it with `Ort::InitApi(api)` (the file defines `ORT_API_MANUAL_INIT` before including the
C++ header), so **no import library (`onnxruntime.lib`) is required** — only the public
headers.

## Build

You need the ONNX Runtime C/C++ API headers (from this repo's
`include/onnxruntime/core/session/`) and any C++17 compiler. `model_package_cxx.h` lives
next to the sample, so no extra include path is needed for it.

### MSVC

```bat
cl /std:c++17 /EHsc ^
   /I "..\..\..\include\onnxruntime\core\session" ^
   model_package_sample.cpp /Fe:model_package_sample.exe
```

### clang / zig

```powershell
zig c++ -std=c++17 -I "..\..\..\include\onnxruntime\core\session" `
    model_package_sample.cpp -o model_package_sample.exe
```

(`clang++ -std=c++17 -I ... model_package_sample.cpp -o model_package_sample.exe`
works the same way.)

## Run

```
model_package_sample.exe <onnxruntime.dll> <package_root> [openvino_plugin.dll]
```

- `<onnxruntime.dll>` — path to a build/nightly onnxruntime.dll (**≥ 1.28**, the version
  that added the `_SinceV28` model-package experimental API). Its folder is added to the
  DLL search path so `onnxruntime_providers_shared.dll` next to it is found.
- `<package_root>` — a package directory (e.g. `..\portable_package`). For the
  non-portable package, run its `resolve.py` first.
- `[openvino_plugin.dll]` — optional. When supplied, the OpenVINO EP is registered and
  the OpenVINO-NPU variant is exercised in addition to CPU. Omit it to run **CPU
  variants only** (works on any machine). Use **OpenVINO EP ≥ 1.8.82.0 (OpenVINO SDK
  2026.2)** — the first build that emits/validates the EPContext `compatibility_string`.

### Example

```powershell
$ort = (python -c "import onnxruntime,os;print(os.path.join(os.path.dirname(onnxruntime.__file__),'capi','onnxruntime.dll'))")
model_package_sample.exe $ort ..\portable_package C:\path\to\onnxruntime_providers_openvino_plugin.dll
```

Expected (abridged):

```
ORT version: 1.28.0
OpenVINO EP registered; NPU device found
[component=prefill target=cpu]
  selected variant: cpu
  session created
      input dims=[1,4,64] output elems=256 all_zero=false absmean=0.0634591
[component=prefill target=ov]
  selected variant: ov
  session created
      input dims=[1,4,64] output elems=256 all_zero=false absmean=0.0634599
...
DONE
```

## What the code shows

The wrapper classes in `model_package_cxx.h` map 1:1 onto the experimental C API:

1. `Ort::ModelPackage pkg{root}` — `CreateModelPackageContext`; `pkg.ComponentNames()`
   — `GetComponentNames`. Open the package and enumerate components.
2. `pkg.SelectComponent(env, name, session_options)` — captures EP intent from the
   `OrtSessionOptions` (which EP device was appended, or none → CPU default) via
   `CreateModelPackageOptionsFromSessionOptions`, then `SelectComponent` picks the best
   variant (matches EP name, device, and the OpenVINO `compatibility_string`).
3. `comp.SelectedVariantName()` — `ModelPackageComponent_GetSelectedVariantName`.
4. `comp.CreateSession(env)` — `CreateSession` with `session_options=nullptr`; ORT merges
   the selected variant's `session_options`/`provider_options` from the package
   (e.g. `ep.share_ep_contexts`, the external-initializers folder, `ep.context_file_path`).
5. `session.Run(...)` — one inference on an all-`0.5` input, via the standard C++ API.
