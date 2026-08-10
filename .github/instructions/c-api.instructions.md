---
description: "C API implementation and review guidance for public C API updates."
applyTo: "include/onnxruntime/core/session/onnxruntime_c_api.h,include/onnxruntime/core/session/onnxruntime_ep_c_api.h"
---

# C API Additions

## API Structs and Initializers

For ordinary public C API additions, append the new function pointer to the end of the applicable API struct and its
corresponding initializer. Never insert or reorder existing members. The append-only API structs are `OrtApi`,
`OrtModelEditorApi`, `OrtCompileApi`, and `OrtInteropApi` in
[`onnxruntime_c_api.h`](../../include/onnxruntime/core/session/onnxruntime_c_api.h), and `OrtEpApi` in
[`onnxruntime_ep_c_api.h`](../../include/onnxruntime/core/session/onnxruntime_ep_c_api.h).

For an `OrtApi` addition, append the implementation pointer to the current `ort_api_1_to_N` table. For the companion
API structs, append it to the corresponding initializer in `model_editor_c_api.cc`, `compile_api.cc`, `interop_api.cc`,
or `plugin_ep/ep_api.cc`.

## Release Versioning

Do not bump `ORT_API_VERSION`, rename the current `OrtApi` version table, or add release-boundary markers or
`static_assert` entries for an ordinary API addition. Those changes are made during release preparation. See
[Versioning](../../docs/Versioning.md) for the release versioning process.

## C++ Wrappers

Add an idiomatic C++ API wrapper for each new public C API, as appropriate. Add the declaration to
[`onnxruntime_cxx_api.h`](../../include/onnxruntime/core/session/onnxruntime_cxx_api.h) and the implementation to
[`onnxruntime_cxx_inline.h`](../../include/onnxruntime/core/session/onnxruntime_cxx_inline.h). If a C++ wrapper is not
appropriate, document the reason in the pull request.
