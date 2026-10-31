---
description: "C API implementation and review guidance for public C API updates."
applyTo: "include/onnxruntime/core/session/onnxruntime_c_api.h,include/onnxruntime/core/session/onnxruntime_ep_c_api.h"
---

# C API Updates

## API Structs and Initializers

Preserve ABI compatibility for all shipped public C API members: do not remove, reorder, or change existing
function-pointer signatures within a shipped C API struct.

For a new public C API function, append its function pointer to the end of the applicable API struct and corresponding
initializer. The append-only API structs are `OrtApi`, `OrtModelEditorApi`, `OrtCompileApi`, and `OrtInteropApi` in
[`onnxruntime_c_api.h`](../../include/onnxruntime/core/session/onnxruntime_c_api.h), and `OrtEpApi` in
[`onnxruntime_ep_c_api.h`](../../include/onnxruntime/core/session/onnxruntime_ep_c_api.h).

For an `OrtApi` addition, append the implementation pointer to the current `ort_api_1_to_N` table. For the companion
API structs, append it to the corresponding initializer in `model_editor_c_api.cc`, `compile_api.cc`, `interop_api.cc`,
or `plugin_ep/ep_api.cc`.

## API Documentation

Fully document every new public C API member with Doxygen comments that explain its behavior, parameters, return value,
and any ownership or lifetime requirements.

Include a `\since Version X.Y.` tag identifying the first ONNX Runtime release that provides the API. `X.Y` should be
`1.<current value of ORT_API_VERSION>`.

## Release Versioning

Do not bump `ORT_API_VERSION`, rename the current `OrtApi` version table, or add release-boundary markers or
`static_assert` entries when adding a public C API function. Those changes are made during release preparation. See
[Versioning](../../docs/Versioning.md) for the release versioning process.

## C++ Wrappers

Add a C++ API wrapper consistent with existing C++ API conventions for each new public C API, as appropriate. For
example, a C++ wrapper is appropriate if it removes the need for manual resource management, or allows use of
conventional C++ types.

Add C++ API declarations to
[`onnxruntime_cxx_api.h`](../../include/onnxruntime/core/session/onnxruntime_cxx_api.h) and implementation to
[`onnxruntime_cxx_inline.h`](../../include/onnxruntime/core/session/onnxruntime_cxx_inline.h).

If a C++ wrapper is not appropriate, document the reason in the pull request.
