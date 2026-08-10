---
description: "C API implementation and review guidance for API additions, ORT_API_VERSION, version tables, and release-boundary checks."
applyTo: "include/onnxruntime/core/session/onnxruntime_c_api.h,onnxruntime/core/session/onnxruntime_c_api.cc"
---

# C API Version-Table Changes

For ordinary C API additions, append new function pointers to the current `ort_api_1_to_N` table.

Do not bump `ORT_API_VERSION`, create a new version table, or add release-boundary `static_assert` entries for an
ordinary API addition. Those changes are made during release preparation. See [Versioning](../../docs/Versioning.md)
for the release versioning process.
