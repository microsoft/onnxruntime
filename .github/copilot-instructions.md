# Copilot Instructions for ONNX Runtime

For detailed codebase conventions, architecture, and coding standards, see [AGENTS.md](../AGENTS.md).

## Code Review

### No C API Version Bump Needed for API Additions

`ORT_API_VERSION`, the `ort_api_1_to_N` function pointer table, and the version-boundary `static_assert` checks in
`onnxruntime/core/session/onnxruntime_c_api.cc` are updated only during release preparation — not each time a new API
is added. See [`docs/Versioning.md`](../docs/Versioning.md) for the full release versioning process.

During development, new API function pointers are appended to the **current** `ort_api_1_to_N` table. This is the
expected workflow and does **not** require a version bump, a new table, or new `static_assert` entries. Do not flag
PRs that append new function pointers to the current table as needing a version bump.
