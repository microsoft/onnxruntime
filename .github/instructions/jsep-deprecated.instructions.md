---
description: "Review guidance for changes to the deprecated JSEP implementation and its removal or migration."
applyTo: "onnxruntime/core/providers/js/**,onnxruntime/contrib_ops/js/**,js/web/lib/wasm/jsep/**,onnxruntime/wasm/pre-jsep.js,cmake/onnxruntime_providers_js.cmake,js/build_jsep.bat"
---

# Deprecated JSEP

JSEP is deprecated and is being replaced by the native WebGPU EP. Follow the contribution policy in
[`docs/JSEP_Deprecation.md`](../../docs/JSEP_Deprecation.md), which is the authoritative source for its status and
accepted changes.

## Review Policy

Report new operators, features, or performance-only work as actionable findings and direct that work to the native
WebGPU EP in `onnxruntime/core/providers/webgpu/` or `onnxruntime/contrib_ops/webgpu/`.

Accept correctness and security fixes to existing behavior. Require focused regression coverage for an accepted JSEP
fix.

For a JSEP WebGPU kernel fix, inspect the corresponding native WebGPU EP implementation. Report an actionable finding
if the same defect applies there but the change does not include the native fix and focused regression coverage.

Accept deprecation, migration, or removal work. Read and follow
[`docs/design/onnxruntime_web_jsep_to_webgpu_ep_migration.md`](../../docs/design/onnxruntime_web_jsep_to_webgpu_ep_migration.md).
