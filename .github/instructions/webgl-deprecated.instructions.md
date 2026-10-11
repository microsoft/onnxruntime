---
description: "Review guidance for maintenance and removal of the deprecated onnxruntime-web WebGL backend."
applyTo: "js/web/lib/onnxjs/backends/webgl/**,js/web/lib/onnxjs/backends/backend-webgl.ts,js/web/lib/backend-onnxjs.ts,js/web/test/unittests/backends/webgl/**,js/web/test/e2e/browser-test-webgl.js,js/web/script/generate-webgl-operator-md.ts,js/web/docs/webgl-operators.md"
---

# Deprecated WebGL Backend

The onnxruntime-web WebGL backend is deprecated and scheduled for removal.

## Review Policy

Report changes that expand the WebGL backend with new operators, features, or performance work. Direct new GPU work
to WebGPU and users who cannot use WebGPU to the WASM/CPU backend.

Accept correctness and security fixes to existing behavior. Require focused regression coverage for an accepted WebGL
backend fix.

Accept deprecation and removal work. Read and follow
[`docs/design/onnxruntime_web_remove_webgl_backend.md`](../../docs/design/onnxruntime_web_remove_webgl_backend.md).
