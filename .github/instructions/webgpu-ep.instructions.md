---
description: "Route native WebGPU EP implementation and review to its scoped correctness, portability, registration, and validation checks."
applyTo: "onnxruntime/core/providers/webgpu/**,onnxruntime/contrib_ops/webgpu/**,onnxruntime/test/providers/webgpu/**,onnxruntime/test/webgpu/**,include/onnxruntime/core/providers/webgpu/**,onnxruntime/core/session/plugin_ep/ep_factory_webgpu.*,plugin-ep-webgpu/**,cmake/onnxruntime_providers_webgpu.cmake,cmake/patches/dawn/**,cgmanifests/webgpu/**,tools/python/wgsl_template/**"
---

# Native WebGPU Execution Provider

For implementation and review of changes affecting native WebGPU, load
[`code-review-webgpu`](../skills/code-review-webgpu/SKILL.md) and apply every matching
row in its reading table. That skill owns the WebGPU checks and their references;
keep this file as the path-based entry point. For reviews, also use
[code-review](../skills/code-review/SKILL.md).

Apply the skill's scope gate first. Do not use WebGPU skills for unrelated
changes or JSEP-only work; in mixed changes, scope checks to WebGPU effects.

A `Program` generates WGSL compiled and reused by program cache key. The same EP
serves native Dawn (D3D12, Vulkan, Metal), browser WebAssembly, and the WebGPU
plugin EP. Classify changes by affected behavior and callers, including shared
helpers outside these paths. JSEP is deprecated; follow its
[review policy](jsep-deprecated.instructions.md), including checking the native
counterpart of a JSEP kernel fix before deciding native applicability.
