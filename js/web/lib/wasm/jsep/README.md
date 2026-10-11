# JSEP — deprecated

This directory holds the TypeScript half of **JSEP**, the JavaScript WebGPU compute path. It is **deprecated** and
will be removed. The replacement is the native WebGPU execution provider, already shipping as
`onnxruntime-web/webgpu` and `onnxruntime-web/jspi`.

**Bug fixes and security fixes only.** New operators, new features and performance work belong in the native
WebGPU EP (`onnxruntime/core/providers/webgpu/`).

**WebNN is not deprecated.** `backend-webnn.ts` and `webnn/` live here only because they share JSEP's
initialization glue. WebNN is already the native C++ WebNN EP in every build, and this code will be relocated to a
neutral path rather than removed.

See [docs/JSEP_Deprecation.md](../../../../../docs/JSEP_Deprecation.md) for more details.
