# Kernel contract

**Owns:** Tracing a kernel from host-visible inputs through program setup and
GPU-read tensor data. Arithmetic belongs in numerical-semantics;
registration rules belong in registration.
**Update when:** The host/device validation or program-setup contract changes.

## Trace each changed kernel

Before judging a changed `Program`, trace it end to end. Read the base version
of every function the diff touches, not only the hunk: cache-key and bounds
defects usually involve lines the diff did not change.

1. **Host side (`ComputeInternal`).** Which inputs are read on the host, what is
   validated, how the output shape is computed, which early exits exist, and how
   64-bit values are narrowed for uniforms.
2. **Program setup.** Each `AddInput` and `AddOutput` with its
   `ProgramTensorMetadataDependency` and component count, `SetDispatchGroupSize`,
   `SetWorkgroupSize`, `SetSubgroupSize`, `CacheHint`, `AddUniformVariables`,
   overridable constants, and `SetIndirectDispatchTensor`.
3. **Shader side.** `GenerateShaderCode`, or the `.wgsl.template` plus the
   parameters C++ passes to it. List every C++ value that changes emitted text;
   apply [cache identity](cache-identity.md).
4. **Registration.** Macro, domain, opset range, type constraints, host-input
   declarations, and table entry; apply [registration](registration.md).
5. **Tests.** Identify a test executing each shader variant on WebGPU;
   apply [validation](validation.md).

## Validate inputs and narrowing

- Validate attributes and host-visible inputs in `ComputeInternal` with
  `ORT_RETURN_IF` or `ORT_RETURN_IF_NOT`, including empty and rank-0 shapes.
  Every attribute or host input feeding sizes or indices needs validation of
  range, sign, rank, and divisibility, returned as a `Status` rather than thrown.
- Search changed host code for `static_cast<int32_t>`, `static_cast<uint32_t>`,
  `gsl::narrow_cast`, and C-style casts applied to shapes, dimensions, pads,
  axes, offsets, attributes, or sizes reaching uniforms. Each needs a range
  check or `onnxruntime::narrow`; ordinary casts truncate silently.
- Element and byte counts need overflow-safe arithmetic; use `SafeInt` for
  byte sizes. More than 2^32 elements cannot be described by a `u32` uniform:
  fail with a clear `Status`, or verify segmented access and representable
  per-segment indexing rather than narrowing the whole count.
- For indices, positions, sequence lengths, and offsets read from tensor data
  on the GPU, find the shader clamp or skip **after negative-index wrapping**.
  The host cannot validate these values without a readback. Compare with
  RotaryEmbedding's `position_ids` bounds validation. Keep decisions on-device;
  use [capture-safe interaction](runtime-and-portability.md#graph-capture).

## Regression anchors

These are historical examples; verify the affected code in the reviewed revision.

| Failure | Evidence | Check |
| --- | --- | --- |
| int64 attribute truncated to int32, then out-of-bounds read | [#28721](https://github.com/microsoft/onnxruntime/pull/28721), Pad | Range-check before narrowing |
| Missing host validation of degenerate inputs | [#29255](https://github.com/microsoft/onnxruntime/pull/29255), GridSample; [#29595](https://github.com/microsoft/onnxruntime/pull/29595), LSTM | Validate before program setup |
| Unbounded GPU-read indices | RotaryEmbedding `position_ids` validation | Clamp or skip in WGSL |
