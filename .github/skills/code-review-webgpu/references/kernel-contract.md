# Kernel contract

**Owns:** New kernel authoring and tracing a kernel from host-visible inputs
through program setup and GPU-read tensor data. Arithmetic belongs in numerical-semantics;
registration rules belong in registration.
**Update when:** Kernel integration, host/device validation, or program setup changes.

## Author a new native kernel

1. Define the ONNX/contrib domain, opset semantics, attributes/defaults, input
   placement, supported types/ranks/layouts, output shape, and optional inputs.
   Compare the schema and reference implementation; a similarly named CPU or
   deprecated JSEP kernel is not proof of the native contract. Decide which
   unsupported configurations should remain on another provider rather than
   registering a kernel that cannot execute them.
2. Reuse an existing native kernel/helper when its semantics match. Place new
   standard kernels under `core/providers/webgpu` and contrib kernels under
   `contrib_ops/webgpu` (relative to `onnxruntime/`). Shader-backed kernels use
   `WebGpuKernel::ComputeInternal` and `Program`/`GenerateShaderCode`; keep
   invocation data local. Follow the
   [template build workflow](../../../../onnxruntime/core/providers/webgpu/wgsl_templates/README.md)
   rather than editing generated build headers. A view/copy-only kernel may
   reuse the existing context path without introducing a shader.
3. Trace the five stages below. Allocate/run through the compute context and
   apply [capture and ownership checks](runtime-and-portability.md), keeping
   per-run state local under supported concurrency. For shaders, use
   [buffer-access, dispatch, and synchronization checks](dispatch-and-wgsl.md)
   and [cache identity](cache-identity.md). Tensor values that merely vary at
   runtime belong in inputs/uniforms; values specializing generated code need
   identity coverage. Apply [numerical semantics](numerical-semantics.md),
   [layout/prepacking](tensor-layout-and-prepacking.md), and
   [device/feature eligibility](devices-and-features.md) as applicable, retaining
   a supported fallback for optional fast paths.
4. Complete [registration](registration.md): definition plus registry entry,
   opset/type/host-input constraints, required layout variants, and generated
   operator documentation. Recheck capability/partitioning or optimizer
   predicates if they separately gate the operator. Follow
   [build integration](build-and-dependencies.md) for new sources/dependencies:
   verify built-in, plugin/static-plugin, and WASM source/template discovery
   as affected, including `plugin-ep-webgpu/paths.txt` for extracted sources.
5. Add [validation](validation.md) for the advertised contract and each new
   specialization, including malformed inputs, scalars/empty tensors, tails,
   broadcasts, dtype/opset boundaries, and optional-feature fallback as
   applicable. Prove WebGPU execution and test cache reuse across changing
   inputs/shapes in one context. Measure any claimed optimization using
   [profiling and performance](profiling-and-performance.md). Record target
   gaps; compilation or CPU fallback does not establish a working new kernel.

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
  The host cannot validate these values without a readback. Keep decisions on-device;
  use [capture-safe interaction](runtime-and-portability.md#graph-capture).
