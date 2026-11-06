# Kernel authoring and registration

**Owns:** Native kernel integration, host validation, program setup, and advertised
domain/opset/type/layout support.
**Update when:** These contracts, registries, or kernel-document generation change.
Shader checks live in [shaders](shaders.md).

## Author a new native kernel

1. Establish schema/reference semantics: domain/opset, attributes/defaults,
   optional inputs, host placement, types/ranks/layouts, and output shape.
   Similar CPU/JSEP names do not prove equivalence. Leave unsupported cases
   on another provider rather than over-registering support.
2. Reuse matching native helpers. Standard/contrib sources belong under
   `onnxruntime/core/providers/webgpu` / `onnxruntime/contrib_ops/webgpu`.
   Shader kernels use `WebGpuKernel::ComputeInternal` and
   `Program`/`GenerateShaderCode`; keep invocation data local. Follow the
   [template workflow](../../../../onnxruntime/core/providers/webgpu/wgsl_templates/README.md),
   without editing generated headers. View/copy kernels may reuse context paths.
3. Trace the stages below. Allocate/run through the context; apply
   [capture/ownership](runtime.md), [shader](shaders.md), and
   [feature eligibility](devices-and-features.md) checks as applicable.
   Runtime values use inputs/uniforms; code specializations need cache identity.
   Retain fallbacks for optional fast paths.
4. Complete [registration](#registration) and separately gated
   capability/optimizer predicates. Check [source/build integration](build-and-dependencies.md)
   for built-in, shared/static plugin, and WASM targets as affected, including
   `plugin-ep-webgpu/paths.txt` and template discovery.
5. Add [validation](validation.md) for the advertised contract and specializations:
   malformed inputs, scalars/empty tensors, tails, broadcasts, dtype/opset
   boundaries, feature fallbacks, and same-context input/shape changes.
   Assess CI evidence of WebGPU execution and [performance](profiling-and-performance.md).
   Record target gaps; compilation or CPU fallback is insufficient.

## Trace each changed kernel

Read changed functions in full, including their base versions, then trace:

1. **Host:** host-read inputs, validation, output shape, early exits, 64-bit narrowing.
2. **Setup:** `AddInput`/`AddOutput`, `ProgramTensorMetadataDependency`, components,
   `SetDispatchGroupSize`, `SetWorkgroupSize`, `SetSubgroupSize`, `CacheHint`,
   `AddUniformVariables`, overrides, and `SetIndirectDispatchTensor`.
3. **Shader:** `GenerateShaderCode` or `.wgsl.template` and C++ parameters;
   map every text-changing value to [cache identity](shaders.md#cache-identity).
4. **Registration:** macro/table, domain/opsets/types, and host-input declarations.
5. **Tests:** WebGPU execution of each variant via [validation](validation.md).

## Validate inputs and narrowing

- Validate every size/index-producing attribute or host input for range, sign,
  rank, and divisibility, including empty/rank-0 shapes. Return `Status` through
  `ORT_RETURN_IF`/`ORT_RETURN_IF_NOT` in `ComputeInternal`; do not throw.
- Audit shape/dimension/pad/axis/offset/attribute/size casts reaching uniforms:
  `static_cast<int32_t>`, `static_cast<uint32_t>`, `gsl::narrow_cast`, and C-style
  casts need range checks or `onnxruntime::narrow`.
- Use overflow-safe element/byte arithmetic (`SafeInt` for bytes). Counts above
  2^32 cannot fit a `u32` uniform: return a clear `Status`, or prove segmented
  access and representable per-segment indices without narrowing the total.
- GPU-read indices, positions, sequence lengths, and offsets need shader
  clamps/skips **after negative-index wrapping**. Avoid host readback; use
  [capture-safe interaction](runtime.md#graph-capture).

## Registration

- Register standard kernels in `onnxruntime/core/providers/webgpu/webgpu_execution_provider.cc`,
  contrib kernels in `onnxruntime/contrib_ops/webgpu/webgpu_contrib_kernels.cc`.
- Keep opset ranges contiguous/non-overlapping: close the prior range at the
  new version minus one; update forward declarations and `BuildKernelCreateInfo`.
- Check corresponding `kMSInternalNHWCDomain` registrations for preferred NHWC.
- Match type constraints to WGSL. Its lack of int64 requires an explicit,
  tested range contract consistent with `GetOpTypeConstraints` and `enableInt64`.
- Declare **every host-read input** with `InputMemoryType(OrtMemTypeCPUInput, index)`
  in every affected registration/opset range.
- Regenerate `docs/OperatorKernels.md`; `windows_gpu_doc_gen.yml` validates
  WebGPU-enabled generation. `js/web/docs/webgpu-operators.md` describes
  deprecated JSEP and does not establish native support.
- Prove tests reach the advertised domain/opsets/types on WebGPU.
