# Operators and computation

**Owns:** What models/operators WebGPU accepts and what values it computes:
host validation, registration, WGSL, cache keys, numerics, layouts, and graph changes.
**Update when:** Operator support, algorithms, shader generation, or graph rewrites change.
Scheduling/resource lifetime belongs to [execution](execution.md); device negotiation,
ABI and artifacts belong to [platform support and release](platform.md).

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
   [capture/ownership](execution.md), [shader](computation.md), and
   [feature eligibility](platform.md) checks as applicable.
   Runtime values use inputs/uniforms; code specializations need cache identity.
   Retain fallbacks for optional fast paths.
4. Complete [registration](#registration) and separately gated
   capability/optimizer predicates. Check [source/build integration](platform.md)
   for built-in, shared/static plugin, and WASM targets as affected, including
   `plugin-ep-webgpu/paths.txt` and template discovery.
5. Add [validation](validation.md) for the advertised contract and specializations:
   malformed inputs, scalars/empty tensors, tails, broadcasts, dtype/opset
   boundaries, feature fallbacks, and same-context input/shape changes.
   Assess CI evidence of WebGPU execution and [performance evidence](validation.md#performance-evidence).
   Record target gaps; compilation or CPU fallback is insufficient.

## Trace each changed kernel

Read changed functions in full, including their base versions, then trace:

1. **Host:** host-read inputs, validation, output shape, early exits, 64-bit narrowing.
2. **Setup:** `AddInput`/`AddOutput`, `ProgramTensorMetadataDependency`, components,
   `SetDispatchGroupSize`, `SetWorkgroupSize`, `SetSubgroupSize`, `CacheHint`,
   `AddUniformVariables`, overrides, and `SetIndirectDispatchTensor`.
3. **Shader:** `GenerateShaderCode` or `.wgsl.template` and C++ parameters;
   map every text-changing value to [cache identity](computation.md#cache-identity).
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
  [capture-safe interaction](execution.md#graph-capture).

## Registration

- Compare native `GetCapability` and plugin `ep/ep.cc` predicates for affected
  operators/attributes/dtypes/features/layouts. Shared source does not prove
  acceptance parity. Check evidence of graph acceptance/execution, rejection, and
  supported CPU partition boundaries.

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

## Cache identity

1. Enumerate every text-changing member/attribute, dtype, rank, component count,
   shape/vectorization choice, feature/vendor branch, provider option, and
   `.wgsl.template` parameter.
2. Map each to `CacheHint`, input/output `ProgramTensorMetadataDependency`
   (`Type`, `Rank`, `Shape`), components, or workgroup/subgroup size. Verify
   `CalculateProgramCacheKey` in `onnxruntime/core/providers/webgpu/program_cache_key.cc`,
   including pipeline overrides; setting an override does not prove keyed identity.
3. An unkeyed specialization is blocking: give two calls differing only in that
   value that reuse the wrong pipeline. Nightly validation after merge does not
   replace this audit.

The `WebGpuContext` owns the cache: device-fixed features/limits need no extra
key; options varying between sessions sharing it do. Check concrete loop/unroll/
`const`-array dimensions with `TypeAndRank`/`None`, component/shape branches,
new generator members/parameters, and float identity (`std::to_string` loses
precision at six decimals; encode exactly).

Use `WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES` / `AddUniformVariables` for per-run
sizes, strides, offsets, and scalar attributes. Embedding them in WGSL creates
per-value pipelines. Redundant keys need measurable compile-churn evidence.
Use [shader dumps/key validation](validation.md#shader-generation-and-cache-keys)
with differing specializations, not an unexercised dump.

## Dispatch and WGSL

### Addressing and resource access

- Use bounded `global_idx`/`workgroup_idx`/`local_idx` for linear work.
  `ProgramManager::NormalizeDispatchGroupSize` reshapes/pads grids exceeding
  `maxComputeWorkgroupsPerDimension`; raw `global_id`/`workgroup_id` addressing
  can fail. Explicit `workgroup_id.x/y/z` requires each dimension within limits.
- Disable out-of-range invocations before memory access, subject to barrier
  participation below. Return before `RunProgram` for empty output: zero dispatch
  is invalid. Check rank-0, broadcast dimensions of 1, and empty spatial dimensions.
- Bounds-check partial M/N/K loads/stores, including subgroup-matrix tiles;
  configurable `enableRobustness` is no substitute.
- Uniform-array indexing uses `GetElementAt` (`getElementAt` in templates) for vec4 packing.
- Check f32/f16 workgroup storage and invocation counts against guaranteed
  minima (16 KiB, 256 invocations) or queried adapter limits; count bindings
  against `maxStorageBuffersPerShaderStage`.
- Tensor storage access at kernel/template call sites uses **only**
  `ShaderVariableHelper` APIs: `Get`, `GetByOffset`, `GetByIndices`, `Set`,
  `SetByOffset`, `SetByIndices`, even for small tensors. No direct storage-name
  indexing. Helpers preserve segmentation above `maxStorageBufferBindingSize`,
  view offsets, and storage/value conversion; extend the API when necessary.
  Helper-internal indexing and workgroup/private arrays are exempt.
- Never bind one buffer twice when either binding is writable.

### Synchronization

- Keep barriers/subgroup operations uniform in their required scope, including
  loops. Predicate invocation-dependent bounds work rather than return before
  a workgroup barrier (`GuardAgainstOutOfBoundsWorkgroupSizes` included).
  A whole-workgroup early exit is valid when WGSL analysis proves uniformity;
  retain all participants needed by workgroup memory/barriers.
- Subgroup-size assumptions require `SetSubgroupSize` plus subgroup-size-control,
  or handling every supported size. Even fixed-size subgroup lane order need
  not match local order: derive shuffle/reduction ownership from subgroup
  coordinates, gate subgroup-ID features, and never infer rows from `local_idx`.
- Weak atomic CAS must retry until `exchanged`; old-value equality can accompany
  spurious failure. Check contention, retry updates, and float/packed bitcasts.
- `select` evaluates both operands; clamp potentially out-of-range indices.

## Numerical semantics

- Trace storage, accumulator, partial storage, divisor, and final cast separately.
  f16 overflows above 65504; use f32 reduction/dot accumulation unless the
  precision contract differs. Wider accumulators do not fix f16 Split-K partials
  or divisors. Follow `enableMatmulFp32Accumulation` through MatMul/Gemm/MatMulNBits,
  shared helpers, Conv/other callers, and generated type aliases.
- WGSL signed `/`/`%` truncate toward zero; preserve ONNX floor semantics (`Mod`,
  `fmod=0`), negative-index wrapping, and full-width integer math. Two-u32 int64
  needs proven carries/signs/comparisons/quotients; low-word conversion or
  saturation is insufficient.
- Preserve absent versus explicit-zero attributes, absent versus present optional
  slots, and domain/opset-specific scale/epsilon/axis/mask/cache semantics.
- `min`/`max`/`clamp` do not guarantee NaN propagation; `round` is half-to-even.
  Preserve NaN/infinity, saturation, and ties. Quantization thresholds and values
  need the same arithmetic domain. Cover below/at/above odd/even ties,
  non-power-of-two scales, subnormals, signed extremes, and relevant bit encodings.
- Precision before TopK/ArgMax/routing can alter indices despite small numeric
  error. Distinguish finite negative maximum from negative infinity; make padded/
  masked candidates lose before stable-index tie rules.
- Normalization needs constant/nearly constant, large-offset, cancellation, and
  permuted rows, non-unit Scale, and optional Mean/InvStd. Raw second moments or
  premature mean rounding fail even in f32. Preserve LayerNorm/RMSNorm differences
  and independent Q/K epsilons.
- Force eligible scalar/packed/subgroup-matrix/Split-K paths and sufficiently
  long reductions. Require error analysis for tolerance changes and evidence
  for performance tradeoffs.

## Tensor layout and prepacking

- Separate logical ONNX shape from packed/padded dimensions/strides. Validate
  rank, channels/groups, bias, and output geometry logically; retain original
  metadata after prepack rather than interpreting permuted filter dimensions.
- Identify operand, timing, and lifetime: initializer prepack, lazy constant
  cache, or runtime activation preparation. Check dynamic/disabled-prepack
  fallbacks, session reuse, failure cleanup, and constant ownership through capture.
- Promote rank-1 MatMul to matrix views before dispatch and restore output rank;
  check scalar/broadcast Gemm bias independently.
- Respect per-group channels/output layout: packed u32 input does not imply vec4
  output. Audit scalar/vec2/vec4 tails, odd N/K, rank layouts, bit widths, scales,
  zero points, and physical row strides in generic and hardware fallbacks.
- Trace views through `GetByOffset`/`SetByOffset`/`GetByIndices`, owner bindings,
  offsets, and segments. Pointer/subgroup-matrix/atomic paths can bypass offsets;
  check byte/element units, alignment, writable aliases, and lifetime.
- Prove initializer/dynamic/disabled-prepack paths via counts or equivalent
  evidence. Use distinguishable data across partial tiles, group boundaries,
  and every new row/column wave. Separate initialization from steady inference.
  Layout/component codegen changes need cache checks, same-context variants,
  and both invocation orders.

Inspect `nn/conv*`, `math/matmul*`, `shader_variable.cc`, and contrib
`quantization/matmul_nbits*` under WebGPU provider roots.

## Graph transformations

- Trace original graph → selector → replacement → provider → executed kernel.
  Inspect optimized domain/opset/type/layout and partitions; preferred NHWC does
  not prove execution. Count added casts/transposes/readbacks in performance claims.
- Preserve optional inputs/outputs, graph outputs, extra consumers, subgraphs,
  symbolic shapes, initializer/overridable inputs, and attribute presence/defaults.
  Reject unrepresentable semantics, including mixed dtypes or distinct Q/K epsilons.
- Preserve already-fused activation/behavior when composing another, or reject
  fusion. Test activation chains, explicit-zero parameters, optional bias slots,
  and parameter semantics independently of operator names.
- Trace legal empty LoRA/adapter branches, zero-sized weights, dynamic/overridable
  inputs, and fallbacks through the whole rewrite; dense-export assumptions fail.
- Compare optimized and unfused controls with identical exported arithmetic,
  I/O, dtype, and options, including auxiliary/cache outputs and discrete
  decisions. Test accepted/rejected patterns and prove the expected graph rewrite
  and WebGPU execution; CPU or unfused success is insufficient.
- Review other providers/minimal builds for shared optimizer edits. Align
  native/plugin predicates through [operator support](computation.md#registration).

Inspect `core/optimizer/graph_transformer_utils.cc`, `conv_activation_fusion.cc`,
`group_query_attention_pre_norm_fusion.cc`, `matmul_nbits_*fusion*`, capability
discovery, and `test/optimizer/`; recheck schemas/guards at the reviewed head.
