# Shaders, cache identity, numerics, and layout

**Owns:** Generated-program identity, WGSL execution/synchronization, numerics,
layouts/views, and prepacking: checks following the same shader specializations.
**Update when:** Codegen/keys, indexing/dispatch, math/formats, packing/strides,
prepacking, helpers, limits, or variants change. Host contracts live in [kernels](kernels.md).

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
