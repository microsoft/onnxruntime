# Dispatch and WGSL

**Owns:** Shader addressing, synchronization, and resource limits.
Host validation belongs in kernel-contract; cache reuse in cache-identity;
arithmetic in [numerical semantics](numerical-semantics.md).
**Update when:** Dispatch normalization, shader helpers, limits, or supported
numeric behavior change.

## Addressing and resource access

- Linear work must use `global_idx`, `workgroup_idx`, or `local_idx`, with
  every invocation bounded. `ProgramManager::NormalizeDispatchGroupSize`
  reshapes and pads grids exceeding `maxComputeWorkgroupsPerDimension`.
  Addressing by `global_id` or `workgroup_id` can then be wrong. A kernel using
  `workgroup_id.x`, `.y`, or `.z` must keep each dimension within the limit.
- Every invocation beyond the logical size must be inactive before any memory
  access. A zero dispatch is an error: return before `RunProgram` for an empty
  output. Check rank-0 scalars, dimensions of 1 in broadcasting, and empty
  spatial dimensions.
- Bounds-check loads and stores for partial M, N, and K tiles, including subgroup
  matrix tiles. Robust buffer access is configurable via `enableRobustness`;
  it does not replace bounds checks.
- Index uniform arrays with `GetElementAt` (`getElementAt` in templates) to
  honor vec4 packing.
- Compute workgroup storage for both f32 and f16 tile variants. Guaranteed
  minimum limits are 16 KiB storage and 256 invocations per workgroup; stay
  within these or the limits queried from the adapter. Count bindings against
  `maxStorageBuffersPerShaderStage`.
- Access tensor storage buffers only through `ShaderVariableHelper` accessors
  (`Get`, `GetByOffset`, `GetByIndices`, `Set`, `SetByOffset`, `SetByIndices`)
  at kernel/template call sites. Do not handwrite `input[index]`, `output[index]`,
  or equivalent storage-name indexing, even for small tensors. The helpers
  preserve binding segmentation above `maxStorageBufferBindingSize`, buffer-view
  offsets, and storage/value conversions. Extend the helper API when an operation
  needs unsupported access instead of bypassing it. This rule permits storage
  indexing emitted inside the helper implementation and does not apply to
  workgroup/private arrays.
- Do not bind one buffer twice in a dispatch when either binding is writable;
  WebGPU rejects aliasing writable bindings.

## Synchronization

- `workgroupBarrier`, `storageBarrier`, and subgroup operations need uniform
  control flow. Check divergent branches and invocation-dependent loop counts.
  An early return, including `GuardAgainstOutOfBoundsWorkgroupSizes`, before a
  barrier is invalid: predicate work instead. Kernels using workgroup memory
  or barriers must not discard participating invocations prematurely.
- A subgroup-size assumption needs `SetSubgroupSize` and the subgroup-size-control
  feature, or code handling every supported size.
- Subgroup lane order need not match local invocation order, even with a fixed
  subgroup size. Derive shuffle/reduction ownership from subgroup coordinates
  and gate required subgroup-ID features; do not infer rows from `local_idx`.
- Atomic compare-exchange loops must retry until `exchanged` is true. Weak CAS
  may fail spuriously even when the returned old value equals the expected one;
  value equality alone does not establish a successful store. Check contention,
  retry updates, and bitcast representations for float/packed reductions.
- WGSL evaluates both operands of `select`; an unselected out-of-range operand
  still reads memory. Clamp the index.
