# CUDA workspace in activation memory patterns

Set `session.enable_static_workspace_preallocation=1` to opt supported in-tree CUDA
MatMulNBits and GroupQueryAttention kernels into run-scoped memory-pattern reuse.
Sequential execution is required. Memory-pattern optimization must remain active;
otherwise kernels retain dynamic scratch.

## Allocation and lifetime

Initialization records each participating kernel's Level-2 workspace requirement.
It does **not** allocate persistent workspace. This pilot supports exactly one
slot per opted-in kernel.

The first compatible full `Run()` uses dynamic scratch and traces workspace
allocation/free alongside activations. Workspace uses negative pattern IDs,
separate from OrtValue indices. The cached pattern can therefore reuse storage
between workspace and non-overlapping activations or other kernels' workspace.

Subsequent compatible runs allocate their own activation-pattern backing buffer.
`GetPreallocatedWorkspace()` returns a non-owning pointer into that current frame,
with the declared alignment. Kernels must not free or retain it between invocations.
Alignment padding is not additional usable capacity: requests exceeding the
declared size use dynamic scratch.

A missing pattern, missing block, unavailable backing allocation, or oversized
request also uses dynamic scratch. CUDA plugin kernels retain their existing
allocation path; no public plugin workspace ABI is introduced here.

## GQA envelope and sizing

The existing `ep.cuda.gqa_workspace_max_total_sequence_length` option supplies
the non-windowed KV-length envelope. Unset and explicit `"0"` both mean
unspecified. Windowed declarations use the cache-capacity bound instead.

GQA uses the shared mainline preparation/backend recipes and checks each planned
suballocation. Non-windowed runs exceeding the configured scalar envelope use
dynamic scratch, even if their input shapes match a cached pattern. Such a first
run still traces the declared slot, so a later in-envelope run can consume it.

This pilot uses the conservative declaration as cached capacity; it does not
implement pattern-specific route sizing. An out-of-envelope dynamic allocation
can coexist with the cached backing block. This is placement/lifetime reuse, not
a peak-memory budget or no-OOM guarantee.

## Boundaries

Minimal builds retain dynamic scratch. In training-enabled builds, full execution
traces combined workspace/activation patterns instead of caching an activation-only
static pattern. Partial-training execution does not publish that combined cache,
so partial-only workloads do not gain the same reuse.

The existing planner disables memory patterns for multiple streams on the same
device. Parallel execution, cross-stream workspace lifetimes, concurrent-Run
budgeting, and CUDA graph-capture guarantees are outside this pilot's scope.
