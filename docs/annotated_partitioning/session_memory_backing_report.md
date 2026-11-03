# Session memory lifetime/backing diagnostics

Set `session.enable_memory_backing_report=1` to enable the first fixed-placement
framework memory report. The default is `0`. This is a diagnostic, not a budget,
allocator, placement selector, measured device peak, or no-OOM guarantee.

```python
options = onnxruntime.SessionOptions()
options.add_session_config_entry("session.enable_memory_backing_report", "1")
options.log_severity_level = 1  # INFO summaries
# For individual uncovered terms:
options.log_severity_level = 0
options.log_verbosity_level = 1
```

Non-minimal builds report at initialization and after ordinary graph execution,
after EP end-of-Run/stream cleanup and before optional arena shrinking. Graph
replay, training execution, and concurrent Runs are not modeled by this first
adapter. No additional device allocation, synchronization, or kernel declaration
call is introduced. Reporting does add host-side metadata and allocator-statistic
queries when enabled.

## What is reported

The internal `SessionState::GetMemoryBackingReport()` result contains:

| Component | Meaning |
|---|---|
| Initializers before kernel creation | Tensor payload bytes after initializer loading, before constructors or PrePack can release source weights. Copies of the same owned Tensor are charged once. |
| Initializers after PrePack | Remaining source initializer payloads. Released sources are not also counted as Run allocations. |
| L1 persistent-prepack estimates | Sum of available per-node estimates, kept separate because packed allocation identities/sharing are not declared. Missing recipes are not proof of zero. |
| L1 initialization scratch estimates | Maximum of available estimates under serial prepacking; conservative sum when parallel prepacking is enabled. This is not a complete initialization peak. |
| Known modeled Run live peak | Static tensor storage plus available L2 workspace slots, deduplicated through allocation-plan `kReuse`/`kShare` roots. Unknown roots are excluded and explicitly listed. |
| Initialization/Run physical backing peaks | **Unavailable** until the backing strategy and all relevant allocation identities/lifetimes are covered. |

L2 declarations are reused from finalization, not queried a second time.
An empty declaration remains unavailable. A nonempty zero-byte declaration is
known-zero workspace.

For sequential execution with one logical stream, allocation/release actions and
the actual execution-step order determine inclusive lifetimes. Sequential node
workspaces do not overlap; a persistent initializer/input or retained output
still overlaps workspace. Output aliases extend their source root's lifetime.
For parallel or multiple-stream execution, roots conservatively overlap for the
entire Run; a host launch/return is not a device-completion event.

Only static executable-graph tensor shapes are used. Dynamic/non-tensor/string
storage remains unavailable. `session.max_shape_override` can inform a kernel's
L2 declaration but does not prove a global tensor-storage or workload bound.
Results are conditional on the supplied declaration/workload assumptions and
one active Run.

## Physical observations are separate

`SessionState::GetMemoryBackingAllocatorSnapshots()` and the log include current
allocator-reported allocated and in-use bytes. Each allocator object is queried
once, including dedicated initializer allocators. These are **allocator-wide
snapshots**, potentially including other sessions, not session-attributed peaks.
An unsupported/default zero-filled statistic is unavailable; the current
allocator API cannot distinguish unsupported fields from a proven zero.

Never add those observations to initializer, activation, or workspace payload
bytes: an arena root can already contain all of them. A retained arena can stay
large while live payload falls. Conversely, observations do not cover private
library allocations or caller-owned buffers outside those allocators. Historical
max-in-use counters are deliberately not labeled physical peaks.

## Uncovered terms and next steps

INFO logs give known components, unavailable physical totals, and an uncovered
term count. VERBOSE logs list individual terms. The typed report retains the
complete list.

Each report covers its local graph. The post-PrePack snapshot is taken before
recursive subgraph finalization, not at a whole-session post-PrePack checkpoint.
Subgraph reports are not composed into a session peak.

This first adapter explicitly leaves unknown:

- Physical pool growth, padding, fragmentation, retained shape-specific pools,
  and non-pool reserve.
- Packed-weight ownership/sharing and constructor/runtime-preparation scratch.
- Caller-owned KV/input capacities, actual I/O locations/aliases, and output
  retention between Runs.
- Library/private EP allocations, capture-retained buffers, and concurrent or
  shared-session/device admission.
- Control-flow/subgraph residency, loop-carried state and accumulated outputs.
- Workload/phase admission beyond caller guarantees.

The next report adapters must identify these backing roots and release events
before a complete session backing peak can be produced. Existing CUDA/WebGPU
workspace-pattern integration supplies runtime reuse, not a hard budget; this
report does not assume that hypothetical workspace/activation sharing occurred.
Pre-allocation budget gates and enforceable contract/CPU-region selection remain
separate follow-ups.
