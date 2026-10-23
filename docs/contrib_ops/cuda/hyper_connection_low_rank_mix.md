# HyperConnectionLowRankMix CUDA execution

The existing `com.microsoft::HyperConnectionLowRankMix` operator accepts FP32,
FP16 and BF16 with unchanged inputs, attributes and ONNX rounding points.
Original weights remain in their ordinary row-major layouts. Eligible constant
weights acquire a private prepacked copy during session initialization; no model
rewriting or allocation/packing during Compute or graph capture is required.

## Constant-weight tile packing

`PrePack` accepts FP16/BF16 down/up initializers only for four branches, hidden
2560, rank 320 on SM90+, and only when all four weight inputs are constant.
Both CPU initializers and device-resident initializers (including GPU
`AddInitializer` tensors) are supported. CPU inputs use a device staging buffer;
device inputs are read directly. Packing uses an initialization-only stream and
synchronizes before publishing the packed pointer. CUDA failures propagate.

Each packed weight contains 160 tiles of 40,960 bytes: the exact `uint4` shared
memory image for one CTA. Down tiles are ordered by K-split then rank-column
tile; up tiles are ordered by feature tile. The cooperative kernel loads these
images with coalesced 16-byte `cp.async` copies. Both phase copies start before
the PDL wait and complete before the corresponding shared-memory tile is consumed.
Arithmetic, reduction and rounding order remain unchanged.

A bulk-copy/mbarrier trial passed the focused CUDA tests but did not improve
repeatable full-model throughput, so it is not enabled. The retained implementation
also avoids persistent tagged handoff state and its concurrent-run lifetime requirements.

Extra persistent allocation is bounded to **6,553,600 bytes per weight,
13,107,200 bytes (12.5 MiB) per eligible operator**. While packing a CPU weight,
one additional 6,553,600-byte staging allocation is live. Across a model this
scales with the number of eligible HC operators. Original initializers are
retained (`is_packed=false`) for all fallback paths. Unsupported shapes,
architectures, dynamic inputs, absent packed copies, and insufficient cooperative
residency keep their existing ordinary-weight execution.

## Small-row fused execution

For aligned FP16/BF16 tensors on SM80+, the fused implementation supports
1–8 rows, 1/2/4/8 branches, hidden widths divisible by 8 up to 4096, and
ranks divisible by 8 from 8 through 512.

For **one or two rows**, it first attempts a single cooperative kernel. This
includes the BF16 4-branch, hidden-2560, rank-320 decode shape, with or without
the optional previous-block post-mix. The down/inject phase and the up/pre-mix
phase reuse the original arithmetic bodies and deterministic reductions.
For four branches and rank 320, both phases' weights are prefetched before the
down phase completes, retaining independent shared-memory storage (86,560 bytes
for one row, 91,200 bytes for two rows). This hides the up-weight gather behind
the down computation instead of adding its latency after the grid barrier.
Other shapes union the phases' storage since their weight lifetimes do not overlap.
Optional pass-through streams are also written by this kernel, without a
separate copy.

The launch requires `cudaDeviceProp::cooperativeLaunch` and sufficient residency
for the **entire** grid. The implementation queries
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the exact instantiated
kernel, including its registers and requested dynamic shared memory, and checks
the block count against active blocks per SM times the device's SM count.
The decode shape needs 168 CTAs; the up phase uses 160 of those CTAs.
Every CTA, including idle ones, participates in `cooperative_groups::grid_group::sync`.
No unchecked global spin barrier, persistent counters, tags, reset kernel,
or externally retained workspace state is used.

Unsupported cooperative devices and insufficient residency fall back to the
existing two-kernel fused implementation. Rows 3–8 also use that implementation.
Other shapes/types use the existing general cuBLAS implementation.
Launch errors propagate rather than attempting a fallback inside a failed
CUDA graph capture.

Cooperative launches support CUDA stream capture and graph replay. Scratch
contains only intermediates that are fully overwritten per invocation.
On SM90+ with CUDA 12.3+, `cudaLaunchKernelEx` enables both cooperative launch
and programmatic stream serialization only when constructor `TryGetConstantInput`
checks confirm that inputs 1–4 (norm/down/up/inject) are all constant. No host
dereference or CPU-memory restriction is imposed on the returned tensors.
Dynamic or unrecognized weight inputs disable PDL for both cooperative and
two-kernel launches, so stream ordering resolves producer completion before
any weight read. Direct launch callers default to this conservative mode and
may set `weights_ready` only for immutable/already-ready weights.
Down/inject weights and, for C4/R320,
up weights are prefetched before `griddepcontrol.wait`. Activation reads use
volatile global-load instructions after that wait to prevent compiler hoisting.
Each CTA triggers dependents early; following consumers must still resolve their
own programmatic dependencies before consuming HC outputs. SM80 uses ordinary
cooperative launches. Without packed copies this retains ordinary-layout gathers. Even with packed
copies, identical scheduling or performance to the reference is not implied.
No allocations or synchronization are introduced in the launch helper.
The four-branch/rank-320 specialization opts into its larger dynamic shared-memory
allocation with `cudaFuncSetAttribute`; it falls back when the device's per-block
opt-in limit or grid residency is insufficient. Attribute and occupancy queries
run during host dispatch, including capture, but not during graph replay.
The kernel uses CUDA's `<cooperative_groups.h>` and ordinary CUDA runtime
cooperative-launch APIs; it does not introduce a new operator domain or require
an external device library.

## Tests

`HyperConnectionLowRankMixTest.FusedDecodeShapeRank320` checks one/two-row
FP16/BF16 against the independent CPU reference, both with and without prologue.
`HyperConnectionLowRankMixCudaKernelTest.*` additionally compares FP16/BF16 outputs
bit-for-bit to forced two-kernel execution, with repeated ordinary launches and
graph replays, changing inputs and poisoning scratch/output buffers between
replays. It checks generic shapes and forced device/residency/row-count fallbacks.
The captured chain includes a test-only independently completing producer that
triggers dependents before writing the changed activation input; HC must wait
for producer completion. Compile/link the companion
`hyper_connection_low_rank_mix_test_producer.cu` with the direct kernel tests.
`DynamicWeightGraphReplay` additionally changes down weights in that producer,
poisons their destination, and validates FP16/BF16 safe cooperative and forced
two-kernel execution against the two-kernel reference.
`PackedWeightLayout` checks every packed element against its original offset;
`PackedWeightGraphReplay` checks packed FP16/BF16 one/two-row execution with and
without prologue and forced fallback against the original two-kernel path.
`HyperConnectionLowRankMixTest.ConstantWeightsPrePackAndFallback` covers the
operator-level PrePack lifecycle, larger-row fallback and unsupported shapes.
The internal CUDA tests require `onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS=ON`;
the standard non-plugin test target discovers the dedicated kernel test through
its existing `cuda_kernels/*.cc` glob.
