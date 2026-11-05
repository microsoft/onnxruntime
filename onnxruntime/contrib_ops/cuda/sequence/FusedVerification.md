# Local fused verification operators

These **provisional, local-only** `com.microsoft` version-1 names require maintainer review.
They replace four frozen plugin node types without decomposing the fused computation.
There is no plugin registration or runtime DSO dependency in these implementations.

## Base extraction and qualification status

This contribution is extracted onto PR #33400's actual base,
`648f708a73450c2f484f56c83ef9b6621e3c2b6b`, not its head. Registrations are anchored
to the base files. The adapters explicitly include the base's `core/common/narrow.h`.
No PR #33400 generic quantization operators or their schemas are required.

The only new cross-contribution dependency is
`quantization/packed_projection_impl.cuh` from the packed projection contribution.
It supplies inline IQ2 decode/MMA/reduction primitives; no dense projection kernel
registration or implementation object is needed by these fused operators.

The original head-worktree implementation passed 12 standalone bit-exact GPU cases
on the reserved H100, including capture, state aliasing and accepted-row replay.
The base extraction preserves CUDA math byte-for-byte. The base extraction was
CPU-only: **base compilation, full ORT linking and provider-test execution are
NOT_RUN**. The C++ provider tests were compiled to objects in the original worktree,
not linked or executed as ORT operator tests.

| Frozen node | Built-in replacement | Inputs / outputs |
|---|---|---|
| `bolt::MambaVerify` | `Mamba2Verify` | 10 / 5, original order |
| `bolt::MoeRoute` | `BiasCorrectedMoERoute` | 3 / 3, original order |
| `bolt::Relu2` | `ReluSquared` | 1 / 1 |
| `bolt_iq2_source::MoeVerify`, `iq2=2` | `MoEPackedIQ2XXS` | 7 / 4, original order; remove `iq2` |

Schemas document types, layouts, attributes, and equations. All inputs are device tensors,
including the Mamba snapshot selector and MoE expert indices.
`Mamba2Verify` keeps `heads`, `head_dim`, `d_state`, `n_groups`, `epsilon`,
and `time_step_min`. The router keeps `topk` and `scale`; the MoE keeps `topk`.

## Frozen provenance and arithmetic

Mamba, routing, and squared ReLU were ported from
`export-nemotron-pr-20261030/src/bolt_ops.cu`, SHA256
`1f587d9eacc356805c37d19e94e7aa4465dd50302576b2ca4b9579d7c5ae8ba5`.

The IQ2 plugin comes from `IQ2_XXS_cuda_bolt/iq2-compat/iq2_plugin.cu`, which includes
**its adjacent** `bolt_ops.cu`, SHA256
`4a539c0e82b8ea48c2f3adc51dc324d0f792554bca2c46706b972ca5afc97345`.
The latter differs from `src/bolt_ops.cu`: IQ2 rounds each complete grid-times-scale
operand to BF16, rather than rounding the scale first. Native-GGUF reference
activation quantization is not used.

IQ2 decode/MMA/reduction helpers are shared with packed dense projections in
`quantization/packed_projection_impl.cuh`. The MoE preserves the frozen default
FC1 two-tile/four-warp and FC2 two-tile/eight-warp reduction orders, BF16 projection
boundaries, squared ReLU boundary, and ordered FP32 weighted sum. The plugin's
`BOLT_MOE_CFG` experimental environment tuning is intentionally not an operator contract.

## State and prefill

Mamba supports 1 through 16 rows, with the original specialized eight-row verification
dispatch. A causal prefill chunk uses the same recurrence. There is no new arbitrary-length
prefill kernel: longer sequences must retain the existing graph/runtime chunking.
IQ2 MoE supports 1 through 8 rows and at most 128 routed pairs, as in the frozen source.

The selector chooses one input snapshot. Each processed row writes its **post-token**
convolution and SSM states to the corresponding output row. Selecting an accepted prefix
is the caller's responsibility on the next invocation. State layout is
`[slots,heads,128,64]`, with head dimension contiguous; convolution history is
`[slots,channels,3]`. Initializing an unselected snapshot must not affect computation.

Corresponding state inputs/outputs may alias (`MayInplace(1,1)` and `(2,2)`).
Nonaliased outputs copy the original device buffers before updating the first R slots.
Trailing slots are therefore preserved rather than left unspecified as in uninitialized
plugin outputs. MoE similarly initializes unused unique-expert metadata slots to -1.
These changes define previously unwritten output storage without changing any computed row.
Selectors must be in range; routed expert indices must be in range and distinct within a row.
Device-side guards define bounded invalid-input behavior without copying selectors to the host:
an out-of-range Mamba selector writes NaNs to Y, xbc, readout and the first R snapshot
slots, preserving trailing slots. Neither recurrence kernel reads the selected input state.
For MoE, an out-of-range or duplicate expert ID invalidates the entire token row:
all its hidden, expert_output and Y values become NaN, and its pairs are excluded from
metadata before any expert weight access. Valid rows still execute normally. Invalid
rows are not clamped, skipped as if valid, or reported through a device assertion.
These are deterministic non-finite output policies, not synchronous ORT Status errors.

Kernels use the supplied ORT CUDA stream. There are no device-to-host copies,
host-side routing, per-call CUDA allocation, synchronization, or host inspection of selectors.
Device-to-device state copies are capture-compatible.

## Integration and tests

Schema registration is in `core/graph/contrib_ops/contrib_defs.cc`; CUDA registrations are
in `contrib_ops/cuda/cuda_contrib_kernels.cc`. Existing recursive CMake globs collect the
new `.cc`, `.cu`, and headers; existing provider-test globs collect the two test files.
No new build option is required.

Focused provider tests:

```text
FusedStateSpaceTest.*
PackedMoETest.*
```

They cover post-token recurrence/readout, nonzero accepted snapshot selection, trailing
state slots, 1/3/8/16-row Mamba paths, 1024-wide gated normalization, corrected routing
versus unbiased weights, ties, nonunit scale, squared ReLU, empty tensors, reused experts,
and raw FP16 super-scales carried in BF16 or FP16 storage.
Additional review regressions cover eligible negative-infinity router scores, both invalid
selector bounds, multi-group nonuniform states and rank-2 convolution weights, explicit
IOBinding device aliasing of each/both state buffers at R=8 with selector=3, sparse
row-distinct expert routes, multiple FC2 superblocks, topk=9 and cancellation-sensitive
ordered FP32 sums. Source-exact MoE checks set both tolerances to zero after adding outputs.
These review regressions have not yet been compiled or run on a GPU.

### Standalone kernel validation

The full Mamba operator still requires `head_dim=64, d_state=128`. The shared
`LaunchMamba2VerifyNormalization` entry point isolates the production normalization
launch so grid limits can be tested without allocating full recurrent states.
Compile `fused_state_space_op_test.cc` with `ORT_FUSED_STATE_SPACE_NORM_TEST` and
link `fused_state_space_impl.cu`, CUDA runtime and `gtest_main` to run
`FusedStateSpaceKernelTest.MambaNorm65536Groups` without ORT libraries.
This alternate test mode does not compile the provider tests in that source file.
It uses `heads=n_groups=65536`, synthetic head/state dimensions of 1, R=slots=1,
width=65536 and projection stride=327680. No convolution or SSM kernel runs in this
normalization-only fixture. The test invokes the same launch used by the full operator;
restoring `dim3(rows, groups)` fails CUDA's grid-y limit before output comparison.

For a standalone harness calling the other launchers directly:

- Require SM80+, BF16 operands and device-resident selectors/indices. Full Mamba
  uses head/state dimensions 64/128, 1..16 rows and slots >= rows; MoE uses
  X width 1024, 1..8 rows, at most 128 pairs, E >= topk and I divisible by 256.
- Before a nonaliased Mamba call, copy the entire input state buffers to the
  snapshot outputs. This preservation copy belongs to `ComputeInternal`, not
  `LaunchMamba2Verify`. Restore both inputs before each alias/nonalias comparison.
  The alias fixture uses R=8, slots=10, selector=3, heads=4 and groups=2.
- The nonuniform Mamba oracle rounds convolution activation to BF16, retains FP32
  recurrent states and uses two ordered readout partitions for R=1, four otherwise.
  Check all five outputs and preserve slots R..slots-1, including invalid selectors.
- IQ2 super-scales are raw FP16 bits even when stored in BF16-typed buffers.
  Decode and round each complete grid-times-scale operand to BF16. Round FC1 to
  BF16 before squared ReLU, round hidden to BF16, and round FC2 to BF16 before
  storing its FP32-valued expert output. Do not reassociate the ordered FP32
  weighted sum or round its running accumulator to BF16.
- Sparse MoE fixtures use three distinct activation rows, E=topk+3, I=512 and
  topk=2/9. Cancellation fixtures use I=256, topk=3/9, X=1/1024, grid zero and
  super-scales=1/16: hidden=1/256 and every expert output=1/16. The first three
  weights are [2^28,48,-2^28], with remaining weights zero. Their weighted terms
  [2^24,3,-2^24] produce ordered FP32 Y=4, reversed Y=3 and BF16 running-sum Y=0.
  Compare the final BF16 values exactly; check actual BF16 NaNs explicitly.
