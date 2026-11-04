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
