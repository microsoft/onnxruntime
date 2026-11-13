# Snapshot-aware Mamba recurrence

`com.microsoft::Mamba2Verify`, version 1, fuses causal convolution, selective
state-space recurrence and gated grouped RMS normalization. It is a built-in
CUDA contrib operator, with no custom-op plugin or external registration library.

## Contract

The implementation requires `head_dim=64`, `d_state=128`,
`heads % n_groups == 0` and 1 through 16 token rows. This Bolt-specific API
processes one sequence per invocation: rows are consecutive tokens, not
independent batch entries. Batched selectors and packed variable-length sequences
are outside this version's scope. Longer prefills must retain graph/runtime chunking;
this is not an arbitrary-length prefill kernel.

The ten inputs are:

| Input | Type and layout |
|---|---|
| `projection` | BF16 `[R, 2*di + 2*n_groups*128 + heads]`, packing gate, xBC, dt |
| `conv_state` | BF16 `[slots, di + 2*n_groups*128, 3]` |
| `ssm_state` | FP32 `[slots, heads, 128, 64]` |
| `conv_weight` | BF16 `[channels, 4]` or `[channels, 1, 4]` |
| `conv_bias` | BF16 `[channels]` |
| `dt_bias`, `A_negative`, `D` | FP32 `[heads]`; decay is negative A, not A_log |
| `norm_weight` | FP32 `[di]`, where `di = heads*64` |
| `state_row` | Device INT32 `[1]`, selecting an input snapshot |

Outputs are BF16 `Y[R, di]`, BF16 convolution snapshots, FP32 SSM snapshots,
and independently optional debug outputs BF16 `xbc[R, channels]` and FP32
pre-gating `readout[R, di]`. Outputs 0 through 2 remain required. To request
`readout` without `xbc`, leave output 3 empty and supply output 4. Omitted
intermediates use internal workspace; requesting them does not change arithmetic.

The public SSM layout remains `[slots, heads, d_state, head_dim]`, with
`head_dim` contiguous for the kernel's coalesced accesses. Changing that layout
would require a separately qualified API and kernel change, not just a shape edit.

Token row `r` writes its post-token state to snapshot row `r` for this sequence;
these rows are not per-sequence batch slots.
The caller selects the accepted/final snapshot on its next invocation;
the operator performs no accepted-length truncation or hidden state mutation.
Corresponding state inputs and outputs may alias. Trailing slots `[R, slots)`
retain their input values, including nonaliased outputs. Both state tensors must
have the same `slots` dimension, with `slots >= R`.

An out-of-range device selector produces NaNs in rows `[0, R)` of both output
state tensors, all of `Y` and any requested `xbc` and `readout`, without reading either input
state; trailing slots remain preserved. This is a non-finite output policy, not
a synchronous ORT Status error. Element-aligned convolution weights use scalar
loads when vector alignment is unavailable.

The supplied ORT CUDA stream is used throughout. There are no host selector
reads, device-to-host copies, direct CUDA allocations or synchronization.
Nonaliased state preservation uses capture-compatible device-to-device copies.
Omitted intermediate outputs use ORT's stream-aware scratch allocator. CUDA-graph
capture requires allocator warmup and stable device input/output bindings.

## Precision and readout order

Convolution SiLU rounds to BF16; recurrence and ordered readout retain FP32.
For `raw = projection_dt + dt_bias`, softplus is `log1p(exp(raw))` when
`raw <= 20` and `raw` otherwise, avoiding exponential overflow for large raw
time steps. The `time_step_min` lower clamp propagates NaN from either input.

Readout partitions the 128 state channels into four contiguous groups of 32.
Each partition accumulates `C*S` in increasing state-channel order with FP32
FMA; its partial sum is added in partition order to `D*x`. The sole exception is
`R == 1 && slots == 1`, which uses two contiguous partitions of 64. Thus
single-token replay with a multi-slot cache uses the same four-partition order
as multi-token verification, while a single-slot decode retains its two-partition
order. Changing cache capacity can change readout rounding even for one token;
the selected `state_row` index does not select the reduction path. The FP32
recurrence itself is unchanged. This readout-order and replay guarantee remains
part of the contract.

Grouped normalization computes SiLU gating, the square reduction, reciprocal
RMS scaling and multiplication by FP32 `norm_weight` before one final BF16
output rounding. In particular, it does not round to BF16 before multiplying
`norm_weight`.

## Implementation note

Eight token rows use specialized convolution and recurrence dispatches. This
specialization does not change the precision, snapshot or readout-order contract.

## Integration and tests

Schema and CUDA registration are in `contrib_defs.cc` and
`cuda_contrib_kernels.cc`. CMake globs collect the source and provider test files;
regenerate CMake when adding files. CUDA internal tests require
`onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS`.

Schema tests run without CUDA:

```text
FusedStateSpaceSchemaTest.*
```

Run the following on SM80+ and verify that every selected test executes:

```text
FusedStateSpaceTest.*
FusedStateSpaceKernelTest.Mamba*
```

Provider tests cover all five outputs, 1/3/8/16 rows, nonzero snapshot selection,
nonuniform 1024-wide normalization, multi-group states, rank-2 convolution
weights, each/both device-state aliases, invalid selectors and trailing/no-trailing
snapshot storage, time-step clamping, large raw time steps and NaN propagation.
`Y` and `xbc` are compared within one BF16 ULP per finite element, with at most
0.1% differing elements per output. CPU-only schema tests validate ranks and
statically known geometry without creating a CUDA session.
Optional-output tests cover all four requested/omitted combinations, including
the empty output-3 placeholder, state aliasing, invalid selectors, trailing slots
and no-trailing storage. CUDA-graph tests verify capture and bitwise replay against
the uncaptured run for each combination at one and eight token rows.
Internal tests cover the normalization launch with 65536 groups, offset convolution
weights at one/eight rows, bitwise multi-row versus single-row replay, and the
single-slot two-partition readout.

Component tests do not establish full-model readiness. Decoder parity,
generation, speculative state correctness, accuracy and performance require
separate qualification of the actual loaded runtime and model package.
