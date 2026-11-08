# Snapshot-aware Mamba recurrence

`com.microsoft::Mamba2Verify`, version 1, fuses causal convolution, selective
state-space recurrence and gated grouped RMS normalization. It is a built-in
CUDA contrib operator, with no custom-op plugin or external registration library.
This layer adds no projection, routing, MoE or squared-ReLU operators.

## Contract

The provisional implementation requires `head_dim=64`, `d_state=128`,
`heads % n_groups == 0` and 1 through 16 token rows. Eight rows use the preserved
specialized dispatch. Longer prefills must retain graph/runtime chunking;
this is not an arbitrary-length prefill kernel.

The ten inputs and five outputs retain their frozen order:

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
BF16 `xbc[R, channels]` and FP32 pre-gating `readout[R, di]`.

Each token writes its post-token state to the corresponding output row.
The caller selects the accepted/final snapshot on its next invocation;
the operator performs no accepted-length truncation or hidden state mutation.
Corresponding state inputs and outputs may alias. Trailing slots `[R, slots)`
retain their input values, including nonaliased outputs.

An out-of-range device selector produces NaNs in computed rows without reading
either input state; trailing slots remain preserved. This is a non-finite output
policy, not a synchronous ORT Status error. Element-aligned convolution weights
use scalar loads when vector alignment is unavailable.

The supplied ORT CUDA stream is used throughout. There are no host selector
reads, device-to-host copies, per-call CUDA allocations or synchronization.
Nonaliased state preservation uses capture-compatible device-to-device copies.

## Provenance and precision

The Mamba implementation is retained from the source-exact operator in #33940,
commit `491dc043952066267b2212ba979f0325da318573`, without recurrence,
dispatch or normalization arithmetic changes. Unused routing and squared-ReLU
helpers are omitted.

Its frozen source is `export-nemotron-pr-20261030/src/bolt_ops.cu`, SHA256
`1f587d9eacc356805c37d19e94e7aa4465dd50302576b2ca4b9579d7c5ae8ba5`.
Convolution SiLU rounds to BF16; recurrence and ordered readout retain FP32.
Grouped normalization gates, reduces and multiplies FP32 gamma before the final
BF16 output rounding. This describes the frozen Bolt contract, not a claim of
independent HuggingFace model parity.

## Integration and tests

Schema and CUDA registration are in `contrib_defs.cc` and
`cuda_contrib_kernels.cc`. Existing CMake globs collect the new source and provider
test files; regenerate CMake when adding them. CUDA internal tests require
`onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS`.

Run the following on SM80+ and verify that every selected test executes:

```text
FusedStateSpaceTest.*
FusedStateSpaceKernelTest.Mamba*
```

Provider tests cover all five outputs, 1/3/8/16 rows, nonzero snapshot selection,
1024-wide normalization, multi-group nonuniform states, rank-2 convolution
weights, each/both device-state aliases, invalid selectors and trailing/no-trailing
snapshot storage. Internal tests cover the production normalization launch with
65536 groups and offset convolution weights at one/eight rows.

Component tests do not establish full-model readiness. Decoder parity,
generation, speculative state correctness, accuracy and performance require
separate qualification of the actual loaded runtime and model package.
