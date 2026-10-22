# Projected sigmoid gates and shared-expert preparation

The CUDA BF16 `SigmoidGateMul` operator retains its existing packed, broadcast
and SiLU modes. Two appended optional inputs support shared-expert fusion:

- Input 3, `prefill_numerics`, is a CPU INT32 tensor with one value, 0 or 1.
  Mode 1 selects BF16 sigmoid intermediates; omitting it preserves the existing
  `sigmoid_in_fp32` attribute.
- Input 4, `gate_weight`, is BF16 `[input_width, 1]`. It requires
  `broadcast_gate=1` and `silu_gate=0`. In this mode input 1 is the projection
  activation with the same leading dimensions as input 0, rather than an
  already projected scalar gate. Its final dimension is `input_width`.

The projected-gate implementation computes the dot product in FP32 using a
256-thread CUB reduction, rounds to BF16, applies the selected sigmoid
arithmetic, and preserves the existing BF16 multiply/residual rounding.
The residual, when supplied, must match input 0.

`LaunchPrepareSharedExpert` shares the same projected-gate reduction and BF16
sigmoid arithmetic. One launch prepares the SiLU/up activation and the scalar
row gates for `MatMulBlockQuantizedFp8Weight(shared_expert=1)` in companion
PR #33473. No behavior changes occur without the new inputs or opt-in mode.

The shared-expert native tests compare these paths bit-exactly with the
unfused CUDA operator sequence, including prefill and fallback dispatch.
