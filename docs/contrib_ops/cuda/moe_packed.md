# MoEPacked

`com.microsoft.MoEPacked` version 1 initially supports
`quant_type="iq2_xxs"`, BF16 CUDA inference on SM80+, and squared ReLU.
It consumes native GGUF tensor block bytes, not a GGUF file.

Expert input `X` and output `Y` have shape `[R,1024]`. `FC1` has UINT8
shape `[E,I,264]`; `FC2` has UINT8 shape `[E,1024,(I/256)*66]`.
Each contiguous 66-byte block contains a little-endian FP16 super-scale
and 64 original IQ2_XXS record bytes. Scales are not separate inputs.
Positive intermediate dimension `I` must fit int32 and be divisible by
256. Attribute `k` is in `[1,32]`, `1 <= R <= 8`, `R*k <= 128`, and
positive int32 expert count `E` must be at least `k`.

Routing inputs are mutually exclusive:

- Preselected `indices` are INT32 `[R,k]`; `weights` are FP32 `[R,k]`.
  Their order and values are preserved without renormalization. Negative
  and cancellation weights are valid. Duplicate or out-of-range expert
  indices produce NaN for the entire affected token row, not other rows.
- Fused `router_input` is BF16 `[R,H]`, `router_weight` is BF16 `[E,H]`
  and `router_bias` is FP32 `[E]`. Require all three, `E <= 512` and
  positive int32 `H` divisible by 256. Selection ranks FP32 sigmoid
  scores plus correction bias, ties by lower expert index. Mixing uses
  the uncorrected selected sigmoid scores, divided by their ordered
  FP32 sum plus `1e-20`, then multiplied by finite `router_scale`.
  This attribute defaults to 1 and must equal 1 in preselected mode.

Each complete dequantized IQ2_XXS operand rounds to BF16 before FP32
MMA/reduction. FC1's result rounds to BF16 before FP32 ReLU and squaring;
the hidden activation rounds to BF16. FC2's result rounds to BF16 before
original-order FP32 expert mixing. The mixture rounds to BF16 only once
at the end. Workspaces and fused routing outputs remain internal.

The grouping, reduction and router arithmetic originate from the
qualified implementation at
`491dc043952066267b2212ba979f0325da318573`. The shared IQ2 decoder/MMA
helper is supplied by the separate `MatMulPacked` change. Native block
loads never assume record alignment; aligned activation loads are
selected independently. No Mamba implementation or additional public
router operator is introduced.

There is no CPU kernel, offloading, expert tracking, additional
activation function or large-row prefill support in this initial version.
Focused tests do not by themselves establish full-model or performance
readiness.
