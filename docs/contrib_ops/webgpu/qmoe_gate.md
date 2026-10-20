# QMoE single-token routing on WebGPU

The WebGPU implementation of `com.microsoft::QMoE` selects the top `k`
experts for each token. See the [operator documentation](../../ContribOperators.md)
for the input, output, quantization, and routing contracts. This page describes
the single-token gate implementation; the operator schema is unchanged.

## Selection and normalization

When there is one token and `normalize_routing_weights=1`, the gate uses a
workgroup-wide reduction to select one expert per round. It ranks higher
router logits first, breaking ties by lower expert index, and invalidates
each selected lane before the next round. A lane that has already been
selected cannot win again, even if every remaining logit is `-inf`.
The gate writes selected indices in rank order and leaves unselected routing
weights at zero.

With optional `router_weights`, the selected weights are normalized by their
sum. Without them, the gate applies a stable softmax over the selected logits.
When `normalize_routing_weights=0`, the existing full-sort path is retained:
the softmax denominator includes **all** experts, not only the selected `k`.
Changing that denominator would change the operator's results.

One workgroup has one lane per expert. The expert count must fit both the
adapter's `maxComputeWorkgroupSizeX` and
`maxComputeInvocationsPerWorkgroup` limits; larger nonempty inputs are
rejected rather than dispatched with invalid workgroup dimensions. The
WebGPU baseline for both limits is 256, so a 512-expert configuration
requires an adapter that reports higher limits. Multi-token routing uses
the existing gate implementation and is not changed by this optimization.

The gate supports both FP16 and FP32 router logits. It selects experts
independently of whether their downstream weights are integer-quantized or
block-FP8. This change does not introduce native FP8 matrix arithmetic or
alter the expert matmul kernels.

## Validation and performance

Focused GPU tests cover odd expert counts, ties, `k` equal to the expert
count, negative-infinity logits, nonnormalized routing, FP16/FP32 gate
indices, and adapters with smaller workgroup limits. In particular,
`[0, -inf, -inf]` with `k=2` selects experts `[0, 1]`, not an invalid
sentinel index.

On H200/Vulkan, a synthetic single-token block-FP8 QMoE graph with 256
experts and top-4 routing measured 7.09 ms/run on the merged FP8 QMoE
baseline and 5.56 ms/run on an integrated branch containing this gate
change (medians of three 100-run trial averages). Trial ranges overlapped:
5.71-7.12 ms versus 5.46-5.89 ms. These are whole-graph timings, not
isolated gate-kernel timings or a full-model speedup claim.
