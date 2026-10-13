# Optional combined MatMulNBits and quantized LoRA fusion

This is an opt-in prototype for quantized models with runtime-selectable
low-rank adapters. It does not change model weights or enable adapter loading.
The feature is disabled by default and requires maintainer review before
being treated as a supported optimization.

## Model contract

The supported graph computes:

```text
base = MatMulNBits(X, B, scales, zero_points?, g_idx?, bias?)
W_A = DequantizeLinear(Q_A, S_A, axis=0, block_size=32)
W_B = DequantizeLinear(Q_B, S_B, axis=0, block_size=32)
low_rank = MatMul(X, W_A)
delta = MatMul(low_rank, W_B)
scaled_delta = Mul(delta, constant_scalar)  # optional
Y = Add(base, scaled_delta)
```

The adapter parameters remain graph inputs with matching empty initializers:
INT8 `Q_A[K, 0]`, `Q_B[0, N]`, and FP32 scales
`S_A[ceil(K/32), 0]`, `S_B[0, N]`. Active ranks are symbolic, including the
scale-group count. Adapter scaling is folded into S_B. Default tensors have
no external or embedded data.

The transformer replaces both the base projection and the LoRA branch with
one internal `com.microsoft.MatMulNBitsLora` operator. Its first six inputs
match MatMulNBits: `X, B, scales, zero_points?, g_idx?, bias?`. They are
followed by `Q_A, Q_B, S_A, S_B`. All base attributes are preserved.
FP16 graphs include a Cast after
each FP32 dequantization; those casts are absorbed too.
It also recognizes the equivalent `Gemm(low_rank, W_B, base)` form when
alpha and beta are both one and neither input is transposed.

## Enabling the prototype

Enable extended graph optimizations and set the session option before
creating the session:

```python
options = onnxruntime.SessionOptions()
options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
)
options.add_session_config_entry("optimization.enable_matmul_nbits_lora_fusion", "1")
```

CPU supports FP32. WebGPU supports FP32 and FP16, subject to the existing
base operator's provider constraints. FP16 CPU graphs are not fused.
The earlier `optimization.enable_lora_mul_add_fusion` key is accepted as an
alias when the combined-operator key is absent.

## Scalar scaling

`lora_alpha` is a finite constant scalar, defaulting to one. An explicit Mul
may place its scalar on either side and use a rank-zero scalar or size-one
vector initializer. The initializer must not be an overridable graph input.
Existing adapters with scaling already folded into S_B use `lora_alpha=1`.
Both forms may be used together without dropping either scale.

Runtime-changing scalars, nonfinite scalars, nonscalar broadcasting and
non-unit Gemm alpha/beta remain unfused. FP16 scaling is materialized at the
delta's dtype before addition, preserving the separate Mul rounding point.

## Execution behavior

Rank is R in `Q_A[K,R]` and `Q_B[R,N]`, not the base projection's K.
The operator validates adapter shapes/scales, then invokes the existing
MatMulNBits kernel directly into the final output. At R=0 it returns
immediately: no adapter dequantization, casts, LoRA allocations, MatMul,
Mul/Add dispatches, second host operator invocation or base-result copy.

At positive rank it computes the two low-rank MatMuls, applies any explicit
scalar, and adds the delta to that same output. It does not merge adapter
weights into the quantized weights or cache mutable adapter weights as
constant MatMul operands.

The CPU implementation inherits MatMulNBits's base prepacking and shared
prepacked-buffer support; adapter inputs are never prepacked. WebGPU reuses
the base kernel's existing algorithm selection. The activation must not
alias the output, because active LoRA still needs the original X.
One graph operator does not imply one active-rank GPU dispatch.

The original adapter input names and model metadata are retained. Vector,
matrix, and batched activations keep their original output shapes.

Fusion requires the same X for both projections, matching execution
providers and single-use intermediate
values that are not graph outputs. Nonempty defaults, unsupported dtypes,
shared intermediates, broadcasted base results, prepacked-layout base weights,
unsupported zero points or quantization layouts, and non-unit or transposed
Gemm variants are left unchanged.

Only `weight_prepacked=0` is supported. Shape inference and both kernel
constructors reject every nonzero value, including CUDA-specific layouts,
even for a directly authored operator or an inactive adapter.

Reduced builds loading an optimized model need the new operator in their
required-operator configuration. The transformer itself is available in
full builds; it is not registered for minimal-build optimization replay.

## Review and validation scope

The provider tests cover actual base projection plus inactive and active
rank, bias, empty output, vectors, rank validation, explicit scalar
scaling and FP16 WebGPU capture/replay with stable GPU-bound buffers.
Structural optimizer tests cover the supported patterns, unsafe-fusion
guards, retained inputs, provider/dtype eligibility, and explicit opt-in.

The revised combined operator was validated on native Windows builds with
full CPU types and bundled Dawn/D3D12, passing all 24 provider tests and all
seven optimizer tests without skips.

On a representative 28-layer FP16 quantized model (196 fused projections) on
WebGPU (D3D12, NVIDIA RTX 4070):
- **Inactive vs Bare Baseline:** 0.99x–1.03x latency (~1.0x neutral) and
  identical 595 dispatches per step, confirming zero overhead when adapters
  are not active.
- **Inactive vs Unfused Empty Slots:** Eliminates 392 empty GPU dispatches per
  step, achieving a 1.46x–1.56x speedup (40%–43% latency drop).
- **Active Adapters:** Eliminates 392 intermediate dispatches, achieving a
  1.16x–1.20x speedup across prefill and decode workloads.
- **Numerical Parity:** Bit-for-bit exact FP16 logit equality (max diff = 0.0)
  across all 200,029 elements for all inactive and active workloads.

Unfused fallback graphs still require the separate empty-rank correctness fix
in PR 33171.
