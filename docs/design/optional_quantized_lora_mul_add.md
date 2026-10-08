# Optional quantized LoRA Mul/Add fusion

This is an opt-in prototype for quantized models with runtime-selectable
low-rank adapters. It does not change model weights or enable adapter loading.
The feature is disabled by default and requires maintainer review before
being treated as a supported optimization.

## Model contract

The supported graph computes:

```text
base = MatMulNBits(A, B, scales, zero_points?, g_idx?, bias?)
W_A = DequantizeLinear(Q_A, S_A, axis=0, block_size=32)
W_B = DequantizeLinear(Q_B, S_B, axis=0, block_size=32)
low_rank = MatMul(A, W_A)
delta = MatMul(low_rank, W_B)
Y = Add(base, delta)
```

The adapter parameters remain graph inputs with matching empty initializers:
INT8 `Q_A[K, 0]`, `Q_B[0, N]`, and FP32 scales
`S_A[ceil(K/32), 0]`, `S_B[0, N]`. Active ranks are symbolic, including the
scale-group count. Adapter scaling is folded into S_B. Default tensors have
no external or embedded data.

The transformer leaves the base projection unchanged and replaces only the
LoRA branch with `com.microsoft.LoraMulAdd(base, A, Q_A, Q_B, S_A, S_B)`.
The base producer need not be MatMulNBits. FP16 graphs include a Cast after
each FP32 dequantization; those casts are absorbed too.
It also recognizes the equivalent `Gemm(low_rank, lora_B, base)` form when
alpha and beta are both one and neither input is transposed.

## Enabling the prototype

Enable extended graph optimizations and set the session option before
creating the session:

```python
options = onnxruntime.SessionOptions()
options.graph_optimization_level = (
    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
)
options.add_session_config_entry(
    "optimization.enable_lora_mul_add_fusion", "1"
)
```

CPU supports FP32. WebGPU supports FP32 and FP16, subject to the existing
base operator's provider constraints. FP16 CPU graphs are not fused.
The previous local `optimization.enable_matmul_nbits_lora_fusion` key is
accepted as an alias when the new key is absent.

## Execution behavior

Rank zero returns the existing base result without dequantization, casts,
LoRA allocations, or adapter MatMul/Add dispatches.
Positive rank computes the two low-rank MatMuls and adds their separate
delta to the base result. The implementation does not merge the adapter
weights into the quantized weights or cache mutable adapter weights as
constant MatMul operands.

The kernels declare MayInplace for the base/output pair, not unconditional
Alias. When ORT's allocation plan safely reuses the base buffer, the empty
path performs no copy. Otherwise it copies the base value to its output.
Active addition uses the output's single read/write binding when in-place,
avoiding duplicate WebGPU bindings to the same buffer.

The fused operator still has a host-side invocation. Zero LoRA GPU dispatch
does not imply the same host operator count as a bare model.

The original adapter input names and model metadata are retained. Vector,
matrix, and batched activations keep their original output shapes.

Fusion requires matching execution providers and single-use intermediate
values that are not graph outputs. Nonempty defaults, unsupported dtypes,
shared intermediates, unsupported zero points or quantization layouts, and non-unit or
transposed Gemm variants are left unchanged.

Reduced builds loading an optimized model need the new operator in their
required-operator configuration. The transformer itself is available in
full builds; it is not registered for minimal-build optimization replay.

## Review and validation scope

The provider tests cover inactive and active rank, bias, empty output,
vector inputs, rank validation, and FP16 WebGPU graph-capture replay.
Structural optimizer tests cover the supported patterns, unsafe-fusion
guards, retained inputs, provider/dtype eligibility, and explicit opt-in.

Native ORT 1.30 FP32 checks pass with both compact adapters: inactive and
reverted logits match the bare model, and active logits match the unfused
quantized reference exactly. All 196 observed projection updates reuse the
base buffer, while a synthetic graph-output case exercises the safe copy
fallback. Matrix, batch, vector, partial-block, shared-value, Gemm, and
invalid-input cases are covered by native synthetic checks.

WebGPU execution, graph-capture transitions, and performance still require
GPU validation. Generated operator/kernel documentation must be refreshed
with a build containing this schema before merge.

No GPU performance-neutrality guarantee is made.
