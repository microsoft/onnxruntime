# Optional LoRA fusion for MatMulNBits

This is an opt-in prototype for quantized models with runtime-selectable
low-rank adapters. It does not change model weights or enable adapter loading.
The feature is disabled by default and requires maintainer review before
being treated as a supported optimization.

## Model contract

The supported graph computes:

```text
base = MatMulNBits(A, B, scales, zero_points?, g_idx?, bias?)
low_rank = MatMul(A, lora_A)
delta = MatMul(low_rank, lora_B)
Y = Add(base, delta)
```

The adapter weights must remain graph inputs with matching empty
initializers: `lora_A[K, 0]` and `lora_B[0, N]`. The rank dimensions of the
graph inputs should be symbolic so active adapters can supply positive-rank
weights. Both adapter matrices must use the activation dtype and the same
rank. Their default tensors must have no external or embedded weight data.

The transformer replaces this pattern with `com.microsoft.MatMulNBitsLora`.
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
    "optimization.enable_matmul_nbits_lora_fusion", "1"
)
```

CPU supports FP32. WebGPU supports FP32 and FP16, subject to the existing
MatMulNBits provider constraints. FP16 CPU graphs are not fused.

## Execution behavior

Rank zero executes only the existing base projection, without allocating
low-rank intermediates or dispatching the adapter MatMuls and final Add.
Positive rank computes the two low-rank MatMuls and adds their separate
delta to the base result. The implementation does not merge the adapter
weights into the quantized weights or cache mutable adapter weights as
constant MatMul operands.

The original adapter input names and model metadata are retained. Vector,
matrix, and batched activations keep their original output shapes.

Fusion requires matching execution providers and single-use intermediate
values that are not graph outputs. Nonempty defaults, unsupported dtypes,
prepacked-weight attributes, shared intermediates, and non-unit or
transposed Gemm variants are left unchanged.

Reduced builds loading an optimized model need the new operator in their
required-operator configuration. The transformer itself is available in
full builds; it is not registered for minimal-build optimization replay.

## Review and validation scope

The provider tests cover inactive and active rank, bias, empty output,
vector inputs, rank validation, and FP16 WebGPU graph-capture replay.
Structural optimizer tests cover the supported patterns, unsafe-fusion
guards, retained inputs, provider/dtype eligibility, and explicit opt-in.

Native FP32 integration checks and FP16 WebGPU reference/fused
base/adapter/base checks have passed on an earlier runtime snapshot.
Those checks are not substitutes for running the new provider and
optimizer tests against current upstream main. Generated operator and
kernel documentation must also be refreshed with a build containing this
schema before the draft is ready to merge.

No GPU performance-neutrality guarantee is made.
