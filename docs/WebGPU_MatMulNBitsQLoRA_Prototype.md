# WebGPU MatMulNBitsQLoRA prototype

`com.microsoft.MatMulNBitsQLoRA` fuses an INT4 base projection, two INT8 adapter
projections, and their residual Add. Adapter weights remain replaceable. This
is an inference operator for quantized adapters; it does not merge adapter
weights into the base or perform adapter training.

Enable it with `session.enable_matmul_nbits_lora_fusion=1` in the native GenAI
decoder session options, as described in [the FP16 prototype](WebGPU_MatMulNBitsLoRA_Prototype.md).
The same option selects the appropriate transformer for FP16 or INT8 adapters.
Both built-in and plugin WebGPU are supported; plugin use requires the host
and provider library from this branch. Disable the option for a comparison
with the original graph.

## Weight format and arithmetic

The operator takes `X, W, scales, lora_A, lora_A_scales, lora_B, lora_B_scales`.
For rank 32, positive K divisible by 32, and positive N divisible by 4:

| Tensor | Type | Shape |
|---|---|---|
| X | FP16 | [..., M, K] |
| W | Packed INT4 in UINT8 | [N, K/32, 16] |
| scales | FP16 | [N, K/32] or flattened |
| lora_A | Packed INT8 in UINT8 | [32, K/32, 32] |
| lora_A_scales | FP16 | [32, K/32] |
| lora_B | Packed INT8 in UINT8 | [N, 1, 32] |
| lora_B_scales | FP16 | [N, 1] |

Packed adapter bytes encode signed integers with an implicit zero point of
128. For down weight index `(r,k)` and up weight index `(n,r)`:

```text
At[r,k] = FP16((FP16(Aq[r,k]) - 128) * As[r,floor(k/32)])
Bt[n,r] = FP16((FP16(Bq[n,r]) - 128) * Bs[n,0])
T       = FP16(X * At^T)
delta   = FP16(T * Bt^T)
Y       = FP16(MatMulNBits(X,W,scales) + delta)
```

Base quantization stays INT4 with block size 32 and implicit zero point 8.
The adapter projections retain accuracy level 0 and their original dot-product
grouping, accumulation policy, and FP16 intermediate rounding. The provider's
`enableMatmulFp32Accumulation` option applies to adapter accumulation as it does
to the original INT8 MatMulNBits kernels.

The input layout is already transposed and block packed. The fusion does not
insert a transpose or materialize a complete dequantized adapter.

## Dispatch

`session.webgpu.lora_fusion_mode=auto` is the default. M is the second-to-last
dimension of X; a one-token prompt follows the decode selection.

| Phase | Selected mode | NVIDIA implementation |
|---|---|---|
| Prefill, M > 1 | `fuse_up` | Original INT8 down projection, then base with INT8 up projection and addition |
| Decode, M = 1 | `fuse_down` | Base and INT8 down projection in one dispatch, then INT8 up projection and addition |

The joint decode kernel supports one total input row and K from 2048 through
8192. It retains 128 threads per workgroup and adds four workgroups for the
32 down-projection outputs. Its reduction matches the original generic INT8
kernel. `QLoRAOutputVec4` uses 64 threads and one output per thread, loading
packed weights as `vec4<u32>` and T as `vec4<f16>` along rank.

DP4A prefill retains the base activation quantization dispatch, making three
dispatches per adapted projection. Explicit `fuse_down` and `fuse_up` remain
available. Shapes outside the joint decode specialization calculate T
separately. Devices that select subgroup matrix arithmetic for the original
INT8 up projection retain that kernel and a separate Add to preserve arithmetic.

## Adapter off and graph matching

The matcher requires two INT8 MatMulNBits nodes with block size 32 and accuracy
level 0, an INT4 base MatMulNBits node, and a final Add. All three projections
must run on WebGPU; the down projection must consume the base's X. Intermediate
outputs must be private. Bias, explicit zero points, group indices, prepacked
weights, and incompatible shapes prevent this fusion.

Private overridable defaults containing only byte value 128, with finite
default scales, are replaced by empty packed adapter tensors in the session
graph. Scale defaults remain unchanged, preserving partial override semantics.
Empty A rank or empty B rank selects the ordinary base kernel before adapter
allocation or dispatch. Nonzero and shared packed defaults retain their data.
Input names and the adapter file stay unchanged.

Like the FP16 prototype, eliding a zero branch assumes finite activations,
adapter scales, and adapter intermediates. NaN/Inf propagation through zero multiplication can differ;
the transformer remains opt-in. Use separate GenAI generators when switching
adapters or their on/off state with graph capture, and rewind only while
retaining the same generator's adapter selection.

## Validation

Run from the native build output directory:

```text
onnxruntime_provider_test.exe --gtest_filter=MatMulNBitsQLoRA.*:MatMulNBitsLoRA.*
onnxruntime_test_all.exe --gtest_filter=GraphTransformationTests.MatMulNBits*LoRA*
```

The quantized tests check original-versus-fused results with zero tolerance,
FP32 accumulation, batched and partial-tile shapes, empty adapters, partial
overrides, nonzero/shared defaults, unsupported shapes and attributes, and
patterns that must remain unfused. Full-model evaluation must separately
compare fusion equivalence against the original INT8 model and quantization
differences against FP16. Performance runs should use no profiling and record
the actual runtime libraries, provider, capture setting, and adapter state.
