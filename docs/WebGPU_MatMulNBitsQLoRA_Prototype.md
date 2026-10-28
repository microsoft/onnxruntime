# WebGPU MatMulNBitsQLoRA prototype

`com.microsoft.MatMulNBitsQLoRA` fuses a quantized base projection, two quantized
adapter projections, and their residual Add. Adapter weights remain replaceable. This
is an inference operator for quantized adapters; it does not merge adapter
weights into the base or perform adapter training.

Graph matching runs automatically at the extended graph-optimization level
(`ORT_ENABLE_EXTENDED`) or higher when WebGPU provides the fused kernel.
CPU execution and unsupported patterns retain the original graph. For a comparison
with the original graph, disable this transformer using the standard session option.
For native GenAI, merge this entry into `model.decoder.session_options`:

```json
{
  "optimization.disable_specified_optimizers": "MatMulNBitsQLoRAFusion"
}
```

Append the transformer name with a semicolon if other optimizers are already disabled.
This transformer handles packed INT8 adapters.
Both built-in and plugin WebGPU are supported; plugin use requires the host
and provider library from this branch.

The transformer matches packed INT8 MatMulNBits adapter projections, not
arbitrary DequantizeLinear/Cast/MatMul graphs. Those graphs require an offline
conversion to the weight format below. A model exported with explicit
MatMulNBitsQLoRA nodes does not need the transformer enabled, and disabling
the transformer does not undo those explicit nodes.

## Weight format and arithmetic

The [operator definition](ContribOperators.md#com.microsoft.matmulnbitsqlora)
follows MatMulNBits quantization settings and permits any positive adapter rank,
including ranks that are not divisible by 8 or by the quantization block size.
Packed storage pads the final block; the logical rank determines how many
elements participate in the up projection.

The current WebGPU kernel and graph matcher support a subset: FP16/FP32
activations and scales, INT4 base weights, INT8 adapter weights, block size 32
for all projections, active adapter rank 32, positive K divisible by 32,
positive N, and X of rank at least two. Other configurations
are not supported by this kernel. Explicit nodes reject unsupported configurations,
and the matcher leaves them unfused. These are
implementation limits rather than restrictions on the operator definition.

The operator takes `X, W, scales, lora_A, lora_A_scales, lora_B, lora_B_scales`.
For the current WebGPU implementation:

| Tensor | Type | Shape |
|---|---|---|
| X | FP16 or FP32 | [..., M, K] |
| W | Packed INT4 in UINT8 | [N, K/32, 16] |
| scales | Same as X | [N, K/32] or flattened |
| lora_A | Packed INT8 in UINT8 | [32, K/32, 32] |
| lora_A_scales | Same as X | [32, K/32] |
| lora_B | Packed INT8 in UINT8 | [N, 1, 32] |
| lora_B_scales | Same as X | [N, 1] |

Packed adapter bytes encode signed integers with an implicit zero point of
128. For down weight index `(r,k)` and up weight index `(n,r)`, with `F`
denoting the activation type (FP16 or FP32):

```text
At[r,k] = F((F(Aq[r,k]) - 128) * As[r,floor(k/32)])
Bt[n,r] = F((F(Bq[n,r]) - 128) * Bs[n,0])
T       = F(X * At^T)
delta   = F(T * Bt^T)
Y       = F(MatMulNBits(X,W,scales) + delta)
```

This implementation uses INT4 base quantization with block size 32 and implicit zero point 8.
The equations describe the weight interpretation and mathematical projections;
the optimized kernels can change reduction order and activation precision.
Eligible prefill down projections use DP4A activation quantization when the
model's `accuracy_level` permits it. Eligible FP32 DP4A decode also shares the
base projection's quantized X with the down projection. Other decode down
projections and all up projections retain floating-point activations.
The provider's `enableMatmulFp32Accumulation` option
controls floating-point lane accumulation for FP16 inputs, except that the
K-parallel DP4A kernel always accumulates in FP32 when N <= 32. FP32 inputs
always use FP32 floating-point accumulation. Intermediate T uses the activation type;
the specialized decode kernel performs its final subgroup reduction in FP32.
These paths are not claimed to be bitwise equivalent to the unfused graph.

The input layout is already transposed and block packed. The fusion does not
insert a transpose or materialize a complete dequantized adapter.

## Dispatch

Selection is automatic. M is the second-to-last dimension of X; a one-token
prompt follows the decode selection.

| Phase | Selected mode | Implementation for eligible inputs |
|---|---|---|
| Prefill, M > 1 | `fuse_up` | INT8 down projection, then base with INT8 up projection and addition; eligible DP4A paths share quantized X |
| Decode, M = 1 | `fuse_down` | Base and INT8 down projection in one dispatch, then INT8 up projection and addition |

The joint decode kernel supports one total input row and K from 2048 through
8192, when the base projection does not select DP4A. FP32 base projections
can select DP4A even at M=1. Eligible FP32 decode inputs with K from 2048
through 8192 use three dispatches: quantize X once, jointly compute the INT4
base and INT8 down projections, then perform the INT8 up projection and addition.
The joint DP4A shader retains the base's 128-thread workgroups and adds eight
workgroups per input row for rank 32, with four ranks per workgroup. Both
projections read the same quantized activation buffer and scales; each writes
its own output. The next dispatch provides synchronization. The existing DP4A
eligibility checks, including `accuracy_level=4`, still apply.

The joint decode paths use shape, type, and capability checks without a vendor
restriction. The subgroup reduction accepts a fixed width of 32, or explicitly
requests 32 with `SetSubgroupSize(32)` when `SubgroupSizeControl` is supported
and 32 is within the adapter's range. Other devices use the workgroup-memory
down reduction. Existing base MatMulNBits device policies still apply.

The floating-input joint kernel retains 128 threads
per workgroup and adds four workgroups for the
32 down-projection outputs. With fixed or explicitly selected 32-lane subgroups, each lane
loads X into registers and reuses it for two rank outputs, then reduces lane
partials with FP32 subgroup sums. The previous workgroup-memory reduction
remains the fallback. `QLoRAOutputVec4` uses 64 threads and one output per
thread, loading packed weights as `vec4<u32>` and T as `vec4<F>` along rank.

FP32 short prefill (1 < M < 4) also uses standalone DP4A for the down
projection when the base selects DP4A. This preserves the base/up fusion and
reuses its activation quantization. FP16 short prefill and inputs that do not
meet the base DP4A requirements retain the floating-input down projection.
DP4A prefill quantizes X once and shares the quantized tensor and scales
between the down projection and base projection, making three dispatches per
adapted projection. Within the shared DP4A implementation,
`DP4AMatMulNBitsKParallelProgram` handles M below
`min_M_for_tile_optimization` (currently 4), or N <= 32 with K >= 2048.
Indirect expert weights also retain this row-wise path. Small M retains four
output columns per tile; otherwise narrow projections use an N-column tile.
This kernel uses FP32 accumulation for N <= 32. The rank-32 down projection
uses this shared selection without a QLoRA-specific override. Each K-parallel iteration covers 1024
elements; requiring two full iterations for narrow N conservatively amortizes
reduction overhead, while shorter K retains tiled reuse across M. This is an
initial policy, not a universal performance crossover. Existing DP4A
eligibility checks and subgroup-matrix preference still apply. Other DP4A
projections retain the upstream tiled kernel: 64x64 tiles, or 128x32 tiles on
eligible Qualcomm devices, with 256 threads and 32-element K steps. This also
handles rank-32 down projections with K < 2048. The base prefill projection
keeps its fused LoRA-up output stage without changing the matrix tiling.
Shapes outside the joint decode specialization calculate T separately.
Devices that select subgroup matrix arithmetic for the original
INT8 up projection retain that kernel and a separate Add to preserve arithmetic.

## Adapter off and graph matching

The matcher requires two INT8 MatMulNBits nodes with block size 32 and accuracy
level 0, an INT4 base MatMulNBits node, and a final Add. All three projections
must run on WebGPU; the down projection must consume the base's X. Intermediate
outputs must be private. Bias, explicit zero points, group indices, prepacked
weights, and incompatible shapes prevent this fusion.

The matcher preserves initializer shapes and data, including full-size zero
adapter defaults and scale defaults. It does not replace zero weights with empty
tensors: doing so would skip arithmetic and can change NaN/Inf propagation.
Already-empty A rank or B rank selects the ordinary base kernel before adapter
allocation or dispatch. Full-size zero weights still execute the adapter projections.
Input names and the adapter file stay unchanged.

GenAI builds without adapter-aware capture
identity require separate generators when changing the adapter selection.
[GenAI #2740](https://github.com/microsoft/onnxruntime-genai/pull/2740) assigns
separate capture IDs after additive adapter activation. Activation does not
recompute an existing KV cache: rewind to zero and append the prompt again
if the adapter should apply to the earlier tokens. This does not add an
adapter-deactivation API; use a new base generator to return to adapter-off.

Graph replay does not re-evaluate the empty-rank branch. Native callers must
use distinct `gpu_graph_id` run options for adapter-off and adapter-on captures,
with stable GPU bindings and buffer lifetimes for each ID. Captures also depend
on tensor shapes and execution mode; prefill and decode cannot share an ID
when their dispatches differ. The provider does not automatically derive an
ID from adapter state. Release and recapture an ID before changing the execution
path or bindings associated with it. Replacing active adapter values in place
is a separate case: the captured path can remain valid only when its shapes,
layout, and bound buffers stay compatible.

### Compact model packaging

Both an explicitly exported `MatMulNBitsQLoRA` node and the raw three-MatMulNBits
pattern accept empty packed defaults. The raw pattern requires a runtime with
empty-weight MatMulNBits support on CPU and WebGPU; its K/N attributes retain the
logical projection sizes. For an adapter-off default, export overridable
initializers as follows:

| Initializer | Default shape | Default payload |
|---|---|---|
| lora_A | [0, K/32, 32] | Empty |
| lora_B | [N, 0, 32] | Empty |
| lora_A_scales | [32, K/32] | Retain the scale defaults |
| lora_B_scales | [N, 1] | Retain the scale defaults |

Keep these tensors in the graph inputs. Declare the rank dimension of each
packed graph input as dynamic, so a rank-32 adapter can override its empty
initializer. Either empty packed weight disables the adapter. Scale-only and
single-weight overrides therefore continue to produce the base result; supplying
both packed weights uses the supplied scales or their retained defaults.
Scale defaults can also be empty, but activating the adapter then requires
supplying both packed weights and their scales. Fusion preserves whichever
representation the model exports; it does not compact full-size zero defaults.

Use the raw MatMulNBits pattern for a model shared by CPU and WebGPU. WebGPU
fuses eligible patterns automatically; CPU executes the original operators.
Saving a fused graph serializes explicit MatMulNBitsQLoRA nodes and requires
the WebGPU kernel to execute them. Disabling fusion does not expand explicit nodes.

### FP32 packages and composed base fusions

The schema, kernel, and transformer support both FP16 and FP32 activations,
base scales, adapter scales, and outputs. All floating tensors in a node must
have the same type. FP32 packages retain FP32 intermediate projections and
scales; they do not convert the model to FP16. Shader cache keys include the
input/output types, so FP16 and FP32 programs have distinct cache entries.

The packed INT8 weights use the same layout for both types, but an adapter
file with FP16 scales cannot directly override FP32 scale inputs (or vice
versa). Export scales in the package's activation type. Sharing one adapter
file across different activation types still requires scale conversion at
the graph boundary or a separate mixed-scale type contract. Casting only
initializer defaults would not handle later adapter overrides.

Empty adapter weights skip the adapter projections and call a standalone
base MatMulNBits. The current QKV/MLP transformers do not compose with the
QLoRA operator, so this can lose the throughput benefit of those base fusions.
Empty defaults alone do not resolve that difference. A follow-up composed
QKV/MLP+QLoRA implementation should dispatch the existing fused base kernels
when every relevant adapter is empty, and preserve per-projection adapter
semantics when any adapter is active. Merely reordering the transformers does
not provide both behaviors.

## Validation

Run from the native build output directory:

```text
onnxruntime_provider_test.exe --gtest_filter=MatMulNBitsQLoRA.*
onnxruntime_test_all.exe --gtest_filter=GraphTransformationTests.MatMulNBitsQLoRAFusion*
```

The quantized tests check original-versus-fused FP16 and FP32 results, FP32 accumulation,
batched and partial-tile shapes, empty adapters, partial overrides,
nonzero/shared defaults, unsupported shapes and attributes, and
patterns that must remain unfused. Full-model evaluation must separately
compare fusion equivalence against the original INT8 model and quantization
differences against FP16. Performance runs should use no profiling and record
the actual runtime libraries, provider, capture setting, and adapter state.

FP32 tests include dense FP64 references with non-half-representable scales,
both fusion modes, DP4A and floating-point down projections, serialized empty
defaults, partial overrides, and separate adapter-on/off graph captures.

Quantized-prefill and reordered-decode FP16 fusion comparisons use 0.002 absolute
and 0.001 relative tolerances; unaffected FP16 paths retain zero tolerance.
FP32 quantized-prefill comparisons use the same approximation bounds; other
FP32 comparisons use 1e-5 absolute/relative bounds. A dense reference test
separately checks the specialized decode reduction.
That reference test allows 0.003 absolute error for K=8192 with FP16 lane
accumulation; shorter K and FP32 accumulation retain 0.002. Its relative
tolerance stays 0.001. Deliberately exact switching checks remain exact.
Serialization regressions check that full-size zero defaults retain their payload,
empty packed defaults remain empty, and a reloaded explicit operator accepts
full and partial overrides without a load-time QLoRA transform. Session tests
exercise automatic fusion and the standard optimizer-disable option.
They also capture adapter-off and adapter-on under separate IDs and alternate
replay with sentinel-filled outputs, checking that both captures execute and
produce the correct results with CPU fallback disabled.
See [the performance evaluation](WebGPU_MatMulNBitsQLoRA_Performance.md) for model-level
accuracy limits and the main-versus-optimized performance comparison. WebGPU
also batches CPU-to-GPU input uploads to amortize per-tensor adapter transfer
cost; this is independent of the shader fusion.
