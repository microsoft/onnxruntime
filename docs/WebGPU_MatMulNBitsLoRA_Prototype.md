# WebGPU MatMulNBitsLoRA prototype

This opt-in prototype replaces `MatMulNBits(X, W) + MatMul(MatMul(X, A), B)` with
one `com.microsoft.MatMulNBitsLoRA` node during session optimization. It retains
the packed base weights and the replaceable FP16 adapter inputs. It does not
merge or requantize the adapter into the base weights, or change the source
model file.

The current matcher supports FP16 rank-32 adapters, INT4 base weights with block
size 32, an implicit zero point, no base bias, and no separate adapter scaling
node. The down projection must use the same activation as the base projection,
and intermediate outputs must have no other consumers or graph outputs.

## Enable in native GenAI

Add this entry to the decoder's `session_options`, or supply it in the native
benchmark's configuration overlay:

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "session.enable_matmul_nbits_lora_fusion": "1"
      }
    }
  }
}
```

Use the original model directory and original `.onnx_adapter` file. The
transformer is registered only when the selected WebGPU provider advertises
the new kernel. Both the ORT host and the WebGPU plugin must come from this
branch for dynamic plugin use. Setting the option to `0` restores the original
graph for a matched comparison.

## Execution

With an adapter, the kernel computes `T = X * A`, then incorporates `T * B`
and the addition into the base kernel's output computation. The small
intermediate `T` remains FP16. The base result and adapter result are rounded
separately before addition. Decode uses two GPU dispatches per projection.
DP4A prefill retains the existing activation quantization dispatch, so it uses
three. Subgroup matrix base kernels use a separate output epilogue fallback.

For offline-transposed adapters, both fusion modes consume the same `At[32,K]`
and `Bt[N,32]` tensors directly. The optimizer absorbs the adapter transpose
wrappers; neither mode dispatches an adapter transpose. Select the mode with
`session.webgpu.lora_fusion_mode` in the same decoder session options:

| Mode | First dispatch | Second dispatch |
|---|---|---|
| `fuse_down` | Base + `T = X * At^T` | `Y = base + T * Bt^T` |
| `fuse_up` | `T = X * At^T` | Base + `Y = base + T * Bt^T` |

The default setting, `auto`, selects `fuse_up` for prefill (`M > 1`) and
`fuse_down` for decode (`M == 1`) on each invocation. Here `M` is the
second-to-last dimension of X, so batched one-token decode also selects
`fuse_down`. A one-token prompt uses the same path as decode. This setting
chooses between the two existing implementations; it adds no shader or GPU
dispatch. Explicit `fuse_down` and `fuse_up` settings force that path for all
input sizes and remain available for comparisons.

On NVIDIA decode (one input row, K from 2048 through 8192), `fuse_down` uses
`MatMulNBitsLoRADown` with eight additional workgroups that calculate T once,
followed by `LoRAOutputVec4`. `fuse_up` uses `LoRADownGemv`, followed by
`MatMulNBitsLoRAUp`. Both use the same FP32 GEMV reduction and FP16 T, with
128 threads per producer workgroup. The standalone output shader retains
64 threads and one output per thread. Both output implementations load Bt and
T as vec4 along rank, preserving scalar rank accumulation order and separate
FP16 rounding before the base/adapter addition.

DP4A prefill retains its activation quantization dispatch. `fuse_down` adds
the tiled down projection to the base dispatch; `fuse_up` runs that same down
projection separately and integrates the vectorized up projection into DP4A.
Other shapes use the existing base kernels, with a separate output fallback
when the selected kernel has no integrated up projection. The mode selection
applies to transposed adapters; original-layout models retain their compatible
down-then-base/up implementation. Empty adapters execute only the base kernel
in either mode. Unsupported mode strings produce an error at initialization.

For an overridable pair of private, all-zero default initializers, the
transformer replaces the defaults in the session graph with empty rank
dimensions and relaxes those graph-input dimensions. Input names, element
types, and non-rank dimensions remain intact. A zero rank in either adapter
input selects the ordinary base kernel directly. The zero check happens once
during optimization; no adapter data is read back from the GPU during inference.
Nonzero or shared defaults are preserved and execute the adapter calculation.

Eliding zero default branches assumes finite activations. IEEE NaN/Inf
propagation through multiplication by zero can differ when that branch is
removed, so this optimization is disabled by default.

Adapters can be replaced between requests on the same model. Captured execution
must use a distinct capture ID when adapter bindings or the active/off state
change, as with ordinary WebGPU graph capture. GenAI generators own separate
capture IDs; create a new generator when selecting another adapter and use
rewind only while retaining that generator's adapter selection.

## Tests

Build the native ORT host and provider test targets, then run from the binary
directory:

```text
onnxruntime_provider_test.exe --gtest_filter=MatMulNBitsLoRA.*
onnxruntime_test_all.exe --gtest_filter=*MatMulNBitsLoRA*
```

The tests cover empty and partially supplied adapters, numerical equivalence
with the original subgraph, adapter replacement, overridable input retention,
invalid shapes, and patterns that must not fuse. Native GenAI evaluation must
also check capture on/off, generator reuse, and the complete model before
interpreting performance results. In a correctness harness, use
`ConfigSetProviderOption` to change `enableGraphCapture`; merely appending a
provider option through a JSON overlay can retain the model's earlier setting.
