# WebGPU quantized LoRA evaluation

`MatMulNBitsQLoRA` keeps rank-32 adapters replaceable while combining an INT4
base projection with packed INT8 adapter projections. Both FP16 and FP32
models are supported, with scales and intermediate tensors matching the
model's floating type. See the [operator guide](WebGPU_MatMulNBitsQLoRA_Prototype.md)
for layouts, automatic graph matching, explicit export, and capture requirements.

## QLoRA-only split regression (2026-10-15)

A fresh paired comparison against the saved pre-split runtime (`0243e40267`)
uses the benchmark protocol below. Only the ORT DLL changes; GenAI, DXC, models
and adapters are identical. Builds, tests and profiling finished before timing.

| Type | Prompt | Adapter | TTFT before / after (ms) | Change | Decode before / after (tok/s) | Change |
|---|---:|---|---:|---:|---:|---:|
| FP16 | 128 | off | 31.30 / 31.32 | +0.04% | 224.34 / 227.71 | +1.50% |
| FP16 | 128 | on | 40.48 / 40.20 | -0.70% | 205.08 / 206.54 | +0.71% |
| FP16 | 1024 | off | 163.33 / 163.04 | -0.18% | 222.47 / 225.62 | +1.42% |
| FP16 | 1024 | on | 173.43 / 173.15 | -0.16% | 194.33 / 193.15 | -0.61% |
| FP32 | 128 | off | 29.83 / 29.73 | -0.33% | 195.47 / 192.94 | -1.29% |
| FP32 | 128 | on | 43.41 / 43.51 | +0.25% | 180.45 / 179.47 | -0.54% |
| FP32 | 1024 | off | 162.45 / 162.55 | +0.06% | 192.53 / 192.49 | -0.02% |
| FP32 | 1024 | on | 195.10 / 192.97 | -1.09% | 166.45 / 169.01 | +1.54% |

No repeatable material regression was observed: TTFT changed by -1.09% to
+0.25%, and decode throughput by -1.29% to +1.54% in these cells. Adapter-on
checks at 512/2048 tokens remained within 1.42%. A noisy 32-token first pass
showed a 2.48% FP16 decode loss and an 8.33% FP32 gain; a reverse-order repeat
measured -0.62%/-1.35% decode changes and -0.19%/+1.55% TTFT changes for
FP16/FP32. The experiment supports preserving performance, not a claimed speedup.
Run-to-run absolute rates varied, so compare paired runtimes rather than
historical absolute timings.

All 38 native QLoRA/fusion/base regression tests passed with existing tolerances.
All 32 full-model before/after comparisons had exact logits and greedy tokens,
covering both types, two adapters, adapter on/off, and capture on/off. Native
build, Emscripten syntax checks and lint passed. These exact comparisons validate
the split; they do not remove the original-versus-optimized accuracy caveats
below. Original model/adapter hashes are unchanged.

## Performance reference

Native GenAI C++, built-in WebGPU, Windows D3D12, RTX 5080. Batch 1, fixed
prompt IDs, 128 generated tokens, capacity 4096, accuracy level 4, graph
capture, generator reuse, and token readback. Three warmups and five measured
requests per cell; medians. Timings were isolated from builds, tests, profiling,
and competing GPU evaluation. These are saved measurements of the QLoRA
implementation before floating-adapter support was separated into a follow-up.

| Model type | Prompt tokens | TTFT off / on (ms) | Decode off / on (tokens/s) |
|---|---:|---:|---:|
| FP16 | 128 | 31.30 / 40.28 | 235.74 / 204.78 |
| FP16 | 1024 | 163.10 / 172.80 | 232.17 / 204.25 |
| FP32 | 128 | 29.62 / 43.47 | 202.64 / 180.06 |
| FP32 | 1024 | 162.50 / 192.40 | 198.13 / 176.73 |

TTFT includes prompt processing and first-token sampling/readback. Adapter-off
uses the same optional graph's empty-adapter bypass. The measured TTFT overhead
is 28.7%/5.9% for FP16 and 46.7%/18.4% for FP32 at 128/1024 tokens. Decode
throughput loss is 10.8-13.1%; per-token latency overhead is 12.1-15.1%.
These are local device measurements, not a guarantee for other GPUs.

An earlier paired comparison against upstream main measured FP16 adapter-on
TTFT reductions of 56-59% at 128 tokens and 29-30% at 1024 tokens, with decode
throughput increasing 75-78%. Those runs used two adapters and ten samples per
configuration on earlier runtime revisions. The original adapter-off graph
failed on that main revision because empty DequantizeLinear was unsupported;
its published base reference was a separately pruned graph. Later main fixes
addressed that empty-tensor issue. The optimized optional graph's adapter-off
decode remained about 10% below the separate base graph with QKV/MLP fusion.
Skipping adapter arithmetic does not recover those composed base fusions.

## Retained execution paths

- Prefill (M > 1) selects `fuse_up`: compute T, then combine the base projection
  with the INT8 up projection and Add. Eligible DP4A projections share one
  quantized X allocation. The narrow down projection reuses the shared
  K-parallel DP4A kernel with four output columns for M < 4 and 32 otherwise.
  FP32 short prefill uses this path when the base also selects DP4A, preserving
  activation reuse without changing fusion. Within eligible DP4A paths,
  this kernel handles M < 4 or (N <= 32 and K >= 2048), as well as indirect
  expert weights. The narrow-N threshold requires two full K-parallel
  iterations to amortize reduction overhead; shorter K retains tiled reuse
  across M.
- Eligible FP16 decode selects `fuse_down`: one dispatch computes the base
  and down projections in independent workgroups, followed by up/Add. Packed
  adapter weights are dequantized as needed; X stays floating point.
- Eligible FP32 decode uses quantize X, joint DP4A base/down, then up/Add.
  Quantized X and its scales are reused by the two projections. This increased
  decode throughput by approximately 34% against the preceding four-dispatch
  FP32 QLoRA implementation in the paired local experiment.
- Empty adapters return through the ordinary base implementation before
  adapter allocation or dispatch. Unsupported specializations retain existing
  kernels, including subgroup-matrix up projections where selected.

Workgroup sizes, reduction order, quantization, and dispatch selection are
preserved when removing floating-adapter-only branches. The deferred row-tile
experiment is not included.

## Numerical interpretation and coverage

Accuracy level 4 permits DP4A activation quantization. Quantization and changed
reduction order can alter logits and generated tokens relative to the original
unfused package; these performance results do not assert bitwise equivalence.
In the FP32 support evaluation, adapter-off logits were exact. Adapter-on
comparison with the original package reached a maximum logit difference of
1.3800392, 71/72 forced-prefix next-token matches, and 3/6 identical generated
sequences. With activation quantization disabled in both configurations, the
maximum difference was 0.0000234097 and all tested choices and sequences matched.

The subsequent joint DP4A decode change matched all six generated sequences
and all 72 forced-prefix choices against its preceding QLoRA implementation;
the maximum decode-logit difference was 0.822086. This is a small numerical
regression set, not a broad model-quality evaluation. Adapter-off and prefill
logits were exact across that decode change.

Provider and fusion coverage includes FP16/FP32, dense FP64 references, K tails,
batching, prefill/decode, empty and partial adapters, compact serialization,
unsupported patterns, and separate adapter-on/off capture IDs. Full-model
capture OFF/ON and generator rewind checks use the GenAI adapter capture-identity
fix from [GenAI #2740](https://github.com/microsoft/onnxruntime-genai/pull/2740).
