# WebGPU LoRA and QLoRA evaluation

This report compares upstream main with the optimized QLoRA implementation
using one evaluation model and two adapters, labeled A and B. The model uses
FP16 activations, INT4 base weights, and rank-32 INT8 adapters. Model names,
prompt contents, local paths, raw logs, and model assets are omitted.

## Main versus optimized

Measurements were collected on 2026-10-14 with native GenAI C++, built-in
WebGPU, Windows D3D12, and an NVIDIA RTX 5080 (driver 610.88).

**The unchanged original optional-adapter graph fails with the adapter off on
main**, at both prompt lengths: empty adapter defaults reach DequantizeLinear,
which rejects dispatch size `(0, 1, 1)`. Main's **Off*** values below therefore
use a separately pruned base-only graph. Optimized Off uses the actual
optional-adapter graph with empty adapters. These are different off baselines.

This failure describes the recorded main revision below. Newer main includes
the empty-rank MatMul/DequantizeLinear fix (#33171), now merged into this branch.
The historical table has not been relabeled as measurements of that newer main.

| Prompt tokens | Adapter | Main TTFT ms | Optimized TTFT ms | TTFT reduction | Main decode tok/s | Optimized decode tok/s | Decode gain |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | Off* | 29.86 | 31.18 | -4.4% | 261.49 | 236.06 | -9.7% |
| 128 | A | 97.44 | 40.46 | 58.5% | 115.73 | 205.66 | 77.7% |
| 128 | B | 92.48 | 40.34 | 56.4% | 115.11 | 205.36 | 78.4% |
| 1024 | Off* | 162.67 | 162.98 | -0.2% | 258.23 | 232.99 | -9.8% |
| 1024 | A | 242.25 | 172.87 | 28.6% | 113.52 | 201.19 | 77.2% |
| 1024 | B | 246.09 | 172.97 | 29.7% | 113.64 | 199.22 | 75.3% |

TTFT includes prompt processing and first-token sampling/readback. Lower TTFT
and higher throughput are better. Negative gains indicate regressions against
the listed reference. Pure prompt-processing throughput is reported separately:

| Prompt tokens | Adapter | Main prefill tok/s | Optimized prefill tok/s | Gain |
|---:|---|---:|---:|---:|
| 128 | Off* | 4296.0 | 4114.5 | -4.2% |
| 128 | A | 1314.8 | 3170.1 | 141.1% |
| 128 | B | 1385.3 | 3179.5 | 129.5% |
| 1024 | Off* | 6299.4 | 6289.8 | -0.2% |
| 1024 | A | 4228.5 | 5926.8 | 40.2% |
| 1024 | B | 4162.7 | 5925.0 | 42.3% |

Within the optimized optional-adapter graph, enabling an adapter adds
29.4-29.8% TTFT at 128 prompt tokens and 6.1% at 1024. Decode latency overhead
is 14.8-14.9% and 15.8-16.9%, respectively, computed as
`off tokens/s / on tokens/s - 1`. The 10% decode-overhead target is not met.

Main's base-only graph receives MatMulNBitsMlpFusion and MatMulNBitsQkvFusion.
The optimized contrib nodes prevent those separate base fusions. Skipping
adapter arithmetic therefore does not give parity with the standalone base
graph: optimized off decode is approximately 10% slower than that reference.

## Measurement method

- Main revision: `9c87bdf2e6676eee0073c711a32e75183dc41199`.
- Optimized source tree: `b3e1633c9567459041474114256d389bc824e569`.
  Publication changed commit metadata only; the tested source tree is identical.
- Both configurations use the same native benchmark executable, GenAI DLL, and
  DXC. The benchmark includes adapter selection and fixed-token inputs. Only
  the ORT runtime and model representation differ.
- Batch size 1, identical fixed prompt IDs, greedy decoding, 128 generated
  tokens, maximum length 4096, graph capture, generator reuse, and token readback.
- Two rounds per configuration, each with three warmups and five measured
  requests. The second round reverses configuration and prompt-length order.
  Tables use the median of all ten individual requests. All 120 measured
  requests completed with the expected token counts and stable token sequences
  within each configuration.
- Measurements run serially after builds finish, with profiling, verbose
  logging, shader dumping, and other evaluation work disabled. Each cell checks
  for competing build/test/benchmark work and waits for three idle-GPU samples.
  Separate diagnostic runs confirm all nodes execute on WebGPU and graph
  capture replays; diagnostic timings are excluded.
- Main prefill varies more: adapter-on TTFT ranges are 91.69-107.07 ms (A) and
  91.12-106.52 ms (B) at 128 tokens; 234.83-252.03 ms (A) and 236.42-250.94 ms
  (B) at 1024. Optimized ranges are 40.06-41.18 ms and 40.01-41.08 ms at 128;
  172.42-173.09 ms and 172.34-173.23 ms at 1024.

The original adapters use signed INT8 weights and FP32 scales. Offline
conversion transposes and packs the same integer weights and rounds scales to
FP16; the base external weights remain byte-identical. DP4A prefill and decode
reduction changes also affect arithmetic. This compares the same source
workload after conversion, with the numerical limits below.

Main includes newer unrelated upstream changes than the optimization branch's
common ancestor. This is an end-to-end configuration comparison, not an
identical-base ablation. Results cover one GPU and model; final performance
measurements cover built-in WebGPU only. Earlier built-in/plugin compatibility
checks do not constitute a plugin rerun of the final optimizations.

## Correctness and tests

The model-level check uses six prompts across both adapters, 72 full-vocabulary
logit rows evaluated at identical forced prefixes, adapter on/off, capture
on/off, and repeated generation after rewind.

| Comparison | Adapter | Identical responses | Matching top-1 rows | Logit RMSE | Maximum absolute difference |
|---|---|---:|---:|---:|---:|
| Final vs previous QLoRA | Off | 6/6 | 72/72 | 0 | 0 |
| Final vs previous QLoRA | On | 6/6 | 71/72 | 0.008171 | 0.0625 |
| Final vs original source model | On | 4/6 | 70/72 | 0.075461 | 1.03125 |
| Capture on vs off | Off | 6/6 | 72/72 | 0 | 0 |
| Capture on vs off | On | 6/6 | 72/72 | 0 | 0 |

The previous QLoRA implementation is `210584c743785f949cfc417a185366fa0343b932`,
before the final decode register-reuse/subgroup-reduction change. Adapter-on
arithmetic is not bitwise equivalent to the original model. Six prompts are
limited quality coverage, and larger quality evaluation remains necessary.

Initial optimization validation passed seven operator tests, eight graph-fusion tests, and
two WGSL smoke tests. The decode regression checks against a dense
double-precision reference at K=2048/4096/8192, with mixed signs, varying INT8
scales, a zero rank, tail columns, and both accumulation options. Quantized
prefill and reordered decode comparisons use absolute tolerance 0.002 and
relative tolerance 0.001; unaffected fusion paths retain zero tolerance.
The input-upload change also has a regression test for chunking and reuse.

See [FP16 LoRA usage](WebGPU_MatMulNBitsLoRA_Prototype.md) and
[QLoRA format, dispatch, and test commands](WebGPU_MatMulNBitsQLoRA_Prototype.md).

## CI integration verification

The CI integration merges main at `a5e205be24` and updates the FP16 original-layout
fallback to the current `MatMulComputeDispatcher` API. Adapter weights remain
nonconstant so that replacement cannot reuse a stale transformed weight. It also
uses explicit `size_t` shape indices for wasm32 and removes plugin-test linkage
to built-in-only capability-inspection symbols. The optimized LoRA/QLoRA shader
code and workgroup sizes are unchanged by these fixes.

### Performance against the pre-fix optimization

The following measurements compare the previously tested optimization with the
merged/fixed build, using the same models, native benchmark, GenAI DLL, and DXC.
This includes upstream main changes, not just the compatibility edit. Hardware
and settings match the measurement method above. All other build, test, and
profiling work was stopped during measurement, with idle checks before each cell.
There were two rounds in reversed order, three warmups and five measured
requests per round: 200 measured requests in total. Values are medians of ten
requests. Every before/after generated token sequence matches.

| Model layout | Prompt tokens | Adapter | Before TTFT ms | Fixed TTFT ms | TTFT change | Before decode tok/s | Fixed decode tok/s | Decode change |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Packed INT8 | 128 | Off | 31.43 | 31.39 | -0.13% | 228.07 | 226.45 | -0.71% |
| Packed INT8 | 128 | A | 40.63 | 40.28 | -0.88% | 197.45 | 198.80 | +0.68% |
| Packed INT8 | 128 | B | 40.36 | 40.54 | +0.44% | 197.33 | 196.49 | -0.43% |
| Packed INT8 | 1024 | Off | 163.28 | 163.32 | +0.03% | 224.57 | 223.77 | -0.36% |
| Packed INT8 | 1024 | A | 173.95 | 173.51 | -0.25% | 196.29 | 196.00 | -0.15% |
| Packed INT8 | 1024 | B | 174.23 | 173.57 | -0.38% | 195.98 | 194.35 | -0.83% |
| FP16 original | 128 | Off | 31.94 | 31.95 | +0.01% | 225.84 | 221.95 | -1.73% |
| FP16 original | 128 | A | 94.74 | 96.33 | +1.68% | 138.70 | 137.97 | -0.53% |
| FP16 original | 1024 | Off | 163.72 | 163.79 | +0.04% | 222.65 | 222.52 | -0.06% |
| FP16 original | 1024 | A | 230.97 | 230.88 | -0.04% | 137.42 | 138.36 | +0.69% |

Positive TTFT change is slower; positive decode change is faster. QLoRA changes
are all within 0.9%. The initial FP16 128-token results warranted an independent,
longer confirmation: four rounds with alternating order, three warmups and eight
measured requests per cell (32 requests per variant/adapter; 128 requests total).
These additional requests use the same settings and also match token-for-token.

| FP16, 128 tokens | Before TTFT ms | Fixed TTFT ms | TTFT change | Before decode tok/s | Fixed decode tok/s | Decode change |
|---|---:|---:|---:|---:|---:|---:|
| Off | 31.82 | 31.82 | -0.00% | 222.76 | 223.70 | +0.42% |
| A | 94.78 | 95.26 | +0.51% | 138.78 | 138.44 | -0.25% |

In this confirmation, adapter-on TTFT ranges overlap: 94.10-100.15 ms before and
93.68-101.22 ms after. Adapter-off decode ranges also overlap: 218.36-228.44 and
219.64-228.64 tok/s. These measurements show no material performance regression
on this GPU, with residual sub-1% median differences in the confirmation and
QLoRA matrix. They do not establish zero change or results on other devices.

### Build and correctness coverage

- Built-in and dynamic-plugin Release builds and test binaries compile and link.
- Built-in: 8 operator/upload tests and 19 graph-fusion tests pass, with no skips.
- Plugin: 5 operator tests and 19 graph-fusion tests pass. Two hardware-specific
  specialization checks explicitly skip because the plugin test binary cannot
  inspect internal device capabilities. Both checks run and pass in built-in
  mode; general plugin operator/fusion tests remain active.
- Two WGSL smoke tests pass. Six changed WebGPU translation units also pass
  wasm32 syntax compilation with Emscripten 4.0.23 and narrowing warnings treated
  as errors. This is not a full linked WebAssembly build.
- Full-model QLoRA checks against the pre-fix runtime are bitwise exact for
  adapter off and on: 6/6 responses and all 72 forced-prefix logit rows per state.
  Built-in/plugin, capture on/off, and generation after rewind also match exactly
  in the 12-case matrix. This verifies the CI fixes preserve the optimized
  runtime's outputs; it does not change the original-model numerical limits above.
- Formatting and whitespace checks pass. Generated contrib documentation now
  includes both new operator schemas. Remote CI must validate other build targets.

## Review-feedback verification (2026-10-15)

Merged main at `9cdf0f11fde12c171eceb4e4ca3b31be8777e7f0`, including the
empty-rank MatMul/DequantizeLinear fix (#33171) and graph-capture buffer-copy
fix (#33362). The LoRA review-feedback patch changes tests and documentation;
it does not change LoRA shaders or workgroup sizes.

The following comparison uses the previously validated PR runtime snapshot
(`7e7079934a`) as Before. Updated uses the merged ORT build and GenAI
`b4023152` (adapter-aware capture identity, based on GenAI main `bdfadcf4`).
These are update-regression measurements, not a new main-versus-fusion study.

Native C++ benchmark, built-in WebGPU / Windows D3D12 / NVIDIA RTX 5080:
batch 1, fixed prompt IDs, 128 generated tokens, maximum length 4096, greedy
decoding, capture, generator reuse, and token readback. Profiling and competing
build/test work were stopped throughout measurement. Each cell has two rounds
of three warmups and five measured requests, with reversed order in round two.
The table reports medians over ten individual requests per runtime/cell.
All 160 measured requests completed and generated identical sequences across
the two runtimes for each case.

| Model layout | Prompt tokens | Adapter | Before TTFT ms | Updated TTFT ms | TTFT change | Before decode tok/s | Updated decode tok/s | Decode change |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Packed INT8 | 128 | Off | 31.48 | 31.43 | -0.17% | 228.69 | 233.12 | +1.94% |
| Packed INT8 | 128 | On | 40.17 | 40.31 | +0.34% | 205.68 | 205.60 | -0.04% |
| Packed INT8 | 1024 | Off | 162.62 | 162.52 | -0.06% | 233.16 | 233.74 | +0.25% |
| Packed INT8 | 1024 | On | 172.67 | 174.30 | +0.95% | 204.11 | 197.54 | -3.22% |
| FP16 original | 128 | Off | 31.95 | 31.90 | -0.17% | 229.53 | 229.59 | +0.03% |
| FP16 original | 128 | On | 97.34 | 96.19 | -1.18% | 140.95 | 141.15 | +0.14% |
| FP16 original | 1024 | Off | 163.49 | 163.97 | +0.29% | 226.22 | 219.26 | -3.07% |
| FP16 original | 1024 | On | 232.54 | 233.87 | +0.57% | 137.72 | 138.31 | +0.43% |

Positive TTFT change means slower; positive decode change means faster.
The two approximately 3% decode slowdowns prompted a longer, independent
confirmation separating the ORT merge from the GenAI update. This used two
rounds with reversed order, six warmups and eight measured requests each:
16 measurements per runtime/case, 96 measured requests total. All sequences
matched across the three runtimes.

| 1024-token case | Runtime | TTFT ms | Decode tok/s | Decode range tok/s |
|---|---|---:|---:|---:|
| FP16, off | Previous | 164.59 | 219.34 | 217.18-221.37 |
| FP16, off | Merged ORT / previous GenAI | 163.76 | 218.64 | 214.75-221.01 |
| FP16, off | Merged ORT / fixed GenAI | 163.91 | 219.66 | 216.30-224.18 |
| Packed INT8, on | Previous | 175.74 | 198.31 | 197.07-199.59 |
| Packed INT8, on | Merged ORT / previous GenAI | 176.12 | 198.22 | 195.54-199.43 |
| Packed INT8, on | Merged ORT / fixed GenAI | 176.00 | 197.80 | 195.55-198.86 |

In this confirmation, final versus previous decode changes are +0.14% for
FP16 adapter-off and -0.25% for packed INT8 adapter-on. TTFT changes are -0.41%
and +0.15%, respectively. The earlier decode slowdowns did not repeat. These
results show no repeatable regression on the tested GPU; they do not establish
zero change or performance on other devices. The initial measurements remain
above rather than being discarded. This does not resolve the pre-existing
approximately 10% adapter-off gap versus a graph with QKV/MLP base fusion.

### Correctness and capture

- Built-in and plugin Release builds pass. Built-in provider/empty-tensor tests:
  23 passed; fusion tests: 20 passed; graph-capture buffer-copy tests: 6 passed.
- Plugin provider tests: 5 passed, with 2 expected hardware-inspection skips;
  fusion tests: 20 passed. Both skipped specializations pass in built-in mode.
- Empty-default serialization/reload tests cover decode and prefill, full and
  partial overrides, and separate on/off capture IDs. Both actual captures are
  asserted; alternating replay overwrites sentinel outputs correctly with CPU
  fallback disabled. Empty packed defaults have no serialized payload.
- Reported RTX 4070 tolerance failures are addressed with narrowly scoped
  tolerances: K=8192 FP16 lane accumulation allows 0.003 absolute error; FP16
  fusion comparisons allow 0.0001 absolute plus two FP16 epsilons relative.
  Shorter K, FP32 accumulation, and deliberately exact switching/cancellation
  checks keep their previous bounds. RTX 4070 was not available locally.
- GenAI previously reused a base capture after late adapter activation. In a
  full-model forced-prefix reproduction, capture-off results matched a fresh
  active generator exactly, while captured decode differed by up to 2.82764.
  With [GenAI #2740](https://github.com/microsoft/onnxruntime-genai/pull/2740),
  all six full-vocabulary logit rows match exactly for built-in capture off/on
  and plugin capture on. The added Python regression passes both WebGPU
  parameterizations; its CUDA variants were not run locally.
- Formatting and whitespace checks pass. Wider device/build validation remains
  the responsibility of fresh CI and hardware testing.
