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

Final validation passed seven operator tests, eight graph-fusion tests, and
two WGSL smoke tests. The decode regression checks against a dense
double-precision reference at K=2048/4096/8192, with mixed signs, varying INT8
scales, a zero rank, tail columns, and both accumulation options. Quantized
prefill and reordered decode comparisons use absolute tolerance 0.002 and
relative tolerance 0.001; unaffected fusion paths retain zero tolerance.
The input-upload change also has a regression test for chunking and reuse.

See [FP16 LoRA usage](WebGPU_MatMulNBitsLoRA_Prototype.md) and
[QLoRA format, dispatch, and test commands](WebGPU_MatMulNBitsQLoRA_Prototype.md).
