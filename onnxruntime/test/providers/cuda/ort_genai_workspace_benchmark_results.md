# ORT GenAI CUDA Workspace Preallocation Benchmark Results

## Scope

This document records results produced by
`ort_genai_workspace_benchmark.py` through the ONNX Runtime GenAI generation
API. It compares dynamic CUDA scratch allocation with static workspace
preallocation for `MatMulNBits` and `GroupQueryAttention` (GQA).

Results from the earlier C++ provider benchmark are maintained separately in
[`workspace_preallocation_benchmark_results.md`](workspace_preallocation_benchmark_results.md).
Numbers from the two benchmark implementations should not be compared as
equivalent measurements: the Python benchmark measures the ORT GenAI request
lifecycle and device-wide `nvidia-smi` memory, while the C++ benchmark exposes
process-local WDDM and ORT allocator statistics.

The measurements below used:

| Component | Version |
|---|---|
| ONNX Runtime | `f8531b4008845f61899f4619773e43cc675de81b` |
| ONNX Runtime GenAI | `0.16.0-dev`, commit `d5b40851ba80ffa8e95b6b01f921dbb9008fac80` |
| CUDA provider build | Release, native `sm_120a`, matched shared ORT and provider DLLs |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, SM120 |
| CUDA toolkit | 13.3 |
| NVIDIA driver | 610.62 |

## Benchmark design

The script creates a temporary `genai_config.json` for each mode and never
modifies the model package. The controller launches every mode in a fresh
worker process and rotates the six possible mode orders:

- `scratch`: static workspace preallocation is disabled.
- `matmul`: static preallocation is enabled without the non-windowed GQA bound.
- `combined`: static preallocation is enabled for both `MatMulNBits` and GQA.

The tested workload uses batch 1, a 1,024-token prompt, and up to 128 generated
tokens. The GQA upper bound and KV-cache capacity are 1,152 tokens. Add
`--fpa-intb` to set `ep.cuda.fpa_intb_gemm=1`; the default
`--no-fpa-intb` setting uses the legacy `MatMulNBits` path.

The report includes:

- Request and model TTFT, `append_tokens` prefill time, and first-token sampling.
- Full request and model scenario latency.
- Per-token decode latency and TPOT.
- Trimmed means, percentiles, and stall rates.
- Device-wide peak VRAM sampled through `nvidia-smi memory.used`.
- Per-worker output hashes and cross-mode hash-set validation.

Each worker fails if identical measured iterations produce different output
hashes. The controller also fails if the observed hash sets differ between
workspace modes; process-level variation common to every mode remains visible
in the report.

### Statistical methodology

Each dispatch and workspace configuration runs in a separate fresh process so
arena state, memory patterns, and tactic profiling from another mode cannot
affect it. The controller uses six fresh-process comparisons arranged in three
order-balanced blocks.

An absolute latency is the median of the six process-level 10% trimmed means.
Paired percentage changes are computed between adjacent processes within each
block and reported as median `[minimum, maximum]`.

A stall is a sample greater than three times the median for that process.
Scenario trimming removes the lowest and highest 10% of complete requests.
Decode trimming operates across every measured decode-token sample.

### Commands

Warm TTFT uses exactly one generated token:

```powershell
.\.venv\Scripts\python.exe `
  onnxruntime\test\providers\cuda\ort_genai_workspace_benchmark.py `
  --model-path C:\path\to\qwen-model `
  --phase ttft --prompt-tokens 1024 `
  --warmups 3 --iterations 30 --repetitions 6 `
  --output qwen-ttft.json
```

The full generation scenario uses 128 generated tokens:

```powershell
.\.venv\Scripts\python.exe `
  onnxruntime\test\providers\cuda\ort_genai_workspace_benchmark.py `
  --model-path C:\path\to\qwen-model `
  --phase scenario --prompt-tokens 1024 --generated-tokens 128 `
  --warmups 3 --iterations 10 --repetitions 6 `
  --output qwen-scenario.json
```

Run peak-memory measurement separately so sampling does not perturb the
latency processes:

```powershell
.\.venv\Scripts\python.exe `
  onnxruntime\test\providers\cuda\ort_genai_workspace_benchmark.py `
  --model-path C:\path\to\qwen-model `
  --phase memory --prompt-tokens 1024 --generated-tokens 128 `
  --warmups 2 --repetitions 6 --output qwen-memory.json
```

The memory phase samples device-wide rather than process-attributed usage. The
GPU must otherwise be idle. The C++ benchmark should be used when exact
process-local WDDM and ORT arena statistics are required.

## Runtime preallocated-workspace verification

`ORT_CUDA_TRACE_PREALLOCATED_WORKSPACE=1` enables an execution-only diagnostic
for `MatMulNBits` and GQA. `MatMulNBits` reports the selected workspace path,
node, requested bytes, and whether slot 0 supplied the pointer. GQA reports the
slot-0 root source and the number and bytes of subregions served by that root or
by individual scratch fallbacks.

The diagnostic used the Qwen 2.5 1.5B and 7B combined modes with one complete
warmup and one measured 1,024-prefill plus 128-generated-token scenario:

```powershell
$env:ORT_CUDA_TRACE_PREALLOCATED_WORKSPACE = "1"
.\.venv\Scripts\python.exe `
  onnxruntime\test\providers\cuda\ort_genai_workspace_benchmark.py `
  --worker --model-path C:\path\to\qwen-model `
  --mode combined --phase scenario `
  --prompt-tokens 1024 --generated-tokens 128 `
  --warmups 1 --iterations 1 `
  --worker-output workspace-consumption.json `
  --fpa-intb
```

The warmup recorded the memory pattern and therefore reported
`scratch_fallback`. The measured scenario then reported:

| Model and dispatch | MatMulNBits measured workspace requests | Unique MatMulNBits nodes using slot 0 | GenAI GQA root | GQA prefill calls using slot 0 | GQA decode calls using slot 0 | GQA subregion fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 1.5B, legacy | 141 legacy requests | 141 | 123,273,472 B | 28 | 3,556 | 0 |
| Qwen 1.5B, fpA-intB | 240 fpA-intB CUTLASS requests plus 28 legacy requests | 113 fpA-intB plus 28 legacy | 123,273,472 B | 28 | 3,556 | 0 |
| Qwen 7B, legacy | 141 legacy requests | 141 | 287,441,152 B | 28 | 3,556 | 0 |
| Qwen 7B, fpA-intB | 240 fpA-intB CUTLASS requests plus 28 legacy requests | 113 fpA-intB plus 28 legacy | 287,441,152 B | 28 | 3,556 | 0 |

The 3,556 decode calls are 28 layers times 127 decode model evaluations; the
first generated token is sampled from the prefill logits. Every GQA node used
the preallocated root for prefill and decode. The selected route consumed
3,195,136 bytes during Qwen 1.5B prefill and 56,064 bytes during each decode
call. Qwen 7B consumed 7,454,976 bytes during prefill and 130,304 bytes during
each decode call. Neither model had a per-region scratch fallback.

For fpA-intB, the trace distinguishes the 113 nodes that selected
workspace-using CUTLASS tactics from zero-workspace GEMV executions; the
remaining 28 QKV projections used the legacy path. This confirms that the
planned modes consume the cached slot-0 buffers rather than merely registering
workspace plans.

## Legacy `MatMulNBits` results

`ep.cuda.fpa_intb_gemm` was disabled for these runs. Warm TTFT used three
complete warmups and 30 measured one-token requests per process.

| Model | Scratch request TTFT | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 61.535 ms | 61.526 ms | 61.766 ms | -0.239% `[-0.525%, +0.464%]` | +0.074% `[-0.459%, +2.033%]` | +0.351% `[-0.055%, +1.652%]` |
| Qwen 2.5 7B | 275.790 ms | 276.062 ms | 276.140 ms | -0.045% `[-1.734%, +0.525%]` | +0.024% `[-2.969%, +1.231%]` | +0.080% `[-1.257%, +0.703%]` |

The model-side TTFT medians, measured from the start of `append_tokens`, were
60.481/60.553/60.602 ms for Qwen 1.5B and 274.452/274.778/274.786 ms for Qwen
7B. The corresponding `append_tokens` prefill medians were
59.962/60.047/60.099 ms and 273.891/274.197/274.173 ms. Static planning
therefore produced no measurable warm TTFT change.

The full scenario used three warmups and ten measured 1,024-prefill plus
128-generated-token requests per process:

| Model | Scratch scenario | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 854.730 ms | 871.920 ms | 858.416 ms | +1.538% `[-3.492%, +13.209%]` | +1.140% `[-1.540%, +11.153%]` | -0.563% `[-3.273%, +2.023%]` |
| Qwen 2.5 7B | 1,584.195 ms | 1,583.405 ms | 1,589.409 ms | +0.111% `[-1.612%, +0.958%]` | +0.149% `[-1.198%, +1.265%]` | +0.379% `[-1.935%, +1.382%]` |

| Model | Scratch TPOT | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 5.979 ms | 6.024 ms | 5.961 ms | +0.352% `[-3.425%, +13.468%]` | +0.389% `[-2.308%, +9.753%]` | -0.303% `[-3.981%, +1.157%]` |
| Qwen 2.5 7B | 10.283 ms | 10.260 ms | 10.320 ms | -0.141% `[-2.177%, +1.112%]` | -0.134% `[-1.297%, +1.141%]` | +0.608% `[-2.125%, +1.503%]` |

The 7B paired medians are all within 0.7%, with ranges spanning zero. The 1.5B
medians are also small, but its process-level range is wider and includes
occasional decode stalls. These runs do not show a repeatable TTFT, scenario,
or TPOT improvement or regression from static workspace planning. All 18
latency workers per phase produced one identical output hash for their model.

Peak VRAM used six order-balanced fresh-process blocks per model. Each worker
performed two warmups and one measured scenario. Values are median `[minimum,
maximum]` device-wide `nvidia-smi memory.used` readings:

| Model | Scratch peak | MatMulNBits planned peak | Combined planned peak | MatMul minus scratch | Combined minus scratch | Combined minus MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 4,197 MiB `[4,008, 4,262]` | 4,008 MiB `[4,004, 4,260]` | 4,006 MiB `[4,004, 4,264]` | -93 MiB `[-256, 0]` | -65 MiB `[-256, +196]` | 0 MiB `[-256, +256]` |
| Qwen 2.5 7B | 8,169 MiB `[7,652, 8,196]` | 8,172 MiB `[7,652, 9,220]` | 8,168 MiB `[8,164, 9,198]` | +18 MiB `[-544, +1,048]` | -1 MiB `[-24, +1,024]` | +127 MiB `[-1,056, +1,018]` |

Every measured scenario remained at its post-warmup device residency, so the
sampled scenario delta was 0 MiB in every worker. The broad, quantized
fresh-process ranges are larger than the paired mode differences, especially
for 7B. Device-wide `nvidia-smi` establishes an approximate steady-state
envelope but cannot attribute a reliable VRAM delta to workspace planning on
this WDDM system.

Qwen 1.5B memory workers produced one output hash. Qwen 7B memory workers
produced the same two process-level hashes in every mode, with an identical
1,152-token output length; no hash was specific to a workspace mode.

## fpA-intB results

The same six-block matrices were repeated with `--fpa-intb`. All other
workload, warmup, iteration, process-order, build, and reporting settings were
unchanged.

Warm TTFT remained neutral across workspace modes:

| Model | Scratch request TTFT | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 58.512 ms | 58.540 ms | 58.174 ms | +0.392% `[-4.082%, +2.343%]` | -0.072% `[-3.399%, +2.956%]` | -0.047% `[-1.922%, +2.186%]` |
| Qwen 2.5 7B | 240.198 ms | 241.984 ms | 240.006 ms | +0.223% `[-3.120%, +2.040%]` | +0.072% `[-2.609%, +2.777%]` | -0.735% `[-1.342%, +3.450%]` |

Compared with the separately measured legacy-path medians, fpA-intB reduced
request TTFT by approximately 5% for Qwen 1.5B and 12-13% for Qwen 7B. This is
an unpaired dispatch comparison, not a workspace-planning effect.

The scenario and TPOT measurements were:

| Model | Scratch scenario | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 1,477.123 ms | 1,359.249 ms | 1,660.078 ms | +8.440% `[-35.608%, +46.177%]` | +3.852% `[-24.635%, +69.816%]` | +4.442% `[-26.995%, +59.833%]` |
| Qwen 2.5 7B | 1,546.984 ms | 1,588.049 ms | 1,554.085 ms | +3.373% `[+0.592%, +15.858%]` | +0.851% `[-6.223%, +72.088%]` | -1.066% `[-11.612%, +48.534%]` |

| Model | Scratch TPOT | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 10.589 ms | 9.790 ms | 11.903 ms | +7.699% `[-34.800%, +45.403%]` | +2.639% `[-26.170%, +73.986%]` | +6.461% `[-24.177%, +55.091%]` |
| Qwen 2.5 7B | 10.333 ms | 10.528 ms | 10.367 ms | +2.819% `[+0.839%, +16.607%]` | +0.770% `[-4.872%, +69.799%]` | -1.171% `[-11.703%, +45.617%]` |

The Qwen 1.5B process medians varied from 956.626 to 1,873.451 ms, and its
decode stall counts reached 104 of 1,270 measured decode tokens in one worker.
Qwen 7B also had isolated slow workers, including a combined worker with
2,784.882 ms scenario latency and 86 decode stalls. These large fresh-process
differences dominate the small paired medians, so the scenario results do not
establish a planning speedup or regression.

Every Qwen 1.5B worker produced one identical output hash. The balanced Qwen 7B
scenario produced three process-level hashes: scratch observed two, while the
planned modes observed all three, so the report failed its strict same-sample
hash-set check. A separate scratch-only diagnostic reproduced the third hash
after four fresh workers. Each individual worker was internally stable across
its ten measured iterations, all outputs had length 1,152, and no hash was
specific to workspace planning. The variation therefore tracks fresh-process
fpA-intB tactic/numerical behavior rather than a planning-only correctness
change.

Peak VRAM was:

| Model | Scratch peak | MatMulNBits planned peak | Combined planned peak | MatMul minus scratch | Combined minus scratch | Combined minus MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 4,942 MiB `[4,942, 4,944]` | 4,961 MiB `[4,942, 5,074]` | 4,944 MiB `[4,942, 5,198]` | +18 MiB `[-2, +132]` | +2 MiB `[-2, +256]` | -2 MiB `[-126, +254]` |
| Qwen 2.5 7B | 11,093 MiB `[11,074, 11,586]` | 11,460 MiB `[11,076, 12,388]` | 12,108 MiB `[11,334, 13,124]` | +246 MiB `[-252, +1,058]` | +1,023 MiB `[+240, +1,538]` | +617 MiB `[-290, +1,790]` |

The 1.5B ranges overlap and do not establish a memory difference. Every 7B
combined block was 240-1,538 MiB above its paired scratch process, although the
large tactic-dependent range prevents precise attribution. These are
device-wide WDDM readings after warmup, not process-local allocator
measurements.

## Dispatch-path comparison

An unpaired comparison of fpA-intB and legacy medians summarizes why the
dispatch paths must be reported separately:

| Model and metric | Scratch | MatMulNBits planned | Combined planned |
|---|---:|---:|---:|
| Qwen 2.5 1.5B TTFT | -4.91% | -4.85% | -5.81% |
| Qwen 2.5 1.5B scenario | +72.82% | +55.89% | +93.39% |
| Qwen 2.5 1.5B TPOT | +77.10% | +62.50% | +99.69% |
| Qwen 2.5 7B TTFT | -12.91% | -12.34% | -13.09% |
| Qwen 2.5 7B scenario | -2.35% | +0.29% | -2.22% |
| Qwen 2.5 7B TPOT | +0.49% | +2.62% | +0.46% |

Negative values favor fpA-intB. Qwen 1.5B fpA-intB improved prefill-dominated
TTFT but selected substantially slower and less stable decode tactics in these
processes. Qwen 7B improved TTFT while scenario and TPOT remained close to the
legacy medians except for isolated tactic/stall outliers.

## Conclusions

- Static workspace planning was neutral for warm TTFT in both dispatch paths
  and model sizes.
- Scenario and TPOT differences were smaller than the observed
  fresh-process/tactic variation and do not establish a planning speedup or
  regression.
- The runtime trace proved that every planned `MatMulNBits` and GQA node
  consumed slot-0 preallocated memory after warmup, with no GQA subregion
  fallback.
- Device-wide `nvidia-smi` memory is too quantized and process-variable for
  precise workspace attribution on this WDDM system. Use the separate C++
  benchmark report for process-local WDDM and ORT arena measurements.
