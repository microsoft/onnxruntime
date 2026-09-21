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
| ONNX Runtime benchmark commit | `dddccbcf04c0eded17fca96d13b0b14e8fbf0ac4` |
| Rebased ONNX Runtime base | `0e18025de7` (`origin/main`) |
| Bounded non-windowed GQA estimator | `a08a73ee44` (rebased `chilo/bounded-non-windowed-gqa-workspace-estimation`) |
| GQA runtime workspace consumer | `9ed038aa39` |
| ONNX Runtime GenAI | `0.16.0-dev`, commit `d5b40851ba80ffa8e95b6b01f921dbb9008fac80` |
| CUDA provider build | Release, native `sm_120a`, matched shared ORT and provider DLLs |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, SM120 |
| CUDA toolkit | 13.3 |
| NVIDIA driver | 610.62 |

## Results at a glance

These long-context results use one fresh process per dispatch and workspace
mode. Each latency process performs one warmup followed by three measured
requests. Peak VRAM comes from a separate fresh process that performs one
warmup and one measured request while polling device-wide
`nvidia-smi memory.used`. The peak values are therefore approximate WDDM
device totals, not process-attributed allocator measurements.

Terminology:

- **Legacy** is the existing `MatMulNBits` path that dequantizes the integer
  weights into a temporary floating-point buffer and then uses cuBLAS.
- **fpA-intB** uses floating-point activations and integer weights directly.
  Runtime profiling selects between eligible fpA-intB GEMV and CUTLASS GEMM
  tactics; only the CUTLASS tactic requires workspace.
- **Prompt** is the number of input tokens: 4K = 4,096, 8K = 8,192, and
  12K = 12,288.
- **TTFT** is median end-to-end request time to the first generated token,
  including generator setup, prompt prefill, and first-token sampling.
- **Scenario** is median end-to-end request time for the listed prompt and all
  128 generated tokens.
- **TPOT** is the trimmed mean time for each of the 127 decode model
  evaluations after the first token. It excludes prompt prefill and is not
  calculated by dividing scenario time by 128.
- **Peak VRAM** is the maximum device-wide memory usage sampled during the
  separate memory run.
- **S (scratch)** disables static workspace preallocation for both operators.
- **M (MatMulNBits planned)** enables static workspace preallocation for
  `MatMulNBits` only; GQA continues to use scratch allocations.
- **C (combined planned)** enables static workspace preallocation for both
  `MatMulNBits` and GQA. The bound scales with prompt capacity, so this mode
  can reserve substantially more VRAM at long context.

### Qwen 2.5 1.5B — legacy

| Prompt | Mode | TTFT | Scenario | TPOT | Peak VRAM |
|---:|:---:|---:|---:|---:|---:|
| 4K | S | 0.252 s | 1.101 s | 6.642 ms | 8,676 MiB |
| 4K | M | 0.265 s | 1.173 s | 7.039 ms | 8,936 MiB |
| 4K | C | 0.256 s | 1.112 s | 6.437 ms | 10,854 MiB |
| 8K | S | 0.541 s | 1.580 s | 8.102 ms | 14,952 MiB |
| 8K | M | 0.538 s | 1.517 s | 7.747 ms | 15,220 MiB |
| 8K | C | 0.539 s | 1.555 s | 7.859 ms | 23,140 MiB |
| 12K | S | 0.876 s | 2.028 s | 9.075 ms | 16,228 MiB |
| 12K | M | 0.851 s | 2.008 s | 9.018 ms | 16,228 MiB |
| 12K | C | 0.851 s | 2.031 s | 9.250 ms | 24,036 MiB |

### Qwen 2.5 1.5B — fpA-intB

| Prompt | Mode | TTFT | Scenario | TPOT | Peak VRAM |
|---:|:---:|---:|---:|---:|---:|
| 4K | S | 0.232 s | 1.340 s | 10.471 ms | 9,614 MiB |
| 4K | M | 0.242 s | 2.755 s | 17.829 ms | 9,618 MiB |
| 4K | C | 0.240 s | 1.428 s | 10.918 ms | 11,662 MiB |
| 8K | S | 0.535 s | 1.596 s | 8.197 ms | 16,146 MiB |
| 8K | M | 0.533 s | 2.587 s | 19.076 ms | 15,890 MiB |
| 8K | C | 0.531 s | 2.095 s | 12.819 ms | 24,014 MiB |
| 12K | S | 0.887 s | 2.209 s | 9.886 ms | 17,232 MiB |
| 12K | M | 0.896 s | 2.144 s | 9.718 ms | 17,166 MiB |
| 12K | C | 0.911 s | 2.234 s | 10.259 ms | 24,024 MiB |

### Qwen 2.5 7B — legacy

| Prompt | Mode | TTFT | Scenario | TPOT | Peak VRAM |
|---:|:---:|---:|---:|---:|---:|
| 4K | S | 0.985 s | 2.540 s | 12.242 ms | 12,900 MiB |
| 4K | M | 0.972 s | 2.502 s | 12.098 ms | 12,910 MiB |
| 4K | C | 0.967 s | 2.522 s | 12.110 ms | 21,094 MiB |
| 8K | S | 2.075 s | 3.950 s | 14.734 ms | 22,134 MiB |
| 8K | M | 2.043 s | 3.925 s | 14.726 ms | 22,372 MiB |
| 8K | C | 2.027 s | 3.898 s | 14.594 ms | 24,036 MiB |

### Qwen 2.5 7B — fpA-intB

| Prompt | Mode | TTFT | Scenario | TPOT | Peak VRAM |
|---:|:---:|---:|---:|---:|---:|
| 4K | S | 0.969 s | 2.701 s | 15.546 ms | 17,860 MiB |
| 4K | M | 0.985 s | 2.634 s | 13.310 ms | 16,842 MiB |
| 4K | C | 0.994 s | 2.751 s | 14.805 ms | 23,956 MiB |
| 8K | S | 2.090 s | 6.603 s | 32.711 ms | 24,006 MiB |
| 8K | M | 22.714 s | 24.553 s | 14.488 ms | 24,002 MiB |
| 8K | C | 20.128 s | 22.006 s | 14.761 ms | 24,010 MiB |

## Benchmark design

The script creates a temporary `genai_config.json` for each mode and never
modifies the model package. For the original 1,024-token results detailed
below, the controller launches every mode in a fresh worker process and
rotates the six possible mode orders:

- `scratch`: static workspace preallocation is disabled.
- `matmul`: static preallocation is enabled without the non-windowed GQA bound.
- `combined`: static preallocation is enabled for both `MatMulNBits` and GQA.

That workload uses batch 1, a 1,024-token prompt, and up to 128 generated
tokens. The GQA upper bound and KV-cache capacity are 1,152 tokens. The
long-context summary above instead uses 4K, 8K, or 12K prompts and sets the
bound and cache capacity to prompt length plus 128. Add `--fpa-intb` to set
`ep.cuda.fpa_intb_gemm=1`; the default `--no-fpa-intb` setting uses the legacy
`MatMulNBits` path.

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
affect it. The original 1,024-token results use six order-balanced blocks, one
for each permutation of the three workspace modes. The long-context summary
uses one block and therefore does not provide the same process-level
replication or order balancing.

For the original balanced results, an absolute latency is the median of the
six process-level 10% trimmed means. Paired percentage changes are computed
between adjacent processes within each block and reported as median
`[minimum, maximum]`.

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

Models configured for paged dynamic batching use the ORT GenAI `Engine` API
instead of `Generator`. For Engine workers, `append_tokens_ms` measures
`begin_turn()` admission rather than prefill, and `sampling_ms` includes Engine
execution from admission through the first token, including prefill. Request
and model TTFT retain their definitions above.

## Runtime preallocated-workspace verification

`ORT_CUDA_TRACE_PREALLOCATED_WORKSPACE=1` enables an execution-only diagnostic
for `MatMulNBits` and GQA. `MatMulNBits` reports the selected workspace path,
node, requested bytes, and whether slot 0 supplied the pointer. GQA reports the
slot-0 root source and the number and bytes of subregions served by that root or
by individual scratch fallbacks.

The diagnostic used the Qwen 2.5 1.5B and 7B combined modes with one complete
warmup and one measured 1,024-prefill plus 128-generated-token scenario. The
command was repeated with `--no-fpa-intb` and `--fpa-intb`:

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
| Qwen 1.5B, fpA-intB | 10,908 fpA-intB CUTLASS requests plus 28 legacy requests | 113 fpA-intB plus 28 legacy | 123,273,472 B | 28 | 3,556 | 0 |
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
remaining 28 QKV projections used the legacy path. Qwen 1.5B selected CUTLASS
for 10,908 measured invocations, including decode, while Qwen 7B selected it
for 240 measured invocations. This confirms that the planned modes consume the
cached slot-0 buffers rather than merely registering workspace plans. All four
rows were reverified after rebasing.

## Legacy `MatMulNBits` results

`ep.cuda.fpa_intb_gemm` was disabled for these runs. Warm TTFT used three
complete warmups and 30 measured one-token requests per process.

| Model | Scratch request TTFT | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 63.284 ms | 63.635 ms | 63.813 ms | +1.444% `[-0.283%, +2.489%]` | -0.043% `[-0.920%, +2.834%]` | -0.703% `[-3.327%, +1.211%]` |
| Qwen 2.5 7B | 278.844 ms | 279.053 ms | 279.232 ms | +0.082% `[-0.093%, +0.500%]` | +0.125% `[-0.434%, +0.233%]` | +0.030% `[-0.562%, +0.221%]` |

The model-side TTFT medians, measured from the start of `append_tokens`, were
60.973/61.443/61.582 ms for Qwen 1.5B and 276.423/276.583/276.835 ms for Qwen
7B. The corresponding `append_tokens` prefill medians were
60.453/60.889/61.041 ms and 275.830/275.995/276.198 ms. The paired ranges span
zero or remain below 0.5%, so the rerun does not establish a warm TTFT change.

The full scenario used three warmups and ten measured 1,024-prefill plus
128-generated-token requests per process:

| Model | Scratch scenario | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 997.656 ms | 987.188 ms | 968.611 ms | -0.137% `[-5.253%, +1.896%]` | -1.407% `[-7.270%, +12.557%]` | -0.874% `[-3.773%, +11.480%]` |
| Qwen 2.5 7B | 1,598.665 ms | 1,620.927 ms | 1,618.198 ms | +1.361% `[-0.026%, +2.882%]` | +1.598% `[-0.354%, +2.960%]` | -0.168% `[-0.684%, +1.584%]` |

| Model | Scratch TPOT | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 7.012 ms | 6.845 ms | 6.832 ms | -2.114% `[-6.755%, +4.052%]` | -1.460% `[-9.712%, +11.523%]` | -0.059% `[-3.655%, +12.295%]` |
| Qwen 2.5 7B | 10.373 ms | 10.539 ms | 10.478 ms | +1.406% `[+0.595%, +2.630%]` | +1.404% `[-0.607%, +2.395%]` | -0.436% `[-1.195%, +0.710%]` |

The 1.5B ranges are much larger than the paired medians. For 7B, both planned
modes were about 1.4% slower than scratch in median TPOT, but combined remained
within 0.5% of MatMul-only planning, so the result does not identify an
incremental GQA-planning cost. Warm TTFT remained neutral.

All Qwen 1.5B workers were hash-consistent. The Qwen 7B scenario sampled two
stable process-level hashes in scratch and MatMul-only workers but only one in
the six combined workers, so the controller failed its strict sampled-set
check after writing the report. Six additional combined-only workers
reproduced both hashes, showing that the missing hash was a finite-sample
fresh-process variation rather than a combined-planning-only output.

Peak VRAM used six order-balanced fresh-process blocks per model. Each worker
performed two warmups and one measured scenario. Values are median `[minimum,
maximum]` device-wide `nvidia-smi memory.used` readings:

| Model | Scratch peak | MatMulNBits planned peak | Combined planned peak | MatMul minus scratch | Combined minus scratch | Combined minus MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 4,067 MiB `[4,002, 4,260]` | 4,099 MiB `[4,002, 4,258]` | 4,005 MiB `[4,002, 4,068]` | +33 MiB `[-256, +132]` | -64 MiB `[-258, +66]` | -35 MiB `[-256, +4]` |
| Qwen 2.5 7B | 8,189 MiB `[8,164, 9,220]` | 8,164 MiB `[7,396, 9,202]` | 8,175 MiB `[8,162, 8,322]` | -129 MiB `[-1,056, +1,014]` | -23 MiB `[-1,040, +150]` | +11 MiB `[-1,008, +766]` |

Every measured scenario remained at its post-warmup device residency, so the
sampled scenario delta was 0 MiB in every worker. The broad, quantized
fresh-process ranges remain larger than the paired mode differences. The 7B
memory run had the same sampled hash-set mismatch as the scenario run; the
additional combined diagnostics above reproduced the second hash.
Device-wide `nvidia-smi` therefore establishes only an approximate
steady-state envelope on this WDDM system.

## fpA-intB results

The same six-block matrices were repeated with `--fpa-intb`. All other
workload, warmup, iteration, process-order, build, and reporting settings were
unchanged.

Warm TTFT measurements were:

| Model | Scratch request TTFT | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 74.866 ms | 60.079 ms | 61.900 ms | -11.603% `[-28.199%, +0.499%]` | -16.972% `[-37.643%, +9.665%]` | +2.781% `[-30.940%, +9.121%]` |
| Qwen 2.5 7B | 240.027 ms | 238.276 ms | 239.995 ms | -0.509% `[-3.447%, +0.021%]` | -0.028% `[-2.774%, +0.686%]` | +0.733% `[+0.083%, +1.584%]` |

The Qwen 1.5B request-TTFT median was distorted by slow fresh-process
generator-setup workers in scratch mode. Its model-side medians were
60.702/56.285/56.815 ms, substantially closer than the request medians, and
the paired request ranges span zero. Qwen 7B model-side medians were
237.448/236.060/237.712 ms. These measurements do not establish a
workspace-planning TTFT effect.

The scenario and TPOT measurements were:

| Model | Scratch scenario | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 1,309.432 ms | 1,052.767 ms | 1,493.837 ms | -7.870% `[-52.143%, +7.892%]` | -4.157% `[-10.918%, +28.004%]` | +13.773% `[-12.874%, +92.648%]` |
| Qwen 2.5 7B | 1,582.895 ms | 1,532.977 ms | 1,530.203 ms | -3.480% `[-38.465%, +2.833%]` | -3.284% `[-39.363%, -0.034%]` | +0.005% `[-2.787%, +0.709%]` |

| Model | Scratch TPOT | MatMulNBits planned | Combined planned | MatMul vs scratch | Combined vs scratch | Combined vs MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 9.794 ms | 7.514 ms | 10.627 ms | -10.605% `[-50.704%, +4.425%]` | -4.453% `[-9.386%, +18.391%]` | +14.517% `[-9.927%, +86.776%]` |
| Qwen 2.5 7B | 10.587 ms | 10.234 ms | 10.197 ms | -3.920% `[-35.608%, +3.030%]` | -3.434% `[-36.642%, -0.077%]` | -0.040% `[-3.015%, +0.639%]` |

Qwen 1.5B again showed large process/tactic variation, including scenario
medians from about 981 to 2,078 ms. For Qwen 7B, both planned modes were about
3.3% below scratch, but combined and MatMul-only differed by approximately
zero; the apparent difference therefore is not an incremental GQA-planning
effect, and one slow scratch process widened the paired range.

Every Qwen 1.5B worker produced one identical output hash. The Qwen 7B
scenario produced the same two process-level hashes in every mode. The Qwen
7B memory sample failed the strict sampled-set check because scratch observed
three hashes, MatMul-only one, and combined two. Six additional combined-only
workers produced three hashes, including both balanced combined hashes and a
further variant. Each worker remained internally stable, so the data shows
fresh-process tactic/numerical variability; the failed strict memory report is
retained rather than reported as a correctness pass.

Peak VRAM was:

| Model | Scratch peak | MatMulNBits planned peak | Combined planned peak | MatMul minus scratch | Combined minus scratch | Combined minus MatMul |
|---|---:|---:|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 4,943 MiB `[4,940, 5,198]` | 4,940 MiB `[4,940, 5,070]` | 4,942 MiB `[4,940, 5,196]` | -1 MiB `[-258, +126]` | -1 MiB `[-258, +252]` | +1 MiB `[0, +126]` |
| Qwen 2.5 7B | 12,101 MiB `[11,072, 12,352]` | 11,339 MiB `[11,072, 12,386]` | 12,102 MiB `[11,586, 13,120]` | -369 MiB `[-1,056, +516]` | +22 MiB `[-544, +2,048]` | +640 MiB `[-288, +1,532]` |

The 1.5B ranges overlap and do not establish a memory difference. The 7B
planned results remain highly tactic-dependent: MatMul-only had a lower median,
while combined returned to the scratch median and ranged up to 13,120 MiB.
The paired ranges are too broad to attribute either behavior to workspace
planning. These are device-wide WDDM readings after warmup, not process-local
allocator measurements.

## Dispatch-path comparison

An unpaired comparison of fpA-intB and legacy medians summarizes why the
dispatch paths must be reported separately:

| Model and metric | Scratch | MatMulNBits planned | Combined planned |
|---|---:|---:|---:|
| Qwen 2.5 1.5B TTFT | +18.30% | -5.59% | -3.00% |
| Qwen 2.5 1.5B scenario | +31.25% | +6.64% | +54.22% |
| Qwen 2.5 1.5B TPOT | +39.67% | +9.77% | +55.55% |
| Qwen 2.5 7B TTFT | -13.92% | -14.61% | -14.05% |
| Qwen 2.5 7B scenario | -0.99% | -5.43% | -5.44% |
| Qwen 2.5 7B TPOT | +2.06% | -2.89% | -2.68% |

Negative values favor fpA-intB. These are unpaired comparisons and are
especially misleading for Qwen 1.5B, where fresh-process generator setup and
decode tactic outliers moved the aggregate medians substantially. Qwen 7B
still shows the repeatable fpA-intB prefill advantage, while scenario and TPOT
remain sensitive to process-level tactic selection.

## Preliminary Qwen 2.5 long-context TTFT

The Qwen 2.5 long-context runs used fpA-intB dispatch and one fresh process per
workspace mode. At 8K for both models and 16K for Qwen 1.5B, each process used
one warmup and three measured requests. These runs do not use the six-block
order-balanced methodology of the main results.

| Model | Prompt | Scratch request TTFT | MatMulNBits planned | Combined planned |
|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 8,192 | 0.516 s | 0.523 s (+1.2%) | 0.564 s (+9.3%) |
| Qwen 2.5 1.5B | 16,384 | 50.396 s | 53.700 s (+6.6%) | 17.015 s (-66.2%) |
| Qwen 2.5 7B | 8,192 | 18.735 s | 24.692 s (+31.8%) | 24.460 s (+30.6%) |

The different direction and magnitude by model and mode do not establish a
workspace-planning latency effect. Fresh processes independently profile and
select fpA-intB tactics. The Qwen 1.5B 16K combined result therefore requires
order-balanced fresh-process reproduction before it can be attributed to
workspace planning. All modes produced matching output hashes.

Qwen 7B at 16K completed one warmup but OOMed on its first measured request.
A cold single request per fresh process completed at 211.626/477.533/212.132
seconds for scratch/MatMulNBits-planned/combined. Those values include lazy
runtime tactic profiling and are not comparable to the warm TTFT rows above;
they show only that a single 16K request fits. Repeating the request in the
same process does not fit with the current allocator state.

At 32K, a maximum-valid 32,767-token prompt requested approximately 19.9 GiB
for attention and failed GPU allocation even in scratch mode. A 64K prompt
exceeds the exported models' 32,768-token context and rotary-cache capacity.
No valid 32K or 64K latency comparison is available for these model packages
on this GPU.

## Preliminary Qwen 3.8 27B long-context TTFT

Qwen 3.8 27B used the ORT GenAI `Engine` API, fpA-intB dispatch, dynamic
batching, and an INT8 paged KV cache. The model has 433 `MatMulNBits` nodes and
16 `PagedAttention` nodes, but no GQA nodes. Consequently, scratch and
MatMulNBits-planned are the only distinct workspace modes; combined planning
would be equivalent to MatMulNBits-only planning.

These runs are deliberately preliminary because one request took 14 to 77
minutes. Each mode used only one fresh process. The 16K value is the median of
three measured requests after one warmup in that process; the 32K and 64K
values each contain one measured request without a warmup.

| Prompt length | Scratch request TTFT | MatMulNBits planned | Planned vs scratch |
|---|---:|---:|---:|
| 16,384 | 861.240 s | 829.303 s | -3.708% |
| 32,768 | 1,892.226 s | 1,948.282 s | +2.962% |
| 65,536 | 4,405.860 s | 4,631.681 s | +5.125% |

The direction reverses between 16K and 32K/64K, so these single-process
measurements do not establish either a workspace-planning speedup or
regression. Fresh processes independently profile fpA-intB tactics, and the
cost prevented the order-balanced repeated-process methodology used above.
Scratch and planned runs produced matching output hashes at every prompt
length.

The model's default 50% paged-cache allocation and a 90% allocation both
rejected the 64K request as `REQUEST_UNSERVICEABLE`. An explicit 258-block
pool was also insufficient. Both modes completed with 512 blocks, which
requires approximately 4 GiB of raw main-model INT8 KV-cache storage. This
shows that the earlier rejection was an Engine admission-capacity constraint,
not the model's 262,144-token context limit.

## Conclusions

- Legacy warm TTFT remained neutral. fpA-intB Qwen 7B was also effectively
  neutral across workspace modes; Qwen 1.5B request TTFT was dominated by
  fresh-process setup outliers.
- Scenario and TPOT differences remain smaller than, or inseparable from,
  fresh-process/tactic variation. Combined planning did not show an
  incremental latency benefit over MatMulNBits-only planning.
- The runtime trace proved that every planned `MatMulNBits` and GQA node
  consumed slot-0 preallocated memory after warmup, with no GQA subregion
  fallback.
- Device-wide `nvidia-smi` memory is too quantized and process-variable for
  precise workspace attribution on this WDDM system. Use the separate C++
  benchmark report for process-local WDDM and ORT arena measurements.
- Preliminary Qwen 3.8 27B long-context TTFT changed by -3.7%, +3.0%, and
  +5.1% at 16K, 32K, and 64K respectively. The opposite directions and
  one-process methodology do not support a latency attribution.
- The strict Qwen 7B sampled hash-set checks failed in two legacy phases and
  the fpA-intB memory phase. Additional fresh-process diagnostics reproduced
  omitted hashes while every individual worker remained internally stable;
  the reports retain these failures instead of treating them as green.
