# Controlled WebGPU Cache and Workspace Benchmark

## Scope

This is the reference report for WebGPU workspace-preallocation results.
It compares the [C++ cached-prefill benchmark](workspace_preallocation_benchmark_test.cc)
and the [ORT GenAI generation benchmark](../cuda/ort_genai_workspace_benchmark.py),
separating storage-buffer caching from workspace planning.

It supersedes the earlier cache-disabled C++-only report. The
[initial default-provider GenAI measurements](ort_genai_workspace_benchmark_results.md)
are retained as historical data; their provider settings and memory-sampling
window differ from this controlled comparison.

Each harness runs all four configurations in fresh processes:

| Configuration | Storage-buffer cache | Static workspace planning |
|---|---|---|
| Disabled / scratch | `disabled` | Off |
| Disabled / planned | `disabled` | On |
| Bucket / scratch | `bucket` | Off |
| Bucket / planned | `bucket` | On |

Within each harness, the model, inputs, provider options, warmup count,
measurement count, and memory instrumentation are held constant. The C++
and GenAI workloads are deliberately retained rather than silently changing
the earlier tests. **Matching provider settings does not make those two
different workloads interchangeable.**

## Results and interpretation

All 96 workers completed: 80 successful workers and 16 OOM outcomes.
Every OOM occurred at 1.5B/12K with caching disabled, in both harnesses and
both workspace modes. Bucket caching allowed that context to run in both
modes. Successful GenAI output hashes matched across all four configurations
and both measurement phases.

**Storage-buffer caching explains most of the earlier large VRAM discrepancy.**
With bucket caching and the same external counter, the C++ and GenAI
whole-process peaks agree to within 8-16 MiB at every successful focused
context. For example, 7B/4K scratch uses 21,151 MiB in C++ and 21,067 MiB in
GenAI with caching disabled, versus 7,328 and 7,320 MiB with bucket caching.
This is not an intrinsic C++-versus-Python memory difference.

**The earlier 24% reduction at 1.5B/12K was a post-warmup reduction, not a
whole-process peak reduction.** With bucket caching, both GenAI modes peak
at 23,639 MiB over the whole process. Planning lowers only the post-warmup peak,
from 23,639 to 17,969 MiB. C++ shows the same distinction. Its planned worker
records approximately 23,646 MiB WDDM usage at the first warmup endpoint and
17,976 MiB at subsequent warmup endpoints. The first recording pass still
uses scratch allocation, so steady-state savings cannot be assumed to reduce
the initial peak.

**Prefill benefits do not imply decode benefits.** Planning improves the
cache-disabled C++ prefill means by about 5-6% in this rerun, with improvements
in both process orders. Under bucket caching, the C++ differences shrink to
about 0.6-1.0%. GenAI's disabled-cache prefill changes are small, while its
decode time increases 21-25%, making the full scenario about 12% slower.
At 1.5B/12K with bucket caching, planning increases decode time 41.1% and the
full scenario 15.1%, despite the lower post-warmup residency.

These measurements establish the cache/planning interaction, not the exact
driver or allocator mechanism behind every change. Two process pairs and
three timed requests per process are insufficient to characterize small
latency differences precisely; they should not be read as universal speedup
estimates. The earlier C++ report used 30 timed forwards per process, so this
rerun reproduces the workload and comparisons rather than its exact sampling
count or historical percentages.

### C++ cached-prefill latency

All values are **scratch / planned**. Each value is the median of the two
process-level means. Planning change is calculated from unrounded values.

| Model / prompt | Cache | Prefill (s) | Planning change |
|---|---|---:|---:|
| 1.5B / 8K | Disabled | 1.887 / 1.792 | -5.0% |
| 1.5B / 8K | Bucket | 1.662 / 1.648 | -0.8% |
| 1.5B / 12K | Disabled | OOM / OOM | N/A |
| 1.5B / 12K | Bucket | 3.054 / 3.024 | -1.0% |
| 7B / 4K | Disabled | 1.770 / 1.668 | -5.8% |
| 7B / 4K | Bucket | 1.508 / 1.499 | -0.6% |

### GenAI prefill and decode latency

All values are **scratch / planned**. Prefill is `append_tokens()`; decode
starts after first-token retrieval and covers the remaining 127 tokens.
The scenario includes setup and first-token sampling as well.

| Model / prompt | Cache | Prefill (s) | Request TTFT (s) | Decode (s) | TPOT (ms) | Scenario (s) |
|---|---|---:|---:|---:|---:|---:|
| 1.5B / 8K | Disabled | 1.828 / 1.822 | 1.835 / 1.829 | 1.657 / 2.070 | 8.958 / 12.464 | 3.491 / 3.898 |
| 1.5B / 8K | Bucket | 1.797 / 1.770 | 1.821 / 1.790 | 1.494 / 1.472 | 8.978 / 8.984 | 3.315 / 3.262 |
| 1.5B / 12K | Disabled | OOM / OOM | OOM / OOM | N/A | N/A | OOM / OOM |
| 1.5B / 12K | Bucket | 3.143 / 3.150 | 3.164 / 3.171 | 1.819 / 2.566 | 9.236 / 15.114 | 4.983 / 5.737 |
| 7B / 4K | Disabled | 1.615 / 1.599 | 1.623 / 1.606 | 2.280 / 2.760 | 15.069 / 18.966 | 3.903 / 4.366 |
| 7B / 4K | Bucket | 1.560 / 1.561 | 1.583 / 1.584 | 2.106 / 2.112 | 15.105 / 15.162 | 3.689 / 3.697 |

The disabled-cache GenAI scenario regressions occurred in both process orders:
9.4% and 14.0% at 1.5B/8K, and 10.6% and 13.1% at 7B/4K. The 1.5B/12K bucket
scenario regression was also present in both orders: 11.7% and 18.6%.

### Common device-memory measurements

All values are **scratch / planned**, in MiB, using the median of two
fresh-process sampled peaks. Whole-process and post-warmup peaks come from
the same external sample stream for each worker.

| Model / prompt | Harness | Cache | Whole-process sampled peak | Post-warmup sampled peak |
|---|---|---|---:|---:|
| 1.5B / 8K | C++ | Disabled | 17,606 / 17,392 | 17,392 / 14,824 |
| 1.5B / 8K | C++ | Bucket | 4,941 / 5,101 | 4,941 / 5,101 |
| 1.5B / 8K | GenAI | Disabled | 19,576 / 17,264 | 17,602 / 14,992 |
| 1.5B / 8K | GenAI | Bucket | 4,933 / 5,093 | 4,933 / 5,093 |
| 1.5B / 12K | C++ | Disabled | 23,670 / 23,733 (OOM) | N/A |
| 1.5B / 12K | C++ | Bucket | 23,655 / 23,655 | 23,655 / 17,985 |
| 1.5B / 12K | GenAI | Disabled | 23,882 / 23,864 (OOM) | N/A |
| 1.5B / 12K | GenAI | Bucket | 23,639 / 23,639 | 23,639 / 17,969 |
| 7B / 4K | C++ | Disabled | 21,151 / 20,517 | 21,151 / 17,559 |
| 7B / 4K | C++ | Bucket | 7,328 / 7,488 | 7,328 / 7,488 |
| 7B / 4K | GenAI | Disabled | 21,067 / 20,475 | 20,577 / 18,257 |
| 7B / 4K | GenAI | Bucket | 7,320 / 7,480 | 7,320 / 7,480 |

OOM peaks describe failed processes, not the memory required for a completed
request. They must not be treated as successful lower-memory configurations.
For example, the two C++ disabled-cache 12K scratch processes sampled 23,878
and 23,462 MiB before failure. All eight disabled-cache 12K memory processes
failed without a completed post-warmup measurement window.

The bucket-cache peaks were identical between the two processes per
configuration. Disabled-cache residency varied: for example, 7B/4K C++ scratch
whole-process peaks were 20,689 and 21,613 MiB, and planned peaks were 20,221
and 20,813 MiB. Raw per-process values remain in `results.json` and the worker
logs rather than being hidden by the medians.

Across the 78,732 consecutive sample intervals, the actual median spacing was
15.99 ms, the 95th percentile was 16.09 ms, and the largest gap was 102.42 ms.
The requested 5 ms interval therefore must not be described as an achieved
5 ms sampling cadence.

## Configuration

Both harnesses use the RTX 5090 Laptop GPU, NVIDIA driver 610.62, Windows
Release builds, the native Dawn Vulkan backend, and the same Qwen model
packages as the earlier reports.

| Setting | Both harnesses |
|---|---|
| WebGPU `enableInt64` | `1` |
| WebGPU `enableGraphCapture` | `0` |
| WebGPU `preferredLayout` | `NHWC` |
| Storage-buffer cache | Explicit `disabled` or `bucket` |
| Other provider options | Defaults from the same ORT source revision |
| Warmups per process | 5 |
| Timed measurements per latency process | 3 |
| Measured requests per memory process | 3 |
| Fresh processes per configuration, per phase | 2 |
| External device-memory sampler | `nvidia-smi`, requested interval 5 ms |

The source revision is `13cb459fe29a30bb9b9c2f15285b46f2875a70bd` plus benchmark
instrumentation. The C++ provider-test target was rebuilt. The GenAI build is
`d5b40851ba80ffa8e95b6b01f921dbb9008fac80`, loading the matching WebGPU ORT DLL
with SHA256
`0620842756c959d4652602e61e9668439daf25321e3a690d28a2cc357f9bd97d`.
The controller records executable and script hashes in `provenance.json`.

The C++ workload is one cached-prefill forward with repeated BOS tokens,
a supplied one-token zero past cache, and full logits requested. The GenAI
workload uses the same seeded synthetic prompt construction as the CUDA
GenAI benchmark, shared past/present cache capacity of prompt plus 128,
and exactly 128 greedy generated tokens.

The focused cases are Qwen 2.5 1.5B at 8,192 and 12,288 prompt tokens, and
Qwen 2.5 7B at 4,096 prompt tokens. Contexts that fail remain visible as OOM
outcomes; they are not replaced with shorter prompts.

### Ordering and correctness

The first block runs disabled/scratch, disabled/planned, bucket/planned,
bucket/scratch. The second block reverses that order. The order of the C++
and GenAI harnesses also reverses between blocks. Every worker waits for
device 0 to report zero memory usage and zero utilization before starting.

The C++ test checks model placement, output shape, and workspace declarations.
GenAI checks exact token/decode counts and identical output hashes, both
within a worker and across successful cache/planning configurations and
measurement phases. Cross-harness hashes are not compared: their inputs and
outputs differ.

Separate diagnostics with matching provider settings confirmed that GenAI
planned mode consumes a planned workspace region with both cache modes.
Tracing and verbose logging are disabled in the measured workers.

## Metrics

Latency and memory use separate processes. Memory sampling never runs during
the latency measurements.

- **C++ prefill:** complete `Session::Run` time for the cached-prefill forward.
- **GenAI prefill:** `append_tokens()` time, retaining the CUDA benchmark's
  timing boundary.
- **Request TTFT:** generator setup, prefill, and first-token sampling/retrieval.
- **Decode:** wall time after first-token retrieval through completion of the
  remaining 127 tokens, excluding final hashing and generator destruction.
- **TPOT:** the 10% trimmed mean of the individual decode-token timings.
- **Scenario:** end-to-end request time through all 128 generated tokens.

For each latency metric, each worker reports its mean or 10% trimmed mean.
There are only three complete requests per worker, so trimming removes no
complete requests. Decode-token trimming pools the 381 measured token times
per GenAI worker. Summary tables use the median of the two worker values.

### Identical memory counters and explicit windows

A single **external** sampler observes each memory worker from before process
launch until after process exit. Both harnesses use the same device-wide
`nvidia-smi memory.used` counter:

- **Whole-process sampled peak:** includes initialization, all five warmups,
  all three measured requests, and teardown. Failed workers also retain this
  measurement.
- **Post-warmup sampled peak:** only samples whose receipt timestamps fall
  inside the child-reported window around the three measured requests.
  GenAI's window includes per-request hashing and generator cleanup; the C++
  window includes output clearing between forwards and ends with the last
  fetched output still live.

The memory controller does not start a second sampler inside GenAI. Each raw
sample is saved with a Unix timestamp, and reports retain sample counts and
the maximum observed gap. A requested 5 ms interval is not a guarantee of
5 ms observation spacing on Windows. These are sampled device-usage maxima,
not exact allocator high-water marks. An OOM allocation itself never becomes
resident, so the sampled peak of a failed process understates its requested
memory requirement.

The existing C++ allocator/WDDM statistics remain in its per-worker JSON for
diagnostics, but the cross-harness memory tables use only the common external
counter. A worker failing before completing the measurement window has no
post-warmup result.

## Reproduction

The controller is
[`workspace_controlled_benchmark.py`](workspace_controlled_benchmark.py).
It checks the effective provider options and run counts, rejects skipped C++
tests and unexpected non-OOM failures, and preserves each worker's results
and log. Use a new output directory for every complete run.

```powershell
.\.venv\Scripts\python.exe `
  onnxruntime\test\providers\webgpu\workspace_controlled_benchmark.py `
  --cpp-executable C:\path\to\onnxruntime_provider_test.exe `
  --genai-python C:\path\to\webgpu-genai-venv\Scripts\python.exe `
  --qwen15-model-path C:\path\to\qwen-1.5b `
  --qwen7-model-path C:\path\to\qwen-7b `
  --output-dir C:\path\to\controlled-results `
  --warmups 5 --iterations 3 --memory-iterations 3 --repetitions 2
```

The default case list is the focused set above. Override it with repeatable
`--case qwen15:8192` / `--case qwen7:4096` arguments. A short instrumentation
smoke run uses `--case qwen15:1024 --warmups 1 --iterations 1
--memory-iterations 1 --repetitions 1`; its timings are not part of the main
results.

GenAI can also be invoked directly with
`--webgpu-storage-buffer-cache-mode disabled` or `bucket`, together with
`--webgpu-controlled-comparison`. The latter matches the C++ int64, graph
capture, and layout settings. `--external-memory-sampling` is for memory
workers launched under an external sampler; it emits the measurement window
instead of collecting memory internally.

The C++ benchmark retains its original defaults. The controller selects
cache mode and measurement phase through
`ORT_WEBGPU_WORKSPACE_BENCHMARK_STORAGE_BUFFER_CACHE_MODE` and
`ORT_WEBGPU_WORKSPACE_BENCHMARK_PHASE` (`latency`, `memory`, or default `all`).
Warmup, timing, and memory counts can be overridden with the respective
`ORT_WEBGPU_WORKSPACE_BENCHMARK_WARMUPS`, `_ITERATIONS`, and `_MEMORY_RUNS`
environment variables.
