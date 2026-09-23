# Initial ORT GenAI WebGPU Workspace Preallocation Benchmark Results

**Historical results.** For the controlled scratch/planned x disabled/bucket
rerun with matching C++ provider settings and common whole-process/post-warmup
memory sampling, use
[Controlled WebGPU Cache and Workspace Benchmark](workspace_controlled_benchmark_results.md).
The results below retain the original default-provider GenAI configuration.

## Scope

The shared
[`ort_genai_workspace_benchmark.py`](../cuda/ort_genai_workspace_benchmark.py)
now accepts `--execution-provider webgpu`. CUDA remains the default, with its
existing scratch/MatMulNBits/combined modes and provider settings. WebGPU uses
two modes: **scratch** disables static workspace preallocation and **planned**
enables it for eligible WebGPU kernel workspaces. There is no CUDA fpA-intB
switch or CUDA GQA workspace bound in the WebGPU configuration.

This reproduces the long-context request workload in the
[CUDA GenAI report](../cuda/ort_genai_workspace_benchmark_results.md), not the
[WebGPU C++ provider microbenchmark](workspace_preallocation_benchmark_test.cc).
The latter uses repeated BOS tokens, a supplied one-token past cache, full
logits fetched by the test, no autoregressive generation, and disabled storage
buffer caching by default. Its latency, allocator statistics, and capacity
boundary are not interchangeable with the GenAI results.

## Results at a glance

The 128-token scenario completed in both modes for Qwen 2.5 1.5B at 4K, 8K,
and 12K, and Qwen 2.5 7B at 4K. Scratch/planned output hashes matched, including
between latency and memory processes. Qwen 2.5 7B at 8K failed during warmup
in both modes; 12K was not attempted for that model.

Here 4K = 4,096 prompt tokens, 8K = 8,192, and 12K = 12,288. TTFT below is
measured inside the **128-generated-token request**, not the separate
one-token TTFT phase. Each successful row uses two fresh latency processes
and two separate memory processes.

| Model | Prompt | Mode | Request TTFT | Scenario, 128 tokens | TPOT | Post-warmup peak VRAM |
|---|---:|---|---:|---:|---:|---:|
| Qwen 2.5 1.5B | 4K | Scratch | 0.763 s | 1.821 s | 6.633 ms | 3,316 MiB |
| Qwen 2.5 1.5B | 4K | Planned | 0.744 s | 1.740 s | 6.552 ms | 3,444 MiB |
| Qwen 2.5 1.5B | 8K | Scratch | 1.819 s | 3.163 s | 7.865 ms | 4,933 MiB |
| Qwen 2.5 1.5B | 8K | Planned | 1.802 s | 3.122 s | 7.599 ms | 5,093 MiB |
| Qwen 2.5 1.5B | 12K | Scratch | 3.216 s | 4.983 s | 8.429 ms | 23,639 MiB |
| Qwen 2.5 1.5B | 12K | Planned | 3.186 s | 5.586 s | 13.982 ms | 17,969 MiB |
| Qwen 2.5 7B | 4K | Scratch | 1.601 s | 3.652 s | 14.444 ms | 7,320 MiB |
| Qwen 2.5 7B | 4K | Planned | 1.597 s | 3.734 s | 14.831 ms | 7,480 MiB |
| Qwen 2.5 7B | 8K | Both | OOM | OOM | N/A | N/A |

Planning did **not** provide a general latency or memory improvement for this
GenAI workload. At 1.5B/4K, scenario latency improved 4.5%, with improvements
in both process orders (-7.0% and -1.8%). The 1.5B/8K and 7B/4K scenario
differences changed sign between process pairs. Those small aggregate
differences should not be treated as repeatable gains or regressions.

At 1.5B/12K, planning lowered the post-warmup sampled peak by **5,670 MiB (24.0%)**, but
scenario latency increased **12.1%** and TPOT increased **65.9%**. Scenario
regressions occurred in both orders (+15.3% and +8.9%). This is a memory/latency
tradeoff, not an across-the-board win. The measurements do not isolate whether
residency management, buffer reuse, or another mechanism caused the difference.
At the other successful workloads, planning raised the sampled peak by
128-160 MiB.

The later controlled rerun measured the whole process as well: at 1.5B/12K
with bucket caching, scratch and planned both reached 23,639 MiB before
restricting sampling to the post-warmup window. The 24.0% figure above is
therefore a steady-state sampled reduction, not a whole-process peak saving.

The earlier cache-disabled C++ prefill results therefore do not predict these
normal-cache GenAI generation results. The workload and resource lifetimes
also differ; changing the Python API alone is not evidence that full prompt
logits are no longer materialized.

### Separate memory processes

Each cell lists the two fresh-process measurements in MiB. All 60 workers
started with `nvidia-smi` reporting zero device memory usage and zero GPU
utilization. The following baselines are sampled **after warmup**, not before
model loading.

| Model / prompt | Scratch baseline | Planned baseline | Scratch peak | Planned peak |
|---|---:|---:|---:|---:|
| 1.5B / 4K | 2,128 / 2,128 | 2,128 / 2,128 | 3,316 / 3,316 | 3,444 / 3,444 |
| 1.5B / 8K | 2,558 / 2,558 | 2,558 / 2,558 | 4,933 / 4,933 | 5,093 / 5,093 |
| 1.5B / 12K | 2,059 / 2,059 | 2,269 / 2,269 | 23,639 / 23,639 | 17,969 / 17,969 |
| 7B / 4K | 6,132 / 6,132 | 6,132 / 6,132 | 7,320 / 7,320 | 7,480 / 7,480 |

The sharp peak increase from 8K to 12K on the 1.5B model is an observed
device-residency effect, not a measurement of the workspace buffer size.
These runs do not expose the exact ORT allocator peak or allocation count.

### Separate one-token TTFT

These use prompt-plus-one capacity, one warmup, three measured requests in
each of two fresh processes per mode, and no memory sampler.

| Model | Prompt | Scratch request TTFT | Planned request TTFT |
|---|---:|---:|---:|
| Qwen 2.5 1.5B | 4K | 0.848 s | 0.847 s |
| Qwen 2.5 1.5B | 8K | 2.058 s | 2.035 s |
| Qwen 2.5 1.5B | 12K | 3.691 s | 3.630 s |
| Qwen 2.5 7B | 4K | 1.745 s | 1.707 s |
| Qwen 2.5 7B | 8K | OOM | OOM |

### Capacity failure

All twelve Qwen 2.5 7B/8K workers failed during `append_tokens()` in the first
warmup: two processes per mode for each of scenario, memory, and one-token
TTFT. Logs report `vkAllocateMemory failed with VK_ERROR_OUT_OF_DEVICE_MEMORY`,
followed by an invalid-buffer validation error. Neither mode completed a
measured request. The memory sampler starts after warmup, so there is no
comparable post-warmup peak VRAM value for these failed runs.

## Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB |
| NVIDIA driver | 610.62 |
| OS / Python | Windows / Python 3.12 |
| ONNX Runtime | `13cb459fe29a30bb9b9c2f15285b46f2875a70bd`, plus the Python benchmark adaptation |
| Runtime build | Release shared library, native WebGPU with Dawn Vulkan; D3D12 disabled |
| ONNX Runtime GenAI | `0.16.0-dev`, `d5b40851ba80ffa8e95b6b01f921dbb9008fac80` |
| Model packages | `qwen2.5-1.5b-instruct-cuda-gpu-4/v4`, `qwen2.5-7b-instruct-cuda-gpu-4/v4` |
| WebGPU buffer caching | Provider defaults; no cache-disabling override |
| GPU graph capture | Not enabled |

GenAI was built against the matching shared ORT build. Its isolated Python
environment loads that custom `onnxruntime.dll`, not the stock runtime installed
as a wheel dependency. The loaded ORT DLL SHA256 is
`0620842756c959d4652602e61e9668439daf25321e3a690d28a2cc357f9bd97d`.
The GenAI DLL SHA256 is
`069206bc4eedd9dbc40d24f55c067c4f9ee935fdb2450b3d08d09b15bb65067c`.

Separate 1,024-prompt/128-generated-token diagnostics on both models confirmed
workspace-pattern recording followed by successful planned-region lookup in
the runtime. The generic framework trace is enabled by
`ORT_MATMULNBITS_TRACE_LEGACY_WORKSPACE=1`, despite that variable's name.
Tracing and verbose logging were disabled for the measurements. A successful
lookup proves consumption by the logged node/slot, not coverage of every node.

## Workload and measurements

- Batch size 1; synthetic prompt token IDs generated with NumPy seed 1234,
  starting with BOS and excluding special tokens thereafter. This is the same
  prompt construction as the CUDA benchmark, not tokenized natural-language
  documents.
- Qwen 2.5 uses the GenAI `Generator` API. Generation is greedy with one beam,
  exactly 128 new tokens, and no early EOS termination. The request and KV-cache
  capacity are prompt length plus 128. Model files are hardlinked into a
  temporary directory; only its GenAI configuration is changed.
- Each mode runs in two fresh processes per phase, in the order
  scratch/planned, planned/scratch. Each latency process has one warmup request
  and three measured requests: six measured requests per mode. Memory uses
  separate processes, each with one warmup and one measured request.
- **Request TTFT** starts before generator/parameter creation and ends after
  the first token is retrieved on the CPU. **Scenario** spans that same start
  through all 128 tokens. Both exclude model loading, prompt construction,
  final output hashing, and generator destruction.
- **TPOT** is the 10% trimmed mean of the 127 decode-token timings after the
  first token, pooled across measured requests in each process. It is not
  scenario time divided by 128.
- **Peak VRAM** is device-wide `nvidia-smi memory.used`, requested every 5 ms
  during the separate post-warmup generation request. It is a sampled maximum,
  not an exact allocator high-water mark or an initialization/first-warmup
  peak. Unlike the latency timers, the memory window also includes result
  hashing and generator cleanup. Other GPU workloads must be absent.
- Every worker verifies the exact output length and decode count, and identical
  hashes across its measured requests. Scratch and planned output hash sets
  are compared across processes. The JSON also retains model-only timings,
  setup/prefill/sampling, percentiles, and stall statistics.

The main table uses the median of the two process-level request medians for
TTFT and scenario, and the median of the two process-level trimmed means for
TPOT. VRAM is the median of two sampled process peaks. With two processes,
these medians are midpoints. Raw per-process memory values are reported
separately so residency variation remains visible.

The controller's JSON `aggregate` instead summarizes process-level trimmed
means for every latency metric; use each `process_results` entry's `median`
for the request-median table. With only three requests per process, request
trimming removes no samples. Two process pairs are preliminary evidence, not
a precise speedup estimate.

An additional **one-token TTFT** phase is separate from the 128-token scenario:
`--phase ttft` forces one generated token and capacity prompt plus one. Do not
substitute it for TTFT measured within a 128-token request.

## Reproduction

Use a GenAI Python environment built against the WebGPU-enabled shared runtime
under test. Verify the loaded DLL, not just `is_webgpu_available()`. On Windows,
GenAI can find ORT under the installed `onnxruntime\capi` directory. Run outside
the ORT source root to avoid shadowing that installed package.

```powershell
$python = "C:\path\to\webgpu-genai-venv\Scripts\python.exe"
$benchmark = "C:\path\to\ort\onnxruntime\test\providers\cuda\ort_genai_workspace_benchmark.py"
$model = "C:\path\to\qwen-model"

& $python $benchmark --execution-provider webgpu --model-path $model `
  --phase scenario --prompt-tokens 4096 --generated-tokens 128 `
  --warmups 1 --iterations 3 --repetitions 2 --output webgpu-4k-scenario.json

& $python $benchmark --execution-provider webgpu --model-path $model `
  --phase memory --prompt-tokens 4096 --generated-tokens 128 `
  --warmups 1 --repetitions 2 --memory-sample-interval-ms 5 `
  --output webgpu-4k-memory.json

& $python $benchmark --execution-provider webgpu --model-path $model `
  --phase ttft --prompt-tokens 4096 `
  --warmups 1 --iterations 3 --repetitions 2 --output webgpu-4k-ttft.json
```

Repeat with prompt lengths 8,192 and 12,288 as capacity permits.
`--device-id` selects the CUDA device in CUDA mode; in WebGPU mode it only
identifies the NVIDIA memory sampler, not a WebGPU adapter. This benchmark
supports index 0 on an otherwise idle single-NVIDIA-GPU system and relies on
the provider's default adapter selection.

To retain results independently when probing an OOM boundary, invoke workers
directly with `--worker --mode scratch` or `--mode planned` and a distinct
`--worker-output` JSON path, redirecting each worker's console output to its
own log. The controller stops on a failing worker.
