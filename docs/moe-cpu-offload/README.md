# MoE/QMoE CPU offload

Keep `K` experts on CPU to reduce GPU memory. The other experts stay on GPU.
CPU offload happens inside the built-in CUDA kernels, not through CPU execution-provider fallback.

## Initial placement and ownership

| Strategy | Placement |
|---|---|
| `adaptive` (default, MoE/QMoE) | Put popular experts on GPU and exchange hot CPU experts with cold GPU experts. Without saved counters, distribute GPU slots round-robin across nodes. |
| `basic` (QMoE only) | Fill chosen QMoE layers on GPU, then put the rest on CPU. Placement never changes and ignores popularity. |
| `layered` (QMoE only) | Fill chosen layers up to a configurable GPU fraction (default 80%), then leave the rest on CPU. Use popularity and swaps within GPU-resident layers, including the boundary layer. |

`K` is global across eligible nodes. Basic uses topological order within each graph; a boundary node may be split
to preserve the exact count, keeping its lower expert IDs on GPU. `K=0` disables offload.

For basic/layered, `session.moe_cpu_offload_layer_order=first|last|middle` chooses GPU layers.
`first` (default) follows graph-scope/topological order, `last` reverses it, and `middle` expands outward:
for 8 layers, `3, 4, 2, 5, 1, 6, 0, 7`. Execution order and saved counter numbering do not change.

Layered uses the same ordering and rounds each layer's GPU target down to whole experts.
If the budget exceeds all layer targets combined, only the necessary excess is added in the chosen layer order,
preserving exactly `K` CPU experts.

Adaptive and layered track selection with `counter = alpha * counter + (1 - alpha) * selected`.
A background swap requires `hottest_CPU > (1 + epsilon) * coldest_GPU`. Each node keeps its initial GPU capacity.
CPU/GPU weights occupy separate compact stores; computation overlaps and completed swaps are published safely.
Supported INT4 SwiGLU configurations use packed MLAS Q4 weights on CPU; other QMoE cases use a dense fallback.

## Configuration

Session keys below use the `session.` prefix. Defaults are runtime defaults; the benchmark defaults to
alpha `0.9` and epsilon `0.1` and requires an explicit `--cpu-offload-experts` value.

| Session key | Benchmark option | Default | Description |
|---|---|---|---|
| `moe_cpu_offload_experts` | `--cpu-offload-experts K` | `0` | Exact number of CPU-resident experts across eligible nodes. `0` disables offload. |
| `moe_cpu_offload_strategy` | `--strategy` | `adaptive` | Placement policy: `adaptive`, `basic`, or `layered`, as described above. Basic never swaps. |
| `moe_cpu_offload_gpu_ratio` | `--gpu-ratio` | `0.8` | Per-layer GPU target in `(0, 1]`, for layered placement only. Exceeded only as needed to preserve exactly `K`. |
| `moe_cpu_offload_layer_order` | `--layer-order` | `first` | GPU layer priority for basic/layered: `first`, `last`, or `middle`. Does not change execution order. |
| **`moe_cpu_offload_on_demand_threshold`** | **`--on-demand-threshold N`** | **`0`** | Synchronously swap required CPU experts onto GPU if at most `N` are missing and all selected experts fit existing GPU capacity. CUDA QMoE adaptive/layered only; `0` disables it. |
| `moe_expert_counter_alpha` | `--counter-alpha` | `0.75` | Popularity decay coefficient in `[0, 1]`; larger values retain longer history. |
| `moe_expert_counter_probability_weight` | Not exposed | `0` | Weight in `[0, 1]` blending binary expert selection with routing probability when updating offload counters. |
| `moe_expert_swap_epsilon` | `--swap-epsilon` | `0.01` | Non-negative relative popularity margin required for background exchanges. Does not gate on-demand swaps. |
| `moe_max_in_flight_swaps_per_device` | `--max-in-flight-swaps-per-device` | `4` | Maximum concurrent background exchanges per CUDA device. `0` disables background swaps, not on-demand swaps. |
| `enable_moe_expert_counting` | Not exposed | `0` | Enable persistent counters without logging or changing placement. Offload maintains counters independently of this flag. |
| `enable_moe_expert_statistics` | `--counter-log` / `--no-counter-log` | `0` | Emit counter updates at INFO severity; also enables counting. The benchmark raises the logger level with `--counter-log`. |
| `moe_expert_counter_state_file` | `--state` | Unset | Load saved popularities, alpha, and epsilon instead of starting with zero counters. |
| `moe_expert_popularity_file` | `--popularity-output` | Unset | Save final popularities and counter parameters when the session is destroyed. |

The benchmark requires `--swap` or `--no-swap` to enable or disable background swaps.
`--no-swap` sets their concurrency limit to `0` without disabling on-demand swaps.
When `--state` is supplied, its alpha and epsilon replace the benchmark's corresponding CLI values.

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry("session.moe_cpu_offload_experts", "4096")
options.add_session_config_entry("session.moe_cpu_offload_strategy", "layered")  # Or adaptive/basic.
options.add_session_config_entry("session.moe_cpu_offload_gpu_ratio", "0.8")  # Layered only; (0, 1].
options.add_session_config_entry("session.moe_cpu_offload_layer_order", "first")  # Basic/layered: first/last/middle.
options.add_session_config_entry("session.moe_cpu_offload_on_demand_threshold", "2")  # Optional; 0 disables it.
options.add_session_config_entry("session.moe_expert_counter_alpha", "0.9")
options.add_session_config_entry("session.moe_expert_swap_epsilon", "0.1")
options.add_session_config_entry("session.moe_max_in_flight_swaps_per_device", "5")
session = ort.InferenceSession("model.onnx", options, providers=["CUDAExecutionProvider"])
```

For CUDA QMoE with `adaptive` or `layered`, set
`session.moe_cpu_offload_on_demand_threshold=N` (default `0`, disabled), or use the benchmark's
`--on-demand-threshold N`. If an invocation needs at most `N` distinct CPU-resident experts,
exchange them synchronously with GPU experts unused by that invocation, then compute on GPU.
This applies to prefill and decode; prefill counts the union across all tokens and tiles.
Swaps reuse cached CUDA-packed weights and the existing outgoing CPU repacking path, preserving exactly `K`.
If the selected union exceeds the layer's GPU capacity, no demand swaps are performed and CPU execution is retained.
This threshold is independent of the background swap limit; synchronous swaps also work when that limit is `0`.

Supported: FP16/BF16 MoE weights and uniform INT4/INT8 QMoE (`quant_type="int"`, `weights_prepacked=0`).
Both require FP16/BF16 activations, constant weights and connected biases, a static hidden dimension, and prepacking.
Offload does not support FC3, sparse mixer, FP32 MoE, other quantization formats, CUDA Graph capture,
plugin CUDA EP, minimal builds, or overlapping runs on one session.

Details: [operator support](../contrib_ops/cuda/moe_qmoe.md) and
[counters, saved state, and logging](../MoEExpertCounting.md).

## Qwen3.5 INT4 benchmark

**Qwen3.5-35B-A3B INT4**, 40 QMoE nodes x 256 experts = 10,240 experts, top-k 8.
Alpha **0.9**, epsilon **0.1**, probability weight **0**, swap limit **5** unless noted (ignored by basic).
At `K=10240`, all experts are on CPU, but other model operations still use CUDA.
The original basic/layered series use `layer_order=first`; the separate position sweep below compares all three orders.

All strategies use the **same H200 GPU 1**. Adaptive/basic alternate for each K using the same binaries;
layered was measured separately after fixing profiling when a boundary layer has fewer GPU experts than top-k.
The 80% and 70% layered series use identical binaries; the default remains 80%.
Each configuration uses a fresh process and the same [five prompts](benchmark-prompts.json), with 256 greedy
tokens per prompt. **Prompts 1-2 are warmup; all four metrics use prompts 3-5 only:**

- **Prefill:** mean `append_tokens()` time.
- **Decode:** 762 steady-state tokens / total decode time, excluding the prefill-logits generation call and
  each prompt's first actual decode inference.
- **Peak GPU:** device-wide NVML memory sampled every 50 ms and at prompt boundaries.
- **Peak CPU:** current process RSS sampled over the same windows, not lifetime `ru_maxrss`.

Loading and warmup peaks are excluded; allocations still resident during prompts 3-5 are included.
Published values come from the JSON `measurements` summary, not the whole-process `memory` peaks.

Measured 2026-10-16 at `358cb402b1` (adaptive/basic), `408fcd0ba2` (layered 80%), `7c50cd1982` (layered 70%)
and `bc0f54ff6b` (position sweep):
Linux, ORT 1.32.0, GenAI CUDA 0.15.2, CUDA 13.3.73, driver 580.105.08,
Xeon Platinum 8480C (96 physical cores). Release build, native CUDA architecture, Flash/MEA enabled.
These are single-run observations on a shared host, not statistical estimates or quality tests.

### Summary: decode throughput (tokens/s)

Highest observed rate per row in **bold**. The matching prefill summary follows; memory peaks are detailed below.
Percentage columns use layered placement; the swap limit is 5 unless noted.
`earlier` identifies the original cohort, kept separate from the fresh position sweep.

| K | CUDA Graph | Adaptive | Basic | 80% first (earlier) | 80% first | 80% last | 60% middle | 80% middle | 60% last, limit 16 |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 129.930 | 132.550 | 132.599 | **133.797** | 132.785 | 130.385 | 132.991 | 128.579 |
| 0 | On | **169.841** | 168.622 | 169.083 | 169.261 | 169.252 | 168.924 | 168.922 | 168.794 |
| 1024 | Off | 79.855 | 85.967 | 93.633 | **94.298** | 80.914 | 63.721 | 75.744 | 64.073 |
| 2048 | Off | 57.834 | **70.539** | 55.676 | 65.174 | 58.292 | 52.113 | 64.308 | 53.446 |
| 3072 | Off | 39.793 | 59.878 | 47.379 | 51.002 | **59.988** | 40.027 | 56.550 | 41.665 |
| 4096 | Off | 37.036 | 46.850 | 47.085 | 48.925 | 46.634 | 37.154 | **49.947** | 35.319 |
| 5120 | Off | 33.336 | 44.232 | 44.921 | 42.609 | 43.093 | 38.010 | 43.170 | **47.282** |
| 6144 | Off | 30.664 | 39.411 | 39.614 | 38.250 | **40.834** | 35.520 | 40.261 | 38.835 |
| 7168 | Off | 29.203 | 36.474 | 37.084 | 36.665 | 35.100 | **39.643** | 36.239 | 31.635 |
| 8192 | Off | 27.003 | 33.758 | 31.864 | 34.057 | 33.221 | 31.682 | 33.543 | **36.897** |
| 9216 | Off | 25.696 | 29.513 | **31.617** | 29.195 | 30.360 | 31.323 | 27.604 | 30.168 |
| 10240 | Off | 27.032 | 27.746 | **29.069** | 28.076 | 27.909 | 27.726 | 26.408 | 27.640 |

### Summary: prefill (ms)

Lowest observed time per row in **bold**, using the same configurations and cohorts as the decode summary.

| K | CUDA Graph | Adaptive | Basic | 80% first (earlier) | 80% first | 80% last | 60% middle | 80% middle | 60% last, limit 16 |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 21.134 | 20.898 | 21.309 | **20.719** | 20.737 | 20.954 | 20.797 | 21.143 |
| 0 | On | **21.391** | 21.840 | 21.683 | 21.669 | 21.774 | 21.496 | 22.871 | 21.560 |
| 1024 | Off | 33.829 | 84.129 | 30.196 | **29.604** | 33.209 | 63.106 | 42.469 | 55.736 |
| 2048 | Off | 46.457 | 146.529 | 48.208 | 43.247 | **42.243** | 115.714 | 44.856 | 63.772 |
| 3072 | Off | 152.712 | 193.798 | 115.884 | 111.006 | 105.315 | 172.761 | 107.787 | **85.188** |
| 4096 | Off | 209.657 | 249.523 | 189.019 | 191.018 | 157.959 | 208.917 | 186.702 | **101.023** |
| 5120 | Off | 297.205 | 335.351 | 214.011 | 266.120 | 194.461 | 217.876 | 272.631 | **118.835** |
| 6144 | Off | 324.400 | 340.948 | 259.104 | 375.611 | 332.403 | 228.786 | 358.748 | **195.748** |
| 7168 | Off | 326.908 | 440.696 | 364.987 | 369.580 | 439.249 | **307.013** | 374.420 | 379.688 |
| 8192 | Off | 434.233 | 577.976 | 554.616 | 521.061 | 422.721 | 452.714 | **369.965** | 440.986 |
| 9216 | Off | **418.149** | 541.466 | 624.173 | 565.593 | 602.627 | 437.221 | 508.575 | 438.548 |
| 10240 | Off | **574.325** | 633.668 | 652.343 | 676.826 | 805.693 | 604.649 | 657.629 | 624.423 |

### Adaptive

| K: CPU experts | CUDA Graph | Prefill (ms) | Decode (tokens/s) | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 21.134 | 129.930 | 39.388 | 1.576 |
| 0 | On | 21.391 | 169.841 | 39.476 | 1.646 |
| 1024 | Off | 33.829 | 79.855 | 38.219 | 7.690 |
| 2048 | Off | 46.457 | 57.834 | 36.539 | 11.266 |
| 3072 | Off | 152.712 | 39.793 | 34.730 | 14.584 |
| 4096 | Off | 209.657 | 37.036 | 32.969 | 18.159 |
| 5120 | Off | 297.205 | 33.336 | 31.154 | 21.733 |
| 6144 | Off | 324.400 | 30.664 | 29.613 | 25.308 |
| 7168 | Off | 326.908 | 29.203 | 27.930 | 28.883 |
| 8192 | Off | 434.233 | 27.003 | 26.207 | 32.456 |
| 9216 | Off | 418.149 | 25.696 | 24.476 | 36.030 |
| 10240 | Off | 574.325 | 27.032 | 22.683 | 39.538 |

### Basic

Each 1,024-expert step moves four more trailing QMoE nodes to CPU.

| K: CPU experts | CUDA Graph | Prefill (ms) | Decode (tokens/s) | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 20.898 | 132.550 | 39.392 | 1.574 |
| 0 | On | 21.840 | 168.622 | 39.449 | 1.645 |
| 1024 | Off | 84.129 | 85.967 | 37.900 | 7.424 |
| 2048 | Off | 146.529 | 70.539 | 36.205 | 11.001 |
| 3072 | Off | 193.798 | 59.878 | 34.637 | 14.579 |
| 4096 | Off | 249.523 | 46.850 | 32.820 | 18.151 |
| 5120 | Off | 335.351 | 44.232 | 31.127 | 21.727 |
| 6144 | Off | 340.948 | 39.411 | 29.431 | 25.302 |
| 7168 | Off | 440.696 | 36.474 | 27.740 | 28.781 |
| 8192 | Off | 577.976 | 33.758 | 26.076 | 32.356 |
| 9216 | Off | 541.466 | 29.513 | 24.383 | 35.915 |
| 10240 | Off | 633.668 | 27.746 | 22.660 | 39.537 |

### Layered (80% GPU target per layer)

| K: CPU experts | CUDA Graph | Prefill (ms) | Decode (tokens/s) | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 21.309 | 132.599 | 39.390 | 1.575 |
| 0 | On | 21.683 | 169.083 | 39.461 | 1.644 |
| 1024 | Off | 30.196 | 93.633 | 38.012 | 7.426 |
| 2048 | Off | 48.208 | 55.676 | 35.937 | 11.024 |
| 3072 | Off | 115.884 | 47.379 | 34.219 | 14.826 |
| 4096 | Off | 189.019 | 47.085 | 33.012 | 18.158 |
| 5120 | Off | 214.011 | 44.921 | 31.273 | 21.733 |
| 6144 | Off | 259.104 | 39.614 | 29.678 | 25.312 |
| 7168 | Off | 364.987 | 37.084 | 27.330 | 28.866 |
| 8192 | Off | 554.616 | 31.864 | 25.607 | 32.441 |
| 9216 | Off | 624.173 | 31.617 | 24.414 | 36.032 |
| 10240 | Off | 652.343 | 29.069 | 22.660 | 39.536 |

### Layered (70% GPU target per layer)

| K: CPU experts | CUDA Graph | Prefill (ms) | Decode (tokens/s) | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 20.923 | 130.596 | 39.517 | 1.577 |
| 0 | On | 21.694 | 176.596 | 39.486 | 1.646 |
| 1024 | Off | 30.333 | 91.353 | 38.012 | 7.426 |
| 2048 | Off | 36.109 | 78.171 | 36.420 | 11.035 |
| 3072 | Off | 156.260 | 39.392 | 34.799 | 14.586 |
| 4096 | Off | 204.432 | 35.276 | 32.096 | 18.129 |
| 5120 | Off | 210.089 | 50.620 | 31.486 | 21.737 |
| 6144 | Off | 274.308 | 38.230 | 29.611 | 25.308 |
| 7168 | Off | 367.447 | 34.966 | 26.945 | 28.852 |
| 8192 | Off | 444.187 | 34.234 | 26.138 | 32.457 |
| 9216 | Off | 568.924 | 30.098 | 24.465 | 36.033 |
| 10240 | Off | 717.923 | 28.232 | 21.722 | 39.507 |

### Layered: 60%/80% and first/last/middle

Separate cohort: 72 fresh runs, identical binaries and GPU, six variants interleaved per K and run serially.
Same prompt/measurement protocol and swap limit 5. No position consistently wins; bold marks each row's fastest decode.
At K=0 and K=10240, placement settings do not change execution; differences show run-to-run variability.

#### Decode (tokens/s)

| K | CUDA Graph | 60% first | 80% first | 60% last | 80% last | 60% middle | 80% middle |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 136.780 | 133.797 | **137.428** | 132.785 | 130.385 | 132.991 |
| 0 | On | **169.287** | 169.261 | 168.827 | 169.252 | 168.924 | 168.922 |
| 1024 | Off | 81.065 | **94.298** | 66.750 | 80.914 | 63.721 | 75.744 |
| 2048 | Off | **68.558** | 65.174 | 52.294 | 58.292 | 52.113 | 64.308 |
| 3072 | Off | 54.918 | 51.002 | 41.284 | **59.988** | 40.027 | 56.550 |
| 4096 | Off | 37.907 | 48.925 | 39.285 | 46.634 | 37.154 | **49.947** |
| 5120 | Off | 39.626 | 42.609 | 42.769 | 43.093 | 38.010 | **43.170** |
| 6144 | Off | 40.025 | 38.250 | 37.666 | **40.834** | 35.520 | 40.261 |
| 7168 | Off | 35.382 | 36.665 | 34.716 | 35.100 | **39.643** | 36.239 |
| 8192 | Off | 31.683 | **34.057** | 33.658 | 33.221 | 31.682 | 33.543 |
| 9216 | Off | **31.689** | 29.195 | 30.717 | 30.360 | 31.323 | 27.604 |
| 10240 | Off | 27.458 | 28.076 | **28.832** | 27.909 | 27.726 | 26.408 |

#### Prefill (ms)

| K | CUDA Graph | 60% first | 80% first | 60% last | 80% last | 60% middle | 80% middle |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 20.547 | 20.719 | 20.430 | 20.737 | 20.954 | 20.797 |
| 0 | On | 21.597 | 21.669 | 21.719 | 21.774 | 21.496 | 22.871 |
| 1024 | Off | 43.849 | 29.604 | 55.373 | 33.209 | 63.106 | 42.469 |
| 2048 | Off | 51.530 | 43.247 | 95.109 | 42.243 | 115.714 | 44.856 |
| 3072 | Off | 90.679 | 111.006 | 158.163 | 105.315 | 172.761 | 107.787 |
| 4096 | Off | 214.915 | 191.018 | 195.642 | 157.959 | 208.917 | 186.702 |
| 5120 | Off | 206.028 | 266.120 | 165.493 | 194.461 | 217.876 | 272.631 |
| 6144 | Off | 325.988 | 375.611 | 210.821 | 332.403 | 228.786 | 358.748 |
| 7168 | Off | 381.340 | 369.580 | 318.008 | 439.249 | 307.013 | 374.420 |
| 8192 | Off | 402.316 | 521.061 | 419.847 | 422.721 | 452.714 | 369.965 |
| 9216 | Off | 499.094 | 565.593 | 571.157 | 602.627 | 437.221 | 508.575 |
| 10240 | Off | 586.115 | 676.826 | 627.106 | 805.693 | 604.649 | 657.629 |

#### Peak GPU (GiB)

| K | CUDA Graph | 60% first | 80% first | 60% last | 80% last | 60% middle | 80% middle |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 39.453 | 39.390 | 39.390 | 39.390 | 39.533 | 39.455 |
| 0 | On | 38.515 | 39.449 | 39.574 | 39.508 | 39.445 | 39.510 |
| 1024 | Off | 38.035 | 38.094 | 38.156 | 38.154 | 38.111 | 38.076 |
| 2048 | Off | 36.367 | 36.437 | 36.379 | 36.500 | 36.379 | 36.449 |
| 3072 | Off | 34.793 | 34.724 | 34.672 | 34.750 | 34.791 | 34.904 |
| 4096 | Off | 33.013 | 33.056 | 33.135 | 33.060 | 33.189 | 33.060 |
| 5120 | Off | 31.416 | 31.336 | 31.521 | 31.308 | 31.478 | 31.340 |
| 6144 | Off | 29.644 | 29.556 | 29.826 | 29.615 | 29.703 | 29.736 |
| 7168 | Off | 27.976 | 27.828 | 27.959 | 27.971 | 27.912 | 28.015 |
| 8192 | Off | 26.150 | 26.170 | 26.295 | 26.238 | 26.338 | 26.121 |
| 9216 | Off | 24.412 | 24.385 | 24.592 | 24.449 | 24.420 | 24.449 |
| 10240 | Off | 22.722 | 22.722 | 22.738 | 22.660 | 22.789 | 22.722 |

#### Peak CPU RSS (GiB)

| K | CUDA Graph | 60% first | 80% first | 60% last | 80% last | 60% middle | 80% middle |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 1.576 | 1.574 | 1.574 | 1.574 | 1.577 | 1.576 |
| 0 | On | 1.616 | 1.645 | 1.649 | 1.646 | 1.644 | 1.647 |
| 1024 | Off | 7.427 | 7.429 | 7.305 | 7.434 | 7.433 | 7.431 |
| 2048 | Off | 11.004 | 11.040 | 10.878 | 11.268 | 11.007 | 11.266 |
| 3072 | Off | 14.613 | 14.841 | 14.453 | 14.842 | 14.586 | 14.847 |
| 4096 | Off | 18.159 | 18.160 | 18.163 | 18.161 | 18.164 | 18.159 |
| 5120 | Off | 21.736 | 21.734 | 21.737 | 21.734 | 21.738 | 21.734 |
| 6144 | Off | 25.307 | 25.307 | 25.312 | 25.310 | 25.309 | 25.314 |
| 7168 | Off | 28.884 | 28.882 | 28.884 | 28.886 | 28.882 | 28.887 |
| 8192 | Off | 32.456 | 32.458 | 32.461 | 32.460 | 32.461 | 32.457 |
| 9216 | Off | 36.031 | 36.031 | 36.037 | 36.032 | 36.031 | 36.033 |
| 10240 | Off | 39.539 | 39.538 | 39.540 | 39.536 | 39.541 | 39.539 |

### Layered: 60% last, swap limit 16

Twelve subsequent runs with the same binaries, GPU and protocol, changing only the swap limit from 5 to 16.
Prefill and memory columns describe the 16-limit runs; decode compares both limits in tokens/s.
There is no consistent decode gain. K=0 and K=10240 perform no swaps and provide variability controls.

| K | CUDA Graph | Prefill (ms) | Decode, limit 5 | Decode, limit 16 | Change | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 21.143 | 137.428 | 128.579 | -6.4% | 39.455 | 1.576 |
| 0 | On | 21.560 | 168.827 | 168.794 | 0.0% | 39.445 | 1.643 |
| 1024 | Off | 55.736 | 66.750 | 64.073 | -4.0% | 37.974 | 7.428 |
| 2048 | Off | 63.772 | 52.294 | 53.446 | +2.2% | 36.363 | 11.005 |
| 3072 | Off | 85.188 | 41.284 | 41.665 | +0.9% | 34.744 | 14.582 |
| 4096 | Off | 101.023 | 39.285 | 35.319 | -10.1% | 33.160 | 18.417 |
| 5120 | Off | 118.835 | 42.769 | 47.282 | +10.6% | 31.512 | 21.737 |
| 6144 | Off | 195.748 | 37.666 | 38.835 | +3.1% | 29.640 | 25.307 |
| 7168 | Off | 379.688 | 34.716 | 31.635 | -8.9% | 27.898 | 28.882 |
| 8192 | Off | 440.986 | 33.658 | 36.897 | +9.6% | 26.154 | 32.457 |
| 9216 | Off | 438.548 | 30.717 | 30.168 | -1.8% | 24.406 | 36.031 |
| 10240 | Off | 624.423 | 28.832 | 27.640 | -4.1% | 22.660 | 39.537 |

### Logged diagnostics: 60% last, swap limit 16

Ten additional runs at `1777beeef6`, same binaries and prompts, with `--counter-log`.
Below: prompts 3-5, **762 steady decode tokens**. Each token selects 8 experts in each of 40 layers:
**243,840 expert-token executions**, classified using each counter record's `device`.
These count repeated executions, not distinct experts. Prefill logs deduplicate IDs and cannot provide that token-level count.
Logged timings do not replace the unlogged benchmarks above.

| K | Swaps started | Completed/reclaimed | Swaps/token | Peak in flight | Blocked observations | CPU expert executions | CPU share |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 5157 | 5169 | 6.768 | 10 | 0 | 7325 | 3.00% |
| 2048 | 6966 | 6995 | 9.142 | 16 | 49 | 8728 | 3.58% |
| 3072 | 8707 | 8731 | 11.427 | 16 | 758 | 10625 | 4.36% |
| 4096 | 10703 | 10711 | 14.046 | 16 | 3323 | 13632 | 5.59% |
| 5120 | 5224 | 5255 | 6.856 | 16 | 154 | 41602 | 17.06% |
| 6144 | 3960 | 3989 | 5.197 | 16 | 68 | 82882 | 33.99% |
| 7168 | 3373 | 3413 | 4.427 | 16 | 20 | 121533 | 49.84% |
| 8192 | 659 | 698 | 0.865 | 11 | 0 | 159167 | 65.28% |
| 9216 | 163 | 180 | 0.214 | 7 | 0 | 201280 | 82.55% |
| 10240 | 0 | 0 | 0.000 | 0 | 0 | 243840 | 100.00% |

Completed/reclaimed = previous in-flight + new starts - current in-flight at each `Run()` boundary.
This can exceed starts within the measured windows because earlier exchanges finish there; pending tail exchanges are excluded.
In-flight includes publication, not just PCIe copying. Blocked observations can count the same candidate repeatedly.
Only mixed CPU/GPU layers can swap: at K=1024 there are 10; at K=9216 there are 7.
At K=4096, the limit of 16 is reached on 451/762 decode boundaries; 40% CPU-resident experts account for only 5.59% of executions.

## Reproduce

Put a matching source build on `PYTHONPATH` and run outside the source package directory:

```bash
export PYTHONPATH=/path/to/onnxruntime/build/linux_cuda/Release
export LD_LIBRARY_PATH="$PYTHONPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$PYTHONPATH"
for scenario in k0-nograph k0-graph k1024 k2048 k3072 k4096 k5120 k6144 k7168 k8192 k9216 k10240; do
  case "$scenario" in
    k0-nograph) k=0; graph=--no-cuda-graph ;;
    k0-graph) k=0; graph=--cuda-graph ;;
    *) k=${scenario#k}; graph=--no-cuda-graph ;;
  esac
  for strategy in adaptive basic layered; do
    output="/tmp/qwen35-int4/${strategy}"
    mkdir -p "$output"
    CUDA_VISIBLE_DEVICES=1 python /path/to/onnxruntime/tools/python/benchmark_moe_cpu_offload.py \
      --name "${strategy}-${scenario}" \
      --model /path/to/qwen3.5-35b-cuda-int4 \
      --prompts /path/to/onnxruntime/docs/moe-cpu-offload/benchmark-prompts.json \
      --max-new-tokens 256 --cpu-offload-experts "$k" --strategy "$strategy" --gpu-ratio 0.8 \
      --counter-alpha 0.9 --swap-epsilon 0.1 --max-in-flight-swaps-per-device 5 \
      --swap --no-counter-log "$graph" \
      --output "$output/${scenario}.json" --log "$output/${scenario}.log" || exit "$?"
  done
done
```

For the 70% series, use `--strategy layered --gpu-ratio 0.7` with a separate output directory.
Use `--layer-order last` or `--layer-order middle` to change GPU layer selection for basic/layered.
For the six-way position sweep, use `--strategy layered`, ratios `0.6`/`0.8`, and each layer order,
with separate output directories and serial runs.
For the 16-swap comparison, use `--strategy layered --gpu-ratio 0.6 --layer-order last`
and `--max-in-flight-swaps-per-device 16`.
Add `--counter-log` for swap and expert-device diagnostics; use separate output files.

## Takeaways for Qwen3.5-35B-A3B INT4

Working conclusions for this model on the H200. The threshold-1/2 sweep is still running.

### Strategy behavior

The user chooses the parameters before loading the model. They stay fixed during generation.

- **Adaptive:** swap experts based on popularity.
  With binary counting, `P(t) = alpha * P(t-1) + (1-alpha)` for selected experts; others only decay.
  In decode, update once per token; in prefill, count the selected-expert union once.
  A swap is a candidate if `P_hottest_CPU > (1+epsilon) * P_coldest_GPU` in the same layer.
  After each decode pass, launch at most five background swaps in these benchmarks.
  Swaps still running count against those five: if three remain, only two new swaps can start.
  Other candidates are ignored and reconsidered after the next pass.
- **Basic:** fill the first QMoE layers on CUDA, then put the rest on CPU. No swaps.
- **Layered 80%:** target 80% GPU / 20% CPU in each chosen layer until the GPU budget runs out.
  Swap experts within mixed layers using the same popularity rule as adaptive.
  All strategies preserve exactly `K` CPU experts.

### Prefill

Adaptive beats basic, but layered can also win.

### Decode

- Without on-demand swaps, adaptive is about **20% slower than basic**, averaged over the mixed CPU/GPU budgets.
- A good compromise is layered: choose GPU layers like basic, then choose their GPU experts by popularity.
- GPU layers can be first, last, or middle. No position is consistently best.

### On-demand swaps

If only one selected expert is on CPU, swap it to CUDA and compute there, provided it fits.
Swapping two is worse for decode, but can still improve prefill.
Set `moe_cpu_offload_on_demand_threshold` to `1` or `2`; `0` keeps the previous behavior.

### Next ideas

- Predict which experts will be needed next, while accounting for the cost and limited number of swaps.
- Automatically increase or reduce parameters when measured throughput improves, instead of relying only on defaults.
  This is not implemented yet.
