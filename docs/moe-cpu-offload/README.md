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
| `enable_moe_expert_counting` | Not exposed | `0` | Legacy counting flag; does not enable counters without offload. Offload maintains counters independently of this flag. |
| `enable_moe_expert_statistics` | `--counter-log` / `--no-counter-log` | `0` | Emit offload counter updates at INFO severity. Disabled without offload. The benchmark raises the logger level with `--counter-log`. |
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

To keep early QMoE layers entirely on CPU and later layers partially on GPU, use `layered` with
`layer_order=last` and choose `K` large enough that the later layers consume all available GPU slots.
For example, with 40 layers of 256 experts, `K=6144` and `gpu_ratio=0.8` leave the first 19 layers entirely on CPU,
the last 20 layers with 204 GPU experts each, and the boundary layer with 16 GPU experts.
Within each layer's fixed GPU capacity, experts are ranked by popularity.
Background swaps remain at the end of Run, subject to epsilon and swap limits.

Supported: FP16/BF16 MoE weights and uniform INT4/INT8 QMoE (`quant_type="int"`, `weights_prepacked=0`).
Both require FP16/BF16 activations, constant weights and connected biases, a static hidden dimension, and prepacking.
Offload does not support FC3, sparse mixer, FP32 MoE, other quantization formats, CUDA Graph capture,
plugin CUDA EP, minimal builds, or overlapping runs on one session.

For operator support, see [MoE/QMoE](../contrib_ops/cuda/moe_qmoe.md).

## Counters

Only offload-eligible built-in CUDA nodes maintain counters, including their CPU-resident experts.
CPU EP nodes, unsupported nodes, and sessions without offload do not participate.
Each successful invocation updates popularity `P`; selected experts contribute once, using the union across tokens
and tiles, including prefill. With probability weight `lambda`, their contribution is
`(1-lambda) + lambda * max_routing_probability`; unselected experts only decay.
Counters start at zero and persist across runs. Updates from successful kernels are not rolled back if a later kernel fails.

GPU usage completion stays deferred until the existing end-of-Run boundary (or the next invocation of the same
kernel when its snapshot must be reused). After collecting usage, the end-of-Run scheduler ranks experts by
popularity for the next invocation. No runtime correlation matrices or predictive placement are maintained.

For a separate mathematical study of exponential averaging and delayed state selection in a fixed Markov chain,
see [Choosing alpha to predict the states of a Markov chain](markov-state-selection.md).

### Saved state and diagnostics

`--popularity-output` saves `P`, alpha, and epsilon at session destruction; `--state` restores them.
Use the same model and graph-optimization settings. Omitted experts start at zero; invalid records fail initialization.
The JSON format is:

```json
{"alpha": 0.9, "epsilon": 0.1, "experts": [{"moe": 0, "expert": 0, "popularity": 0.42}]}
```

`moe` is a zero-based ordinal sorted by graph scope and resolved node index, **not** the topological QMoE numbering
used by the analysis scripts; `expert` is local to that node.
`--counter-log` enables INFO records: `moe_expert_counters`, `moe_expert_swaps`, and `moe_expert_demand_swaps`.
The 1,024 counter-record limit per run emits `moe_expert_counters_truncated` on overflow; updates continue,
but analyzers reject truncated traces. Logging adds overhead; keep it off for performance measurements.

Reuse [qmoe_prompt_runner.py](../../tools/python/qmoe_prompt_runner.py) for phase-tagged traces,
[qmoe_expert_distribution.py](../../tools/python/qmoe_expert_distribution.py) for timelines,
[qmoe_expert_activation_correlations.py](../../tools/python/qmoe_expert_activation_correlations.py) for correlations,
and [optimize_qmoe_expert_counters.py](../../tools/python/optimize_qmoe_expert_counters.py) for counter replay.
Analysis requires the runner's prefill/decode markers: timelines and correlations exclude prefill;
replay applies its updates but scores only decode.

To log the five standard prompts and compare every QMoE with the **first** QMoE at the same token,
itself at the previous token, and the **last** QMoE at the previous token, reuse the existing scripts:

```bash
python tools/python/qmoe_prompt_runner.py /path/to/qwen3.5-35b-cuda-int4 \
  --prompts-file docs/moe-cpu-offload/benchmark-prompts.json --max-new-tokens 256 \
  --moe-cpu-offload-experts 1 --moe-expert-counter-alpha 0.9 --moe-max-in-flight-swaps-per-device 0 \
  --output /tmp/qmoe-prompts.json --counter-log /tmp/qmoe-counters.log
python tools/python/qmoe_expert_activation_correlations.py /tmp/qmoe-counters.log \
  --reference-layer first --include-last-layer-previous-token --output-prefix /tmp/first-last-layer-correlations
```

One CPU expert is required to enable collection; swaps are disabled. Generation can stop on EOS before 256 tokens.
`--reference-layer first` computes binary Pearson correlations offline; it does not affect runtime placement.
The default remains `previous` for adjacent-layer analysis. Temporal pairs never cross prompt boundaries.
In the NPZ archive, `first_qmoe_same_token[q,j,i]` is `corr(S[t,m,j], S[t,0,i])`, with `m` identified by
`first_qmoe_current_node_names[q]`; `same_qmoe_previous_token[m,i,j]` correlates the current and previous token
in the same QMoE. Constant expert activations yield undefined (`NaN`) correlations.
`--include-last-layer-previous-token` adds `last_qmoe_previous_token[m,i,j]`, correlating `S[t,m,i]` with
`S[t-1,M-1,j]`, where `M-1` is the last QMoE in the same graph scope. Its heatmap and third series are included
in the PDF; node names and activation counts are retained in the NPZ. For the last QMoE itself, this relation
equals its own previous-token correlation.
The PDF shows all matrices; the summary CSV lists supported extreme pairs, and the selected-experts CSV retains
all decode selections.

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

- **Prefill:** 84 input tokens (25, 28, and 31 for prompts 3-5) / total `append_tokens()` time.
  For these same prompts, convert the JSON's mean `prefill_ms` to token/s as `28000 / prefill_ms`,
  not `1000 / prefill_ms`.
- **Decode:** 762 steady-state tokens / total decode time, excluding the prefill-logits generation call and
  each prompt's first actual decode inference.
- **Peak GPU:** device-wide NVML memory sampled every 50 ms and at prompt boundaries.
- **Peak CPU:** current process RSS sampled over the same windows, not lifetime `ru_maxrss`.

Loading and warmup peaks are excluded; allocations still resident during prompts 3-5 are included.
Published values come from the JSON `measurements` summary, with prefill converted as above,
not the whole-process `memory` peaks. Prefill and decode are both in **token/s; higher is better**.
Memory stays in GiB.
For CPU-only runs, use `--provider cpu --cpu-offload-experts 0 --no-cuda-graph`.
These runs sample process RSS without requiring `CUDA_VISIBLE_DEVICES` or NVML; GPU memory fields are JSON `null`.

Measured 2026-10-16 at `358cb402b1` (adaptive/basic), `408fcd0ba2` (layered 80%), `7c50cd1982` (layered 70%)
and `bc0f54ff6b` (position sweep):
Linux, ORT 1.32.0, GenAI CUDA 0.15.2, CUDA 13.3.73, driver 580.105.08,
Xeon Platinum 8480C (96 physical cores). Release build, native CUDA architecture, Flash/MEA enabled.
These are single-run observations on a shared host, not statistical estimates or quality tests.

### Beta comparison: 60% last, more than half of experts on CPU

**Historical experiment:** runtime beta and correlation-based placement have been removed.
The measurements and provenance below are retained for reference.

Measured **2026-10-19** using the existing `tools/python/benchmark_moe_cpu_offload.py`, with the earlier first-QMoE predictor
on top of `314bc93103` (uncommitted working-tree changes). Both series used the same freshly rebuilt runtime and CUDA
provider on **H200 GPU 1**, with `layered`, `gpu_ratio=0.6`, `layer_order=last`, alpha `0.9`, epsilon `0.1`,
and swap limit **5**. On-demand swaps, counter logging, and CUDA Graph were disabled.
Each K used a fresh process for beta **1**, then another for beta **0.5**, running serially.
All five existing prompts generated 256 tokens; **all four metrics below use only prompts 3-5** after prompts 1-2
warm up each process. Decode counts 762 steady-state tokens per run.

The per-layer GPU target is `floor(0.6 * 256) = 153` experts, or **59.77%**. For K=6144/7168/8192/9216,
the first 13/19/26/33 layers are entirely CPU-resident, the next boundary layer has 118/12/59/106 GPU experts,
and the remaining last layers have 153 GPU experts each. **K=10240 is an all-CPU-expert control**:
other model operations still run on CUDA, but no expert can benefit from predictive GPU placement.

| K: CPU experts | CPU share | Decode beta=1 (token/s) | Decode beta=0.5 (token/s) | Decode change | Prefill beta=1 (token/s) | Prefill beta=0.5 (token/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 6144 | 60% | 38.359 | 32.935 | -14.14% | 125.367 | 100.143 |
| 7168 | 70% | 33.509 | 28.513 | -14.91% | 99.192 | 78.386 |
| 8192 | 80% | 34.420 | 32.679 | -5.06% | 67.923 | 66.849 |
| 9216 | 90% | 32.486 | 29.219 | -10.06% | 57.950 | 51.254 |
| 10240 | 100% | 27.630 | 25.963 | -6.03% | 43.236 | 41.376 |

| K: CPU experts | Peak GPU beta=1 (GiB) | Peak GPU beta=0.5 (GiB) | Peak CPU RSS beta=1 (GiB) | Peak CPU RSS beta=0.5 (GiB) |
|---:|---:|---:|---:|---:|
| 6144 | 29.638 | 29.607 | 25.308 | 25.326 |
| 7168 | 28.039 | 27.992 | 28.886 | 28.906 |
| 8192 | 26.213 | 26.275 | 32.459 | 32.479 |
| 9216 | 24.467 | 24.527 | 36.033 | 36.054 |
| 10240 | 22.658 | 21.658 | 39.537 | 39.525 |

**In this sweep, beta=0.5 did not improve throughput:** decode throughput was 5.1%-14.9% lower and prefill throughput
was 1.6%-21.0% lower. That implementation eagerly completed usage after every QMoE, including a host wait for the
CUDA counter-copy completion event in addition to the CPU-expert wait already required by offload.
This reduced host/GPU overlap and started prediction after the first QMoE's computation,
not immediately when routing is available. These results include that synchronization, correlation updates, and
predictive scheduling; they do not isolate the scoring formula. The all-CPU-expert row cannot show a placement benefit
and also retains those opt-in costs.
Its 39 correlation matrices occupied 19.5 MiB; measured RSS includes other allocations as well.
These measurements do not evaluate the later same-QMoE predictor, which removed eager per-layer completion
and has also been removed.
GPU memory is device-wide NVML memory sampled every 50 ms. No external GPU compute processes were present at
the start of any run, and runtime/provider hashes were unchanged across the sweep, but continuous isolation was not
enforced. These are one run per configuration on a shared host, not statistical estimates or quality comparisons.
The 1 GiB lower GPU peak in the all-CPU-expert beta=0.5 run is an observed allocation difference, not evidence of
expert-placement savings.

[Exact measurements and binary/source provenance](beta-comparison-20261019.json) are retained separately from the
older cohorts below.

### GPU ratio and beta sweep: 33 of 45 configurations completed

Measured **2026-10-19**, after the paired comparison, with the same earlier first-QMoE runtime/provider binaries
and benchmark script, not the later, now-removed temporal predictor.
The sweep was stopped on request after **33 completed configurations**; it was not resumed.
Placement is `layered/last`, swap limit **5**, beta **1/0.5/0.1**, with the same five prompts and measurement windows.
All GPU-1 compute-process checks before runs were empty; continuous isolation was not enforced.

Beta columns show **prefill / decode in token/s**. The final column compares against the **best non-layered strategy
at the same K**: historical Adaptive, Basic, and the on-demand Adaptive variants. Basic has the highest decode
throughput among those strategies at all four measured budgets. These baselines were not rerun.
The partial-sweep results remain separate from the paired-comparison columns in the Summary below.

For reference, the original Adaptive and Basic baselines are also shown explicitly, in **prefill / decode token/s**:

| K: CPU experts | Adaptive | Basic |
|---:|---:|---:|
| 6144 | 86.313 / 30.664 | 82.124 / 39.411 |
| 7168 | 85.651 / 29.203 | 63.536 / 36.474 |
| 8192 | 64.482 / 27.003 | 48.445 / 33.758 |
| 9216 | 66.962 / 25.696 | 51.711 / 29.513 |

| K: CPU experts | GPU target per layer | beta=1 | beta=0.5 | beta=0.1 | Best non-layered decode (token/s) and strategy |
|---:|---:|---:|---:|---:|---|
| 6144 | 60% | 124.715 / 35.123 | 112.586 / 34.558 | 114.817 / 31.606 | **39.411** - Basic |
| 6144 | 70% | 131.807 / 42.161 | 112.304 / 31.364 | 115.604 / 31.838 | **39.411** - Basic |
| 6144 | 80% | 78.763 / 40.208 | 87.111 / 36.023 | 103.450 / 35.843 | **39.411** - Basic |
| 7168 | 60% | 94.385 / 34.842 | 74.305 / 30.641 | 63.228 / 28.782 | **36.474** - Basic |
| 7168 | 70% | 76.499 / 35.994 | 80.104 / 32.001 | 70.721 / 32.586 | **36.474** - Basic |
| 7168 | 80% | 67.606 / 36.534 | 68.902 / 33.626 | 67.243 / 34.718 | **36.474** - Basic |
| 8192 | 60% | 63.793 / 36.306 | 65.925 / 30.521 | 61.086 / 31.280 | **33.758** - Basic |
| 8192 | 70% | 63.549 / 35.796 | 82.178 / 32.407 | 59.812 / 30.455 | **33.758** - Basic |
| 8192 | 80% | 59.330 / 33.014 | 59.441 / 30.601 | 61.887 / 28.459 | **33.758** - Basic |
| 9216 | 60% | 47.917 / 31.595 | 59.584 / 29.329 | 52.812 / 27.619 | **29.513** - Basic |
| 9216 | 70% | 52.532 / 30.591 | 50.401 / 27.063 | 50.788 / 26.934 | **29.513** - Basic |
| 9216 | 80% | n/a | n/a | n/a | **29.513** - Basic |

`n/a` means not measured. No K=10240 configuration completed in this sweep.
Beta=1 gave the highest decode throughput in all **11 measured ratio/budget groups**.
Beta<1 sometimes improved prefill, but retains the eager-completion overhead described above.
These are single-run observations, not paired reruns or a statistically established ranking against the historical
baselines. The all-strategy best values, including Layered, remain in the final Tables section.
[Exact partial results, memory peaks, provenance, and missing configurations](ratio-beta-comparison-20261019.json).

### Summary: decode throughput (token/s)

Highest observed rate per row in **bold**. The matching prefill summary follows; memory peaks are detailed below.
Percentage columns use layered placement; the swap limit is 5 unless noted.
`earlier` identifies the original cohort, kept separate from the fresh position sweep.
`N=1`/`N=2` are on-demand thresholds. **Those series have possible GPU interference;
see the [sweep warning](#on-demand-threshold-sweep).** Their K=0 columns share two fresh all-GPU controls
with the threshold disabled. `new` identifies the 2026-10-19 beta series, kept separate from historical cohorts.
`n/a` means the configuration was not measured: K=10240 for the on-demand series, K<6144 for the beta series.

| K | CUDA Graph | Adaptive | Basic | 80% first (earlier) | 80% first | 80% last | 60% middle | 80% middle | 60% last, limit 16 | Adaptive N=1 | Adaptive N=2 | 80% first N=1 | 80% first N=2 | 60% last beta=1 (new) | 60% last beta=0.5 (new) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 129.930 | 132.550 | 132.599 | **133.797** | 132.785 | 130.385 | 132.991 | 128.579 | 127.829 | 127.829 | 127.829 | 127.829 | n/a | n/a |
| 0 | On | **169.841** | 168.622 | 169.083 | 169.261 | 169.252 | 168.924 | 168.922 | 168.794 | 168.862 | 168.862 | 168.862 | 168.862 | n/a | n/a |
| 1024 | Off | 79.855 | 85.967 | 93.633 | 94.298 | 80.914 | 63.721 | 75.744 | 64.073 | 92.155 | 87.180 | **105.088** | 96.845 | n/a | n/a |
| 2048 | Off | 57.834 | 70.539 | 55.676 | 65.174 | 58.292 | 52.113 | 64.308 | 53.446 | 68.595 | **74.055** | 73.977 | 65.891 | n/a | n/a |
| 3072 | Off | 39.793 | 59.878 | 47.379 | 51.002 | **59.988** | 40.027 | 56.550 | 41.665 | 54.189 | 49.423 | 54.272 | 56.441 | n/a | n/a |
| 4096 | Off | 37.036 | 46.850 | 47.085 | 48.925 | 46.634 | 37.154 | 49.947 | 35.319 | 44.061 | 37.448 | **50.377** | 45.613 | n/a | n/a |
| 5120 | Off | 33.336 | 44.232 | 44.921 | 42.609 | 43.093 | 38.010 | 43.170 | **47.282** | 34.105 | 25.579 | 44.327 | 42.273 | n/a | n/a |
| 6144 | Off | 30.664 | 39.411 | 39.614 | 38.250 | **40.834** | 35.520 | 40.261 | 38.835 | 28.010 | 20.759 | 38.657 | 38.375 | 38.359 | 32.935 |
| 7168 | Off | 29.203 | 36.474 | 37.084 | 36.665 | 35.100 | **39.643** | 36.239 | 31.635 | 27.997 | 17.475 | 36.679 | 34.063 | 33.509 | 28.513 |
| 8192 | Off | 27.003 | 33.758 | 31.864 | 34.057 | 33.221 | 31.682 | 33.543 | **36.897** | 27.923 | 23.315 | 33.569 | 15.156 | 34.420 | 32.679 |
| 9216 | Off | 25.696 | 29.513 | 31.617 | 29.195 | 30.360 | 31.323 | 27.604 | 30.168 | 25.263 | 25.418 | 4.866 | 29.398 | **32.486** | 29.219 |
| 10240 | Off | 27.032 | 27.746 | **29.069** | 28.076 | 27.909 | 27.726 | 26.408 | 27.640 | n/a | n/a | n/a | n/a | 27.630 | 25.963 |

### Summary: prefill throughput (token/s)

Highest observed rate per row in **bold**, using the same configurations and cohorts as the decode summary.
The same [GPU-interference warning](#on-demand-threshold-sweep) applies to the N=1/N=2 columns.

| K | CUDA Graph | Adaptive | Basic | 80% first (earlier) | 80% first | 80% last | 60% middle | 80% middle | 60% last, limit 16 | Adaptive N=1 | Adaptive N=2 | 80% first N=1 | 80% first N=2 | 60% last beta=1 (new) | 60% last beta=0.5 (new) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 1324.879 | 1339.841 | 1313.999 | **1351.417** | 1350.244 | 1336.260 | 1346.348 | 1324.315 | 1328.399 | 1328.399 | 1328.399 | 1328.399 | n/a | n/a |
| 0 | On | 1308.962 | 1282.051 | 1291.334 | 1292.169 | 1285.937 | 1302.568 | 1224.258 | 1298.701 | **1312.644** | **1312.644** | **1312.644** | **1312.644** | n/a | n/a |
| 1024 | Off | 827.692 | 332.822 | 927.275 | **945.818** | 843.145 | 443.698 | 659.304 | 502.368 | 711.852 | 520.494 | 934.112 | 775.752 | n/a | n/a |
| 2048 | Off | 602.708 | 191.088 | 580.816 | 647.444 | **662.832** | 241.976 | 624.220 | 439.064 | 449.662 | 480.291 | 570.800 | 281.798 | n/a | n/a |
| 3072 | Off | 183.352 | 144.480 | 241.621 | 252.239 | 265.869 | 162.074 | 259.772 | 328.685 | **479.888** | 312.993 | 257.391 | 277.200 | n/a | n/a |
| 4096 | Off | 133.551 | 112.214 | 148.133 | 146.583 | 177.261 | 134.025 | 149.972 | 277.165 | **294.650** | 246.000 | 137.493 | 114.252 | n/a | n/a |
| 5120 | Off | 94.211 | 83.495 | 130.834 | 105.216 | 143.988 | 128.513 | 102.703 | **235.621** | 178.868 | 172.314 | 103.969 | 113.048 | n/a | n/a |
| 6144 | Off | 86.313 | 82.124 | 108.065 | 74.545 | 84.235 | 122.385 | 78.049 | **143.041** | 103.608 | 127.565 | 81.583 | 66.066 | 125.367 | 100.143 |
| 7168 | Off | 85.651 | 63.536 | 76.715 | 75.762 | 63.745 | 91.201 | 74.782 | 73.745 | 75.703 | 83.558 | 67.989 | 65.239 | **99.192** | 78.386 |
| 8192 | Off | 64.482 | 48.445 | 50.485 | 53.737 | 66.238 | 61.849 | **75.683** | 63.494 | 72.493 | 73.435 | 62.090 | 39.938 | 67.923 | 66.849 |
| 9216 | Off | 66.962 | 51.711 | 44.859 | 49.506 | 46.463 | 64.041 | 55.056 | 63.847 | 68.219 | **71.103** | 31.137 | 43.009 | 57.950 | 51.254 |
| 10240 | Off | **48.753** | 44.187 | 42.922 | 41.370 | 34.753 | 46.308 | 42.577 | 44.841 | n/a | n/a | n/a | n/a | 43.236 | 41.376 |

### Adaptive

| K: CPU experts | CUDA Graph | Prefill (token/s) | Decode (token/s) | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 1324.879 | 129.930 | 39.388 | 1.576 |
| 0 | On | 1308.962 | 169.841 | 39.476 | 1.646 |
| 1024 | Off | 827.692 | 79.855 | 38.219 | 7.690 |
| 2048 | Off | 602.708 | 57.834 | 36.539 | 11.266 |
| 3072 | Off | 183.352 | 39.793 | 34.730 | 14.584 |
| 4096 | Off | 133.551 | 37.036 | 32.969 | 18.159 |
| 5120 | Off | 94.211 | 33.336 | 31.154 | 21.733 |
| 6144 | Off | 86.313 | 30.664 | 29.613 | 25.308 |
| 7168 | Off | 85.651 | 29.203 | 27.930 | 28.883 |
| 8192 | Off | 64.482 | 27.003 | 26.207 | 32.456 |
| 9216 | Off | 66.962 | 25.696 | 24.476 | 36.030 |
| 10240 | Off | 48.753 | 27.032 | 22.683 | 39.538 |

### Basic

Each 1,024-expert step moves four more trailing QMoE nodes to CPU.

| K: CPU experts | CUDA Graph | Prefill (token/s) | Decode (token/s) | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 1339.841 | 132.550 | 39.392 | 1.574 |
| 0 | On | 1282.051 | 168.622 | 39.449 | 1.645 |
| 1024 | Off | 332.822 | 85.967 | 37.900 | 7.424 |
| 2048 | Off | 191.088 | 70.539 | 36.205 | 11.001 |
| 3072 | Off | 144.480 | 59.878 | 34.637 | 14.579 |
| 4096 | Off | 112.214 | 46.850 | 32.820 | 18.151 |
| 5120 | Off | 83.495 | 44.232 | 31.127 | 21.727 |
| 6144 | Off | 82.124 | 39.411 | 29.431 | 25.302 |
| 7168 | Off | 63.536 | 36.474 | 27.740 | 28.781 |
| 8192 | Off | 48.445 | 33.758 | 26.076 | 32.356 |
| 9216 | Off | 51.711 | 29.513 | 24.383 | 35.915 |
| 10240 | Off | 44.187 | 27.746 | 22.660 | 39.537 |

### Layered (80% GPU target per layer)

| K: CPU experts | CUDA Graph | Prefill (token/s) | Decode (token/s) | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 1313.999 | 132.599 | 39.390 | 1.575 |
| 0 | On | 1291.334 | 169.083 | 39.461 | 1.644 |
| 1024 | Off | 927.275 | 93.633 | 38.012 | 7.426 |
| 2048 | Off | 580.816 | 55.676 | 35.937 | 11.024 |
| 3072 | Off | 241.621 | 47.379 | 34.219 | 14.826 |
| 4096 | Off | 148.133 | 47.085 | 33.012 | 18.158 |
| 5120 | Off | 130.834 | 44.921 | 31.273 | 21.733 |
| 6144 | Off | 108.065 | 39.614 | 29.678 | 25.312 |
| 7168 | Off | 76.715 | 37.084 | 27.330 | 28.866 |
| 8192 | Off | 50.485 | 31.864 | 25.607 | 32.441 |
| 9216 | Off | 44.859 | 31.617 | 24.414 | 36.032 |
| 10240 | Off | 42.922 | 29.069 | 22.660 | 39.536 |

### Layered (70% GPU target per layer)

| K: CPU experts | CUDA Graph | Prefill (token/s) | Decode (token/s) | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 1338.240 | 130.596 | 39.517 | 1.577 |
| 0 | On | 1290.679 | 176.596 | 39.486 | 1.646 |
| 1024 | Off | 923.087 | 91.353 | 38.012 | 7.426 |
| 2048 | Off | 775.430 | 78.171 | 36.420 | 11.035 |
| 3072 | Off | 179.189 | 39.392 | 34.799 | 14.586 |
| 4096 | Off | 136.965 | 35.276 | 32.096 | 18.129 |
| 5120 | Off | 133.277 | 50.620 | 31.486 | 21.737 |
| 6144 | Off | 102.075 | 38.230 | 29.611 | 25.308 |
| 7168 | Off | 76.201 | 34.966 | 26.945 | 28.852 |
| 8192 | Off | 63.037 | 34.234 | 26.138 | 32.457 |
| 9216 | Off | 49.216 | 30.098 | 24.465 | 36.033 |
| 10240 | Off | 39.001 | 28.232 | 21.722 | 39.507 |

### Layered: 60%/80% and first/last/middle

Separate cohort: 72 fresh runs, identical binaries and GPU, six variants interleaved per K and run serially.
Same prompt/measurement protocol and swap limit 5. No position consistently wins; bold marks each row's fastest decode.
At K=0 and K=10240, placement settings do not change execution; differences show run-to-run variability.

#### Decode (token/s)

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

#### Prefill (token/s)

| K | CUDA Graph | 60% first | 80% first | 60% last | 80% last | 60% middle | 80% middle |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 1362.729 | 1351.417 | 1370.534 | 1350.244 | 1336.260 | 1346.348 |
| 0 | On | 1296.476 | 1292.169 | 1289.194 | 1285.937 | 1302.568 | 1224.258 |
| 1024 | Off | 638.555 | 945.818 | 505.662 | 843.145 | 443.698 | 659.304 |
| 2048 | Off | 543.373 | 647.444 | 294.399 | 662.832 | 241.976 | 624.220 |
| 3072 | Off | 308.782 | 252.239 | 177.033 | 265.869 | 162.074 | 259.772 |
| 4096 | Off | 130.284 | 146.583 | 143.119 | 177.261 | 134.025 | 149.972 |
| 5120 | Off | 135.904 | 105.216 | 169.191 | 143.988 | 128.513 | 102.703 |
| 6144 | Off | 85.893 | 74.545 | 132.814 | 84.235 | 122.385 | 78.049 |
| 7168 | Off | 73.425 | 75.762 | 88.048 | 63.745 | 91.201 | 74.782 |
| 8192 | Off | 69.597 | 53.737 | 66.691 | 66.238 | 61.849 | 75.683 |
| 9216 | Off | 56.102 | 49.506 | 49.023 | 46.463 | 64.041 | 55.056 |
| 10240 | Off | 47.772 | 41.370 | 44.650 | 34.753 | 46.308 | 42.577 |

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
Prefill and memory columns describe the 16-limit runs; decode compares both limits in token/s.
There is no consistent decode gain. K=0 and K=10240 perform no swaps and provide variability controls.

| K | CUDA Graph | Prefill (token/s) | Decode, limit 5 | Decode, limit 16 | Change | Peak GPU (GiB) | Peak CPU RSS (GiB) |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 1324.315 | 137.428 | 128.579 | -6.4% | 39.455 | 1.576 |
| 0 | On | 1298.701 | 168.827 | 168.794 | 0.0% | 39.445 | 1.643 |
| 1024 | Off | 502.368 | 66.750 | 64.073 | -4.0% | 37.974 | 7.428 |
| 2048 | Off | 439.064 | 52.294 | 53.446 | +2.2% | 36.363 | 11.005 |
| 3072 | Off | 328.685 | 41.284 | 41.665 | +0.9% | 34.744 | 14.582 |
| 4096 | Off | 277.165 | 39.285 | 35.319 | -10.1% | 33.160 | 18.417 |
| 5120 | Off | 235.621 | 42.769 | 47.282 | +10.6% | 31.512 | 21.737 |
| 6144 | Off | 143.041 | 37.666 | 38.835 | +3.1% | 29.640 | 25.307 |
| 7168 | Off | 73.745 | 34.716 | 31.635 | -8.9% | 27.898 | 28.882 |
| 8192 | Off | 63.494 | 33.658 | 36.897 | +9.6% | 26.154 | 32.457 |
| 9216 | Off | 63.847 | 30.717 | 30.168 | -1.8% | 24.406 | 36.031 |
| 10240 | Off | 44.841 | 28.832 | 27.640 | -4.1% | 22.660 | 39.537 |

### On-demand threshold sweep

38 runs at `9ac01d912c`: thresholds N=1/2, adaptive and layered 80% first, K=1024..9216,
plus two shared K=0 controls with the threshold disabled. K=10240 was cancelled.
Same five-prompt protocol, alpha 0.9, epsilon 0.1, background swap limit 5, and no counter logging.
The four variants were interleaved per K and run serially on the same H200 with unchanged binaries.

**Warning: these measurements are retained as observed, despite possible concurrent GPU use.**
Nine measured windows show device-wide GPU peaks of 106-140 GiB instead of the expected 25-39 GiB.
Their timings and memory peaks may include interference; they are not controlled evidence of a threshold's effect.
The absence of an abnormal memory peak does not prove that a run was interference-free.

| Configuration | K with abnormal measured GPU peaks |
|---|---|
| Adaptive N=1 | 6144 |
| Adaptive N=2 | 1024, 6144, 7168 |
| 80% first N=1 | 3072, 6144, 9216 |
| 80% first N=2 | 7168, 8192 |

#### Decode (token/s)

| K | CUDA Graph | Adaptive N=1 | Adaptive N=2 | 80% first N=1 | 80% first N=2 |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | **127.829** | **127.829** | **127.829** | **127.829** |
| 0 | On | **168.862** | **168.862** | **168.862** | **168.862** |
| 1024 | Off | 92.155 | 87.180 | **105.088** | 96.845 |
| 2048 | Off | 68.595 | **74.055** | 73.977 | 65.891 |
| 3072 | Off | 54.189 | 49.423 | 54.272 | **56.441** |
| 4096 | Off | 44.061 | 37.448 | **50.377** | 45.613 |
| 5120 | Off | 34.105 | 25.579 | **44.327** | 42.273 |
| 6144 | Off | 28.010 | 20.759 | **38.657** | 38.375 |
| 7168 | Off | 27.997 | 17.475 | **36.679** | 34.063 |
| 8192 | Off | 27.923 | 23.315 | **33.569** | 15.156 |
| 9216 | Off | 25.263 | 25.418 | 4.866 | **29.398** |

#### Prefill (token/s)

| K | CUDA Graph | Adaptive N=1 | Adaptive N=2 | 80% first N=1 | 80% first N=2 |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 1328.399 | 1328.399 | 1328.399 | 1328.399 |
| 0 | On | 1312.644 | 1312.644 | 1312.644 | 1312.644 |
| 1024 | Off | 711.852 | 520.494 | 934.112 | 775.752 |
| 2048 | Off | 449.662 | 480.291 | 570.800 | 281.798 |
| 3072 | Off | 479.888 | 312.993 | 257.391 | 277.200 |
| 4096 | Off | 294.650 | 246.000 | 137.493 | 114.252 |
| 5120 | Off | 178.868 | 172.314 | 103.969 | 113.048 |
| 6144 | Off | 103.608 | 127.565 | 81.583 | 66.066 |
| 7168 | Off | 75.703 | 83.558 | 67.989 | 65.239 |
| 8192 | Off | 72.493 | 73.435 | 62.090 | 39.938 |
| 9216 | Off | 68.219 | 71.103 | 31.137 | 43.009 |

#### Peak GPU (GiB)

These are device-wide peaks, including any other GPU users, not the benchmark process's allocation alone.

| K | CUDA Graph | Adaptive N=1 | Adaptive N=2 | 80% first N=1 | 80% first N=2 |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 39.392 | 39.392 | 39.392 | 39.392 |
| 0 | On | 39.570 | 39.570 | 39.570 | 39.570 |
| 1024 | Off | 38.217 | 136.660 | 38.076 | 38.074 |
| 2048 | Off | 36.529 | 36.510 | 36.629 | 36.629 |
| 3072 | Off | 34.834 | 34.912 | 135.613 | 34.748 |
| 4096 | Off | 33.101 | 33.103 | 32.996 | 33.189 |
| 5120 | Off | 31.297 | 31.340 | 31.310 | 31.338 |
| 6144 | Off | 106.804 | 108.671 | 106.550 | 29.617 |
| 7168 | Off | 27.967 | 139.951 | 28.021 | 106.279 |
| 8192 | Off | 26.293 | 26.346 | 26.107 | 129.390 |
| 9216 | Off | 24.547 | 24.553 | 110.966 | 24.578 |

#### Peak CPU RSS (GiB)

| K | CUDA Graph | Adaptive N=1 | Adaptive N=2 | 80% first N=1 | 80% first N=2 |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 1.573 | 1.573 | 1.573 | 1.573 |
| 0 | On | 1.648 | 1.648 | 1.648 | 1.648 |
| 1024 | Off | 7.690 | 7.691 | 7.428 | 7.428 |
| 2048 | Off | 11.266 | 11.265 | 11.045 | 11.045 |
| 3072 | Off | 14.840 | 14.842 | 14.841 | 14.842 |
| 4096 | Off | 18.416 | 18.417 | 18.158 | 18.163 |
| 5120 | Off | 21.991 | 21.991 | 21.733 | 21.735 |
| 6144 | Off | 25.567 | 25.565 | 25.306 | 25.310 |
| 7168 | Off | 28.882 | 29.139 | 28.888 | 28.888 |
| 8192 | Off | 32.458 | 32.457 | 32.456 | 32.462 |
| 9216 | Off | 36.032 | 36.033 | 36.033 | 36.036 |

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
Reproducing the historical beta sweeps requires their recorded source; the current runtime and benchmark
no longer expose beta or correlation-based placement.

For the on-demand sweep, use `--on-demand-threshold 1` or `2` with `--strategy adaptive`,
or `--strategy layered --gpu-ratio 0.8 --layer-order first`, for K=1024..9216.
Add `--counter-log` for swap and expert-device diagnostics; use separate output files.

## Takeaways for Qwen3.5-35B-A3B INT4

Working conclusions for this model on the H200; see the [on-demand sweep warning](#on-demand-threshold-sweep).

### Strategies

Many parameters can be changed by the user, but they do not change during generation.

- **Adaptive**
  - Experts are swapped based on popularity.
  - In decode, `P(t) = alpha * P(t-1) + (1-alpha)` if the expert is chosen for token `t`;
    otherwise, `P(t) = alpha * P(t-1)`.
  - If `P_hottest_CPU > (1+epsilon) * P_coldest_GPU` in the same layer, the swap becomes a candidate.
    At most five background swaps are started per token in these benchmarks; the rest are ignored for that pass.
- **Basic**
  - Experts are placed on CUDA for the first QMoE layers, then on CPU when the GPU budget is exhausted.
- **Layered 80%**
  - First layer: target 80% of experts on CUDA, 20% on CPU.
  - Second layer: the same, unless the GPU budget is exhausted.
  - Continue layer by layer.
  - Swap experts in the same way as adaptive.

All strategies preserve exactly `K` CPU experts.

### Findings

- **Prefill**
  - Adaptive beats basic, but layered can also win.
- **Decode**
  - Without on-demand swaps, adaptive is about **20% slower than basic**, averaged over the mixed CPU/GPU budgets.
  - A good compromise is layered: choose GPU layers like basic, then choose their GPU experts by popularity.
  - GPU layers can be first, last, or middle. No position or GPU fraction is consistently best:
    at K=6144, 80% last reached **40.834 token/s** versus **37.666** for 60% last in the same position sweep;
    at K=7168, 60% middle reached **39.643** versus **34.716** for 60% last.
- **Historical beta comparison**
  - At 60% last and K=6144..10240, beta=0.5 lowered decode throughput by **5.1%-14.9%** versus beta=1.
  - The new beta=1 run at K=7168 gives the best observed prefill, **99.192 token/s**;
    at K=9216 it gives the best observed decode, **32.486 token/s**.
  - These measurements used eager first-QMoE prediction, not the later end-of-Run temporal predictor.
    They include synchronization overhead and do not isolate prediction quality.
    Both predictors have been removed.
- **On-demand swaps**
  - If only one selected expert is on CPU, swap it to CUDA and compute there, provided it fits.
  - Swapping two is worse for decode, but can still improve prefill.
  - Set `moe_cpu_offload_on_demand_threshold` to `1` or `2`; `0` keeps the previous behavior.

### Next ideas

- Automatically increase or reduce parameters when measured throughput improves, instead of relying only on defaults.
  This is not implemented yet.

### Tables

**Best observed throughput across all configurations**

Includes the paired and partial beta sweeps and historical series omitted from the Summary.
Higher is better for both metrics.
At K=0, strategy names identify the all-GPU control cohort. The [GPU-interference warning](#on-demand-threshold-sweep) still applies.

| K | CUDA Graph | Best prefill (token/s) | Strategy | Best decode (token/s) | Strategy |
|---:|:---:|---:|---|---:|---|
| 0 | Off | 1370.534 | 60% last | 137.428 | 60% last |
| 0 | On | 1312.644 | All-GPU control (on-demand cohort) | 176.596 | 70% first (earlier) |
| 1024 | Off | 945.818 | 80% first | 105.088 | 80% first N=1 |
| 2048 | Off | 775.430 | 70% first (earlier) | 78.171 | 70% first (earlier) |
| 3072 | Off | 479.888 | Adaptive N=1 | 59.988 | 80% last |
| 4096 | Off | 294.650 | Adaptive N=1 | 50.377 | 80% first N=1 |
| 5120 | Off | 235.621 | 60% last, limit 16 | 50.620 | 70% first (earlier) |
| 6144 | Off | 143.041 | 60% last, limit 16 | 42.161 | 70% last beta=1 (partial sweep) |
| 7168 | Off | 99.192 | 60% last beta=1 (new) | 39.643 | 60% middle |
| 8192 | Off | 82.178 | 70% last beta=0.5 (partial sweep) | 36.897 | 60% last, limit 16 |
| 9216 | Off | 71.103 | Adaptive N=2 | 32.486 | 60% last beta=1 (new) |
| 10240 | Off | 48.753 | Adaptive | 29.069 | 80% first (earlier) |

Each cell below is **prefill / decode, both in token/s**; higher is better. Best rates per row are in **bold**.
`n/a` means the configuration was not measured, not a zero throughput.

| K | CUDA Graph | Adaptive | Basic | 80% first | 80% first N=1 | 60% last, limit 16 | 60% last beta=1 (new) | 60% last beta=0.5 (new) |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | Off | 1324.879 / 129.930 | 1339.841 / 132.550 | **1351.417** / **133.797** | 1328.399 / 127.829 | 1324.315 / 128.579 | n/a | n/a |
| 0 | On | 1308.962 / **169.841** | 1282.051 / 168.622 | 1292.169 / 169.261 | **1312.644** / 168.862 | 1298.701 / 168.794 | n/a | n/a |
| 1024 | Off | 827.692 / 79.855 | 332.822 / 85.967 | **945.818** / 94.298 | 934.112 / **105.088** | 502.368 / 64.073 | n/a | n/a |
| 2048 | Off | 602.708 / 57.834 | 191.088 / 70.539 | **647.444** / 65.174 | 570.800 / **73.977** | 439.064 / 53.446 | n/a | n/a |
| 3072 | Off | 183.352 / 39.793 | 144.480 / **59.878** | 252.239 / 51.002 | 257.391 / 54.272 | **328.685** / 41.665 | n/a | n/a |
| 4096 | Off | 133.551 / 37.036 | 112.214 / 46.850 | 146.583 / 48.925 | 137.493 / **50.377** | **277.165** / 35.319 | n/a | n/a |
| 5120 | Off | 94.211 / 33.336 | 83.495 / 44.232 | 105.216 / 42.609 | 103.969 / 44.327 | **235.621** / **47.282** | n/a | n/a |
| 6144 | Off | 86.313 / 30.664 | 82.124 / **39.411** | 74.545 / 38.250 | 81.583 / 38.657 | **143.041** / 38.835 | 125.367 / 38.359 | 100.143 / 32.935 |
| 7168 | Off | 85.651 / 29.203 | 63.536 / 36.474 | 75.762 / 36.665 | 67.989 / **36.679** | 73.745 / 31.635 | **99.192** / 33.509 | 78.386 / 28.513 |
| 8192 | Off | 64.482 / 27.003 | 48.445 / 33.758 | 53.737 / 34.057 | 62.090 / 33.569 | 63.494 / **36.897** | **67.923** / 34.420 | 66.849 / 32.679 |
| 9216 | Off | **66.962** / 25.696 | 51.711 / 29.513 | 49.506 / 29.195 | 31.137 / 4.866 | 63.847 / 30.168 | 57.950 / **32.486** | 51.254 / 29.219 |
| 10240 | Off | **48.753** / 27.032 | 44.187 / 27.746 | 41.370 / **28.076** | n/a | 44.841 / 27.640 | 43.236 / 27.630 | 41.376 / 25.963 |
