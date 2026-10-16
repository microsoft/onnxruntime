# MoE/QMoE CPU offload

Keep `K` experts on CPU to reduce GPU memory. The other experts stay on GPU.
CPU offload happens inside the built-in CUDA kernels, not through CPU execution-provider fallback.

## Initial placement and ownership

| Strategy | Placement |
|---|---|
| `adaptive` (default, MoE/QMoE) | Put popular experts on GPU and exchange hot CPU experts with cold GPU experts. Without saved counters, distribute GPU slots round-robin across nodes. |
| `basic` (QMoE only) | Fill earlier QMoE nodes on GPU, then put the rest on CPU. Placement never changes and ignores popularity. |
| `layered` (QMoE only) | Fill earlier layers up to a configurable GPU fraction (default 80%), then leave the rest on CPU. Use popularity and swaps within GPU-resident layers, including the boundary layer. |

`K` is global across eligible nodes. Basic uses topological order within each graph; a boundary node may be split
to preserve the exact count, keeping its lower expert IDs on GPU. `K=0` disables offload.

Layered uses the same ordering and rounds each layer's GPU target down to whole experts.
If the budget exceeds all layer targets combined, only the necessary excess is added to the earliest layers,
preserving exactly `K` CPU experts.

Adaptive and layered track selection with `counter = alpha * counter + (1 - alpha) * selected`.
A swap requires `hottest_CPU > (1 + epsilon) * coldest_GPU`. Each node keeps its initial GPU capacity.
CPU/GPU weights occupy separate compact stores; computation overlaps and completed swaps are published safely.
Supported INT4 SwiGLU configurations use packed MLAS Q4 weights on CPU; other QMoE cases use a dense fallback.

## Configuration

```python
import onnxruntime as ort

options = ort.SessionOptions()
options.add_session_config_entry("session.moe_cpu_offload_experts", "4096")
options.add_session_config_entry("session.moe_cpu_offload_strategy", "layered")  # Or adaptive/basic.
options.add_session_config_entry("session.moe_cpu_offload_gpu_ratio", "0.8")  # Layered only; (0, 1].
options.add_session_config_entry("session.moe_expert_counter_alpha", "0.9")
options.add_session_config_entry("session.moe_expert_swap_epsilon", "0.1")
options.add_session_config_entry("session.moe_max_in_flight_swaps_per_device", "5")
session = ort.InferenceSession("model.onnx", options, providers=["CUDAExecutionProvider"])
```

Defaults: `K=0`, strategy `adaptive`, alpha `0.75`, epsilon `0.01`, swap limit `4`.
Basic never swaps; for adaptive/layered, a zero swap limit freezes placement.

Supported: FP16/BF16 MoE weights and uniform INT4/INT8 QMoE (`quant_type="int"`, `weights_prepacked=0`).
Both require FP16/BF16 activations, constant weights and connected biases, a static hidden dimension, and prepacking.
Offload does not support FC3, sparse mixer, FP32 MoE, other quantization formats, CUDA Graph capture,
plugin CUDA EP, minimal builds, or overlapping runs on one session.

Details: [operator support](../contrib_ops/cuda/moe_qmoe.md) and
[counters, saved state, and logging](../MoEExpertCounting.md).

## Qwen3.5 INT4 benchmark

**Qwen3.5-35B-A3B INT4**, 40 QMoE nodes x 256 experts = 10,240 experts, top-k 8.
Alpha **0.9**, epsilon **0.1**, probability weight **0**, swap limit **5** (ignored by basic).
At `K=10240`, all experts are on CPU, but other model operations still use CUDA.

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

Measured 2026-10-16 at `358cb402b1` (adaptive/basic), `408fcd0ba2` (layered 80%) and `7c50cd1982` (layered 70%):
Linux, ORT 1.32.0, GenAI CUDA 0.15.2, CUDA 13.3.73, driver 580.105.08,
Xeon Platinum 8480C (96 physical cores). Release build, native CUDA architecture, Flash/MEA enabled.
These are single-run observations on a shared host, not statistical estimates or quality tests.

### Summary: decode throughput (tokens/s)

Highest observed rate per row in **bold**. Prefill times and memory peaks are detailed below.

| K | CUDA Graph | Adaptive | Basic | Layered 80% | Layered 70% |
|---:|:---:|---:|---:|---:|---:|
| 0 | Off | 129.930 | 132.550 | **132.599** | 130.596 |
| 0 | On | 169.841 | 168.622 | 169.083 | **176.596** |
| 1024 | Off | 79.855 | 85.967 | **93.633** | 91.353 |
| 2048 | Off | 57.834 | 70.539 | 55.676 | **78.171** |
| 3072 | Off | 39.793 | **59.878** | 47.379 | 39.392 |
| 4096 | Off | 37.036 | 46.850 | **47.085** | 35.276 |
| 5120 | Off | 33.336 | 44.232 | 44.921 | **50.620** |
| 6144 | Off | 30.664 | 39.411 | **39.614** | 38.230 |
| 7168 | Off | 29.203 | 36.474 | **37.084** | 34.966 |
| 8192 | Off | 27.003 | 33.758 | 31.864 | **34.234** |
| 9216 | Off | 25.696 | 29.513 | **31.617** | 30.098 |
| 10240 | Off | 27.032 | 27.746 | **29.069** | 28.232 |

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
