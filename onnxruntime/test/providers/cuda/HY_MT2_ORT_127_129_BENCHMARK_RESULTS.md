# HY-MT2 ORT and llama.cpp CUDA Benchmark Results

## Summary

This document records the HY-MT2 CUDA benchmark results obtained with ONNX
Runtime (ORT) GPU 1.27, ORT GPU 1.29, and llama.cpp Q4_K_M on the same machine.

The benchmark used BiliBili's 130 built-in danmaku samples and the
`bililive_translate_pipe` named-pipe protocol. Each configuration was run in a
fresh translation process. GPU memory and utilization were measured with
PID-scoped Windows Display Driver Model (WDDM) counters rather than
`nvidia-smi`.

The most important observations are:

- ORT 1.27 showed no meaningful latency difference between the baseline and
  `session.disable_prepacking=1`.
- ORT 1.29 baseline was faster than its disable-prepacking case, but this was
  not FpA-IntB weight prepacking because FpA-IntB was disabled.
- ORT 1.29 `session.use_device_allocator_for_initializers=1` retained baseline
  latency while substantially reducing dedicated GPU memory.
- The official ORT 1.29 provider contains FpA-IntB, but enabling it failed for
  the HY-MT2 prefill shape `M=22, N=2048, K=2048` because no valid tactic was
  found.
- llama.cpp Q4_K_M completed the same 130 inputs in 61.72 ms average, about
  34-39% faster than the representative ORT cases, while using at least 363 MB
  less dedicated GPU memory.

## Environment

| Item | Value |
| --- | --- |
| Machine | `ORT-GPU-BENCH-9` |
| OS | Windows 11 Enterprise, build `26200` |
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU |
| NVIDIA driver | 610.62 (`32.0.16.1062` from WMI) |
| Foundry Local Core | July 23 development stack; runtime DLL reports 1.2.0 |
| ORT GenAI | `0.15.0-dev202607231321155` |
| CUDA dependencies | CUDA 12.8.x |
| Model alias | `hy-mt2-cuda:1` |
| Model source | `justinchuby/Hy-MT2-1.8B-ONNX`, `Q4_KQuant_tie/cuda` |
| `model.onnx` size | 522,204 bytes |
| `model.onnx.data` size | 1,102,761,792 bytes (1051.68 MiB) |
| `model.onnx.data` SHA-256 | `9B052A3776321696736C2ABE3C34DB84DB3A524DCB7ED0492D5AFEEEEDD1B17E` |
| llama.cpp | tag `b10156`, commit `91f8c9c5fb038c086e13e9cd823c29b33b07ba54` |
| llama.cpp CUDA compiler | CUDA 12.8.93 |
| llama.cpp model | `tencent/Hy-MT2-1.8B-GGUF`, `Hy-MT2-1.8B-Q4_K_M.gguf` |
| GGUF size | 1,133,080,448 bytes |
| GGUF SHA-256 | `DC5F44FCF1FA496EE7AD725982C0C8C553A4DE00259B53AF84C4B89FB0C06699` |

Model cache:

```text
C:\Users\lochi\hy_mt2_wayne_repro\foundry-cache
```

Actual model folder:

```text
C:\Users\lochi\hy_mt2_wayne_repro\foundry-cache\Tencent\hy-mt2-cuda-1\v1
```

llama.cpp GGUF:

```text
C:\Users\lochi\hy_mt2_wayne_repro\models\Hy-MT2-1.8B-Q4_K_M.gguf
```

## Benchmark Method

- Engines: Foundry/ORT and llama.cpp, each run in a separate process.
- Samples: 130 built-in Chinese danmaku inputs.
- Warmups: 3 per process.
- Process lifecycle: fresh `livehime_translate.exe` process per case.
- IPC: message-mode named pipe `bililive_translate_pipe`.
- Resource accounting:
  - `\GPU Process Memory(*)\Dedicated Usage`
  - `\GPU Process Memory(*)\Shared Usage`
  - `\GPU Engine(*)\Utilization Percentage`
- Latency outliers were retained.
- External quality judge was disabled.
- The same deterministic heuristic quality score was used for every result:
  - Empty output: minus 100.
  - Output equals source: minus 60.
  - Output contains Chinese characters: minus 25.
  - Translation-label prefix: minus 10.
  - Output longer than `max(30, source length * 3)`: minus 15.
  - Each row was clamped to zero, then averaged across all 130 rows.

The supplied `livehime_translate.exe` was an incomplete .NET apphost whose
required `FoundryPipeRunner.dll` was unavailable. A native compatible runner
was reconstructed using the Foundry C++ SDK. It uses this translation prompt:

```text
将以下文本翻译为英语，只输出译文：
<source>
```

The runner sets `ORT_ENABLE_CUDNN_FLASH_ATTENTION=0`. This was required because
the RTX 5090 selected a cuDNN GroupQueryAttention path that failed for this
model. This differs from Wayne's RTX 4070 SUPER execution path.

The available `livehime_translate_llama.exe` was also an incomplete
`FoundryPipeRunner` apphost rather than a llama.cpp implementation. A compatible
native named-pipe runner was built against the exact documented llama.cpp
revision. It used:

```text
n_gpu_layers=99
n_ctx=512
n_batch=512
temperature=0.7
top_p=0.6
top_k=20
repetition_penalty=1.05
max_tokens=100
```

The llama.cpp prompt was:

```text
system:
/no_think 你是弹幕翻译器，只负责把中文翻译为英语。输出必须完全使用英语，只能输出翻译结果，不要解释，
不要输出源语言、原文或与输入完全相同的文本。即使输入是拟声词、语气词或重复情绪音，也必须翻译成
目标语言中的自然表达。

user:
[Translate]: <source>
```

## Session Configurations

### Baseline

No additional session option:

```json
{
  "model": {
    "decoder": {
      "session_options": {
        "log_id": "onnxruntime-genai",
        "provider_options": [
          {
            "cuda": {
              "enable_skip_layer_norm_strict_mode": "1",
              "enable_cuda_graph": "0"
            }
          }
        ]
      }
    }
  }
}
```

### Disable prepacking

```diff
 "session_options": {
+  "session.disable_prepacking": "1"
 }
```

### Device allocator for initializers

```diff
 "session_options": {
+  "session.use_device_allocator_for_initializers": "1"
 }
```

## ORT 1.27 Results

The 1.27 case used the CUDA provider from
`Microsoft.ML.OnnxRuntime.Gpu.Windows 1.27.0`. The surrounding July 23
development runtime reported ORT Core `1.28.0-dev`, but the loaded CUDA
provider itself was the official 1.27 package binary.

| Case | GPU dedicated peak | Avg latency | P50 | P90 | Min | Max | Working set max | Private memory max | GPU shared max | GPU util avg | GPU util max | Quality | Success / empty / error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2310.14 MB | 101.95 ms | 98 ms | 131 ms | 55 ms | 228 ms | 1077.11 MB | 3577.21 MB | 76.73 MB | 28% | 56% | 88.96 | 130 / 0 / 0 |
| Disable prepacking | 1930.14 MB | 100.36 ms | 97 ms | 129 ms | 54 ms | 214 ms | 1020.39 MB | 3137.83 MB | 76.73 MB | 26% | 52% | 88.96 | 130 / 0 / 0 |
| Device allocator | 1932.14 MB | 101.13 ms | 97 ms | 130 ms | 49 ms | 234 ms | 1022.79 MB | 3154.70 MB | 76.73 MB | 26% | 52% | 88.96 | 130 / 0 / 0 |

### ORT 1.27 interpretation

- Disabling prepacking reduced dedicated GPU memory by 380 MB.
- Latency remained effectively unchanged.
- Device allocator produced nearly the same memory and latency as disabling
  prepacking.
- Official ORT 1.27 GPU packages compiled FpA-IntB out. The package's CMake
  option `onnxruntime_USE_FPA_INTB_GEMM` was disabled.
- The CUDA provider DLL contained none of these strings:

```text
ORT_FPA_INTB_GEMM
ORT_FPA_INTB_DEBUG
ep.cuda.fpa_intb_gemm
[fpA_intB_debug]
```

Therefore, the ORT 1.27 memory difference was not caused by FpA-IntB packed
`MatMulNBits` weights. It reflected global initializer allocation, lifetime,
memory-pattern, fragmentation, or prepacking by other operators.

## ORT 1.29 Results

The isolated 1.29 runtime used:

| Component | Version |
| --- | --- |
| `onnxruntime.dll` | `1.29.0.20260811.4.2e2543f` |
| `onnxruntime_providers_shared.dll` | `1.29.0.20260811.4.2e2543f` |
| `onnxruntime_providers_cuda.dll` | `1.29.0.20260811.4.2e2543f` |
| `onnxruntime-genai.dll` | `0.15.0-dev202607231321155` |

The 1.29 CUDA provider was taken from:

```text
Microsoft.ML.OnnxRuntime.Gpu.Windows 1.29.0
```

The provider file was 174,478,688 bytes. Foundry's normal EP registration
would overwrite it with the older 1.27 provider, so the isolated runner used:

```text
ORT_SKIP_FOUNDRY_EP_REGISTRATION=1
```

| Case | GPU dedicated peak | Avg latency | P50 | P90 | Min | Max | Working set max | Private memory max | GPU shared max | GPU util avg | GPU util max | Quality | Success / empty / error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2740.14 MB | 98.62 ms | 96 ms | 130 ms | 51 ms | 197 ms | 1082.16 MB | 4013.44 MB | 76.73 MB | 27.5% | 55% | 88.96 | 130 / 0 / 0 |
| Disable prepacking | 1966.14 MB | 143.73 ms | 136 ms | 212 ms | 54 ms | 269 ms | 1012.50 MB | 3166.99 MB | 76.73 MB | 22.5% | 45% | 88.96 | 130 / 0 / 0 |
| Device allocator | 2030.14 MB | 100.52 ms | 96 ms | 131 ms | 46 ms | 196 ms | 1014.61 MB | 3233.34 MB | 76.73 MB | 26% | 52% | 88.96 | 130 / 0 / 0 |

### ORT 1.29 interpretation

- Disabling prepacking reduced dedicated GPU memory by 774 MB.
- The original 143.73 ms disabled-prepacking result was not reproducible and
  must be treated as an outlier, not a 45.7% causal regression.
- Device allocator reduced dedicated GPU memory by 710 MB while retaining
  latency close to baseline.
- All three configurations produced identical translations and the same
  heuristic quality score.

### ORT 1.29 repeat and allocator-isolation results

Fresh-process A-B-A-B repeats initially appeared to show a smaller baseline
advantage:

| Run | Configuration | Avg latency | P50 | P90 | GPU dedicated peak |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | Baseline | 92.08 ms | 88 ms | 124 ms | 2676.14 MB |
| B1 | Disable prepacking | 98.45 ms | 95 ms | 136 ms | 1966.14 MB |
| A2 | Baseline | 94.47 ms | 88 ms | 129 ms | 2682.14 MB |
| B2 | Disable prepacking | 101.40 ms | 96 ms | 145 ms | 1966.14 MB |

However, each baseline preceded its disabled-prepacking partner on a shared
laptop GPU. Later repetitions showed substantial run-order/system-state
variation, so the approximately 6.65 ms difference in that ordering is not
evidence of a prepacking speedup.

The decisive isolation added this configuration:

```diff
 "session_options": {
+  "session.disable_prepacking": "1",
+  "session.enable_mem_pattern": "0"
 }
```

`session.disable_prepacking=1` normally enables the initializer memory-pattern
allocator because `SessionState` creates `TensorAllocatorWithMemPattern` when
both prepacking is disabled and session memory patterns are enabled. Baseline
instead uses `SimpleTensorAllocator`, allowing initializers to be released
individually if a kernel packs them. Disabling the memory pattern in the
disabled-prepacking case forces the same simple initializer allocation mode as
baseline.

Foundry runs confirmed the allocator identity through memory:

| Configuration | Representative avg latency | GPU dedicated peak | Private memory peak |
| --- | ---: | ---: | ---: |
| Baseline | 92.08-94.74 ms | 2676.14-2682.14 MB | 3935.62-3943.42 MB |
| Disable prepacking, memory pattern enabled | 87.72-101.40 ms | 1966.14 MB | 3167.26-3174.88 MB |
| Disable prepacking, memory pattern disabled | 86.33-106.62 ms | 2674.14-2678.14 MB | 3934.42-3938.67 MB |

The latency ranges overlap and move substantially with run order, while the
memory result is deterministic: disabling the memory pattern restores the
baseline memory footprint even though prepacking remains disabled.

A direct native `InferenceSession` diagnostic then read ORT's
`SessionState::GetNumberOfPrepacksCounter()` and independently toggled the two
controls. Each case was run twice with FpA-IntB disabled:

| Configuration | Prepack count | Avg latency, two runs | Init peak delta |
| --- | ---: | ---: | ---: |
| Baseline | 0 | 19.75 / 18.50 ms | 2058 MiB |
| Disable prepacking, memory pattern enabled | 0 | 18.66 / 18.69 ms | 1182 MiB |
| Disable prepacking, memory pattern disabled | 0 | 18.86 / 19.27 ms | 2058 MiB |

This proves that no kernel successfully prepacked an initializer in the
non-FpA baseline. The large memory difference comes from the initializer
allocator selected as a side effect of `session.disable_prepacking`, not from
packed `MatMulNBits` weights. No repeatable latency benefit from the prepack
pass was measured; the earlier Foundry latency differences were benchmark
variance from GPU/system state and, in one run, different generated output.

## Model Payload and VRAM Accounting

ORT's higher VRAM usage is not caused by a larger serialized model. The ONNX
tensor payload is smaller than the GGUF Q4_K_M tensor payload.

| Model storage | ONNX Q4 | GGUF Q4_K_M |
| --- | ---: | ---: |
| Logical quantized weights | 1.791B | 1.791B |
| Quantized weight payload | 987.42 MiB | 1075.20 MiB |
| Other tensor payload | 64.27 MiB | 0.54 MiB |
| Total tensor payload | 1051.69 MiB | 1075.74 MiB |
| Complete model files | 1052.17 MiB | 1080.59 MiB |
| Effective quantized bits per weight | 4.625 bits | 5.036 bits |

The ONNX figures are the sum of serialized `TensorProto` payload lengths,
including tensors stored in `model.onnx.data`. These tensors are deserialized
and copied or mapped into runtime allocations, but the `MatMulNBits` weights do
not expand to FP16 merely because they are loaded into VRAM. Their packed
4-bit representation remains usable by the CUDA kernel.

### ONNX quantization layout

The ONNX graph contains 225 block-32 `MatMulNBits` nodes. After accounting for
the tied embedding/output weight, their unique quantized storage represents
approximately 1.791 billion logical weights.

| ONNX component | Size | Bits per logical weight |
| --- | ---: | ---: |
| Packed 4-bit values | 854.02 MiB | 4.000 |
| FP16 scales | 106.75 MiB | 0.500 |
| Packed zero points | 26.69 MiB | 0.125 |
| **Total** | **987.42 MiB** | **4.625** |

### GGUF quantization layout

| GGUF tensor type | Elements | Size |
| --- | ---: | ---: |
| Q4_K | 1.325B | 711.00 MiB |
| Q6_K | 465.5M | 364.20 MiB |
| F32 | 141K | 0.54 MiB |

GGUF Q4_K_M is larger because it stores approximately 26% of the model's
weights using the higher-precision Q6_K format. The ONNX model stores its
`MatMulNBits` matrices uniformly as 4-bit values with FP16 scales and packed
zero points.

### VRAM above serialized tensor payload

| Runtime/configuration | Tensor payload | Dedicated VRAM peak | VRAM above serialized tensors |
| --- | ---: | ---: | ---: |
| llama.cpp | 1075.74 MiB | 1603.55 MiB | 527.81 MiB |
| ORT 1.29 compact initializer allocation | 1051.69 MiB | 1966.14 MiB | 914.45 MiB |
| ORT 1.29 baseline | 1051.69 MiB | 2676.14 MiB | 1624.45 MiB |

The benchmark labels these process-memory values as MB, but the script divides
bytes by PowerShell's `1MB` constant, which is 1,048,576 bytes. The values are
therefore binary MiB in practice.

The difference above the original tensor payload can include:

- CUDA context and CUDA library state.
- ORT CUDA arena capacity and fragmentation.
- ORT GenAI and Foundry runtime state.
- Persistent execution buffers.
- KV-cache and temporary kernel workspaces.
- Initializers introduced or transformed by graph optimization.
- Alignment and allocation granularity.
- The model's precomputed rotary embedding caches.

These categories overlap in the end-to-end process peak and cannot be assigned
exact byte counts from the WDDM measurement alone.

### Initializer allocator isolation

The direct native test isolates initializer allocation more precisely:

| ORT initializer mode | Initialization GPU increase | Above original tensor payload |
| --- | ---: | ---: |
| Memory-pattern allocator | 1182 MiB | 130 MiB |
| Simple/arena allocator | 2058 MiB | 1006 MiB |

The approximately 876 MiB difference is allocator and arena behavior, not
additional model weights or Q4-to-FP16 expansion.

### Potential ORT VRAM improvements

1. Use `session.disable_prepacking=1` for this model while FpA-IntB remains
   disabled:

   ```json
   "session.disable_prepacking": "1"
   ```

   In the tested ORT implementation, this allows the default-enabled session
   memory pattern to be used for initializer allocation. It reduced dedicated
   VRAM from approximately 2676-2740 MiB to 1966 MiB. Runtime instrumentation
   reported `prepack_count=0`, so the successful non-FpA baseline had no
   effective operator prepacking to lose.

2. Prefer the initializer memory-pattern configuration over
   `session.use_device_allocator_for_initializers=1` when minimizing VRAM:

   ```text
   Memory-pattern initializer allocation: 1966 MiB
   Device allocator for initializers:      2030 MiB
   ```

   The memory-pattern case used approximately 64 MiB less dedicated GPU
   memory in this benchmark.

3. Consider generating a model variant with shorter rotary caches. Almost all
   of the 64.27 MiB non-quantized ONNX payload consists of:

   ```text
   model.rotary_emb.cos_cache  [262144, 64] FP16: 32 MiB
   model.rotary_emb.sin_cache  [262144, 64] FP16: 32 MiB
   ```

   For a maximum context of 512, the equivalent storage is:

   ```text
   2 * 512 * 64 * 2 bytes = 0.125 MiB
   ```

   A shortened model could therefore save approximately 63.9 MiB of model and
   initializer memory. This requires changing and validating the ONNX model;
   changing only `genai_config.json` does not resize existing tensors.

4. Lower GenAI context limits for the translation workload where application
   requirements permit it. The current configuration declares:

   ```json
   "context_length": 262144,
   "max_length": 262144
   ```

   Short translations with at most 100 generated tokens can use a much lower
   limit, such as 512. Whether this reduces memory depends on when and how the
   GenAI version allocates KV-cache and other context-dependent buffers, so it
   must be measured rather than assumed.

5. Profile the approximately 130 MiB difference between the original
   serialized tensor payload and the native compact-initializer peak. Useful
   next diagnostics include saving the optimized graph, summing its
   post-optimization initializers, and logging persistent CUDA allocations.
   The entire 130 MiB should not be assumed to be removable because it also
   includes unavoidable runtime and alignment overhead.

An approximate near-term target is:

```text
Current ORT compact result:          1966 MiB
Potential rotary-cache reduction:    ~64 MiB
Additional overhead to investigate: ~130 MiB
Plausible range after optimization: 1770-1900 MiB
llama.cpp observed result:           1604 MiB
```

The `1770-1900 MiB` range is an engineering estimate, not a measured result.
Closing the remaining gap likely requires reducing CUDA arena and runtime
overhead in addition to changing model tensors.

### Consolidated ORT allocator and llama.cpp comparison

The following table uses representative same-output ORT runs. The ORT
initialization peak is the CUDA device-memory increase above the pre-session
baseline from the native diagnostic test. GPU dedicated and private-memory
peaks are from the PID-scoped 130-sample end-to-end benchmark.

| Configuration | Avg latency | P50 / P90 | Init GPU peak delta | GPU dedicated peak | Private memory peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORT 1.29 baseline | 92.08 ms | 88 / 124 ms | 2058 MiB | 2676.14 MB | 3935.62 MB |
| ORT 1.29 disable prepacking, memory pattern enabled | 98.45 ms | 95 / 136 ms | **1182 MiB** | 1966.14 MB | 3174.88 MB |
| ORT 1.29 disable prepacking, memory pattern disabled | 88.02 ms | 83 / 116 ms | 2058 MiB | 2676.14 MB | 3934.94 MB |
| llama.cpp Q4_K_M | **61.72 ms** | **60 / 78 ms** | Not captured | **1603.55 MB** | **2181.52 MB** |

The llama.cpp resource sampler began after model initialization, so a directly
comparable llama.cpp initialization peak was not recorded. The nearly
identical ORT baseline and memory-pattern-disabled footprints demonstrate that
the ORT memory reduction comes from compact initializer memory-pattern
allocation. The ORT latency differences remain within the observed
fresh-process and machine-state variance.

## Direct ORT 1.27 versus 1.29 Comparison

| Case | ORT 1.27 GPU | ORT 1.29 GPU | Difference | ORT 1.27 avg | ORT 1.29 avg | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2310.14 MB | 2740.14 MB | +430.00 MB | 101.95 ms | 98.62 ms | -3.33 ms |
| Disable prepacking | 1930.14 MB | 1966.14 MB | +36.00 MB | 100.36 ms | 143.73 ms (outlier) | not causal |
| Device allocator | 1932.14 MB | 2030.14 MB | +98.00 MB | 101.13 ms | 100.52 ms | -0.61 ms |

## llama.cpp Q4_K_M Results

The llama.cpp CUDA run used the same 130 samples, three warmups, fresh-process
lifecycle, named-pipe latency measurement, and PID-scoped WDDM counters.

| Metric | llama.cpp Q4_K_M |
| --- | ---: |
| Success / empty / error | 130 / 0 / 0 |
| Average latency | 61.72 ms |
| P50 / P90 | 60 / 78 ms |
| Min / max | 39 / 104 ms |
| Working set peak | 1479.06 MB |
| Private memory peak | 2181.52 MB |
| Dedicated GPU peak | 1603.55 MB |
| Shared GPU peak | 88 MB |
| GPU utilization average / maximum | 42.33% / 64% |
| Deterministic heuristic quality | 90.77 |

The heuristic quality deductions included 77 overlong outputs under the strict
length rule, two outputs beginning with `Translate`, and one output retaining
Chinese characters. There were no empty outputs or request failures.

### llama.cpp versus ORT

| Runtime/configuration | Avg latency | P50 / P90 | Dedicated GPU | Private memory | Quality |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp Q4_K_M | **61.72 ms** | **60 / 78 ms** | **1603.55 MB** | **2181.52 MB** | **90.77** |
| ORT 1.27 baseline | 101.95 ms | 98 / 131 ms | 2310.14 MB | 3577.21 MB | 88.96 |
| ORT 1.27 disable prepacking | 100.36 ms | 97 / 129 ms | 1930.14 MB | 3137.83 MB | 88.96 |
| ORT 1.29 baseline, repeat range | 92.08-94.47 ms | 88 / 124-129 ms | 2676.14-2682.14 MB | 3935.62-3943.42 MB | 88.96 |
| ORT 1.29 device allocator | 100.52 ms | 96 / 131 ms | 2030.14 MB | 3233.34 MB | 88.96 |

Relative to the ORT 1.29 repeated baseline, llama.cpp was approximately
33-35% faster and used approximately 40% less dedicated GPU memory. Relative
to the lower-memory ORT 1.29 configurations, llama.cpp was approximately 38%
faster and used at least 363 MB less dedicated GPU memory.

llama.cpp used more working-set memory than ORT but substantially less private
memory. This is consistent with GGUF file mapping and a smaller runtime-owned
allocation footprint.

These numbers compare complete application stacks, not only matrix
multiplication kernels. The llama.cpp run used GGUF Q4_K_M and a system/user
chat template. The Foundry runs used the ONNX quantized model, a different
prompt, and model-config sampling defaults. The performance and memory
comparison is valid for these tested stacks, but it is not a controlled
same-format kernel comparison.

## FpA-IntB Verification

### ORT 1.27

The official 1.27 Windows CUDA provider did not include the active FpA-IntB
implementation. Setting either the environment variable or newer session
configuration had no effect.

### ORT 1.29

ORT 1.28 is the first released ORT version that enables FpA-IntB in official
CUDA packages. ORT 1.29 also includes it, and its CUDA provider contains:

```text
ORT_FPA_INTB_GEMM
ORT_FPA_INTB_DEBUG
ep.cuda.fpa_intb_gemm
[fpA_intB_debug]
```

When the process genuinely loaded the ORT 1.29 provider and
`ORT_FPA_INTB_GEMM=1` was set, FpA-IntB activated but failed during the first
HY-MT2 prefill:

```text
No valid fpA_intB MatMulNBits tactic for M=22, N=2048, K=2048
```

The failing node was:

```text
model/layers.0/self_attn/q_proj/MatMul_node_14_Q4
```

The successful 130-sample ORT 1.29 runs therefore had FpA-IntB disabled. In
that mode:

- CUDA `MatMulNBits::PrePack()` does not generate FpA-IntB weight, scale, or
  zero-point buffers.
- The kernel first tries the ordinary fused raw-Q4 `TryMatMulNBits` path.
- If the fused path is ineligible, it falls back to dequantization followed by
  a conventional GEMM.
- `session.disable_prepacking` remains a global session option and can still
  affect other operators and initializer allocation/lifetime behavior.

### Measured 130-sample MatMulNBits dispatch and fallback workspace

The actual Foundry named-pipe benchmark was instrumented with FpA-IntB
disabled and run over all 130 translation samples. This routing pass used zero
warmups so the counts below contain exactly the 130 requests, without the
script's usual three additional warmup requests.
The per-node console tracing adds substantial overhead, so this run is used
only for dispatch and workspace accounting, not latency comparison.

| Legacy MatMulNBits route | Calls | Share |
| --- | ---: | ---: |
| Fused raw-Q4 `TryMatMulNBits` | 242,320 | 89.0% |
| Dequantize plus cuBLAS GEMM fallback | 29,881 | 11.0% |
| **Total** | **272,201** | **100.0%** |

Routing changes between prompt prefill and single-token generation:

- Prompt lengths ranged from `M=16` through `M=23`.
- 128 prompts had `M>16`, so all 225 `MatMulNBits` nodes fell back for those
  prefills.
- Two prompts had `M=16`. Their 224 transformer matrices used fused raw-Q4,
  while the output projection still fell back.
- The `M=1` generation phase produced 241,872 fused calls and 1,079 fallback
  calls. Every `M=1` fallback was the output projection.

The output projection has `N=120818`, which is not divisible by the fused
kernel's 8-column requirement. It therefore uses the fallback during both
prefill and generation. ORT chunks this matrix into 32768-row sections,
reducing its FP16 dequantization buffer from approximately 472 MiB to 128 MiB.

| Fallback matrix shape `(N, K)` | Calls | FP16 scratch per call | Chunked |
| --- | ---: | ---: | --- |
| `(512, 2048)` | 8,192 | 2 MiB | No |
| `(2048, 2048)` | 8,192 | 8 MiB | No |
| `(6144, 2048)` | 8,192 | 24 MiB | No |
| `(2048, 6144)` | 4,096 | 24 MiB | No |
| `(120818, 2048)` | 1,209 | 128 MiB | Yes |

The maximum live fallback request was 128 MiB. Summing every request across
the benchmark gives 519.125 GiB of temporary allocation traffic, but those
buffers are not simultaneously resident. Nodes execute sequentially and can
reuse CUDA arena capacity. Arena retention and fragmentation may nevertheless
preserve the 128 MiB high-water capacity after an operation finishes.

The model's Q4 weights remain packed persistently in VRAM. Fused calls consume
that packed representation directly. Fallback calls temporarily dequantize
only the active matrix or output-projection chunk to FP16, call cuBLAS GEMM,
and release the scratch allocation; ORT does not permanently expand the whole
model to FP16.

Consequently, the ORT 1.29 latency difference between baseline and disabled
prepacking must not be described as an FpA-IntB benefit. Direct runtime
instrumentation reported `prepack_count=0` for the successful non-FpA
baseline.

## Notable Latencies

No extreme latency outlier occurred in any completed 130-sample run.

| Runtime and case | Maximum latency | Sample |
| --- | ---: | --- |
| ORT 1.27 baseline | 228 ms | 1 |
| ORT 1.27 disable prepacking | 214 ms | 1 |
| ORT 1.27 device allocator | 234 ms | 1 |
| ORT 1.29 baseline | 197 ms | 1 |
| ORT 1.29 disable prepacking | 269 ms | 34 and 116 |
| ORT 1.29 device allocator | 196 ms | 1 |
| llama.cpp Q4_K_M | 104 ms | 93 |

Wayne's previously observed 8046 ms device-allocator outlier on sample 84 was
not reproduced.

## Result Artifacts

### ORT 1.27

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\A_baseline_foundry_raw
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\B_disable_prepacking_foundry_raw
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\C_device_allocator_foundry_raw
```

### ORT 1.29

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_A_baseline_foundry_raw
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_B_disable_prepacking_foundry_raw
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_C_device_allocator_foundry_raw
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_repeat_A1
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_repeat_B1
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_repeat_A2
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_repeat_B2
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_isolation_D1_no_prepack_no_mem_pattern
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_isolation_D2_no_prepack_no_mem_pattern
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_isolation_A3_baseline
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_isolation_D3_no_prepack_no_mem_pattern
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_isolation_B3_no_prepack_mem_pattern
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\ORT129_isolation_D4_no_prepack_no_mem_pattern
```

### llama.cpp

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\LLAMA_B10156_Q4_CUDA_130_CLEAN
```

Each result directory contains:

```text
results.json
results.csv
summary.md
```

Saved model configurations:

```text
genai_config.A.baseline.json
genai_config.B.disable_prepacking.json
genai_config.C.device_allocator.json
genai_config.ORT129.A.baseline.json
genai_config.ORT129.B.disable_prepacking.json
genai_config.ORT129.C.device_allocator.json
genai_config.ORT129.D.disable_prepacking_no_mem_pattern.json
```

## Reproduction Caveats

1. The supplied .NET named-pipe runner was incomplete, so a compatible native
   runner was reconstructed.
2. The exact prompt used by the original runner cannot be proven.
3. cuDNN Flash Attention was disabled to avoid an RTX 5090 GQA failure.
4. The ORT 1.29 experiment combines stable ORT 1.29 with the July 23
   development Foundry/ORT GenAI binaries. GenAI CUDA 0.15.2 officially
   declares ORT 1.28.0 as its dependency, not ORT 1.29.0.
5. End-to-end latency varied materially across fresh processes on this shared
   laptop GPU. The direct native isolation should be used for attributing the
   prepack and initializer-allocation behavior.
6. WDDM resource sampling produced three samples per short benchmark process,
   so GPU utilization averages are coarse.
7. The original llama.cpp runner binary was unavailable. The replacement used
   the documented llama.cpp commit, model, generation settings, prompt, and
   named-pipe protocol, but byte-for-byte equivalence with BiliBili's runner
   cannot be proven.
