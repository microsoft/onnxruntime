# HY-MT2 ORT Versus llama.cpp Memory Gap Analysis

## Status and Scope

This document records the validated GPU-memory comparison between ONNX Runtime
(ORT) CUDA and llama.cpp CUDA for the HY-MT2 1.8B Q4 translation model. It
supersedes the earlier preliminary 1930.14 MiB ORT result. That result came from
a deployment containing binaries built against incompatible internal layouts
and must not be used for source-level A/B conclusions.

The current measurements use a coherent source-built deployment:

- ORT branch: `chilo/static-workspace-preallocation-test`
- ORT commit: `e648c9c2fe684082d0c1b786049215ee2dbbd570`
- ORT GenAI commit: `ed5f4e87147731e5b07810f9f5c90103b3603cdf`
- CUDA 12.8 and cuDNN 9.5
- CUDA EP source-provider SHA-256:
  `F6785E36707EAD7DB10A38EEDE4CAF7555AED7D43F45596413D6AAC08125AB9D`
- llama.cpp tag `b10156`
- llama.cpp model: `Hy-MT2-1.8B-Q4_K_M.gguf`

## Experiment Method

Each configuration uses the following lifecycle:

1. Start fresh llama.cpp and Foundry processes.
2. Initialize the model.
3. Run one translation as warmup.
4. Call `Shrink()` exactly once on all registered ORT GPU arenas.
5. Run the same translation as the measured cached request.
6. Record process-local WDDM memory and ORT allocator checkpoints.

The workload is:

```text
主播声音有点小，能不能调大一点
```

Static workspace preallocation is disabled in the single-sample allocator
experiments below. The legacy CUDA `MatMulNBits` path in those experiments
reports:

```text
[legacy_workspace_check] source=scratch_fallback N=2048 K=2048 required_bytes=8388608
```

Foundry is forced to load the coherent source provider with:

```text
ORT_SKIP_FOUNDRY_EP_REGISTRATION=1
```

Arena and workspace diagnostics use:

```text
ORT_ARENA_DIAGNOSTICS=1
ORT_ARENA_DIAGNOSTIC_WARMUP_REQUESTS=1
ORT_MATMULNBITS_TRACE_LEGACY_WORKSPACE=1
ORT_BENCH_CAPTURE_RUNNER_LOGS=1
```

## Allocator Terminology

All allocator values are binary MiB (`1 MiB = 1,048,576 bytes`).

| Term | Meaning |
|---|---|
| `total_allocated_bytes` | All memory tracked by an allocator: direct reserves plus BFC backing regions |
| `reserved_bytes` | Direct allocations made through `Reserve()`; unavailable to normal BFC allocation |
| `bfc_region_bytes` | `total_allocated_bytes - reserved_bytes` |
| `bytes_in_use` | Live reserved and BFC chunk capacity |
| `bytes_requested_in_use` | Bytes requested by live allocation owners |
| Internal fragmentation | `bytes_in_use - bytes_requested_in_use` |
| BFC slack | Free capacity inside BFC backing regions |
| WDDM dedicated memory | Total process-local GPU memory, including ORT, CUDA libraries, contexts, and other allocations |

WDDM dedicated memory is the whole `livehime_translate.exe` process PID's
`GPU Process Memory(pid_...)\Dedicated Usage`; it is not an ORT-arena-only
measurement. It includes GPU resources allocated in that process by Foundry
Local Core, ONNX Runtime GenAI, the ORT CUDA EP, CUDA/cuDNN/cuBLAS, and the
runner itself. Examples include model weights, KV caches, activations,
workspaces, CUDA contexts, and library-internal buffers.

Consequently, process dedicated memory is normally larger than the sum of
ORT's allocator `total_allocated_bytes`. The difference represents
same-process GPU allocations outside the instrumented ORT allocators, not
unaccounted BFC slack or an allocator leak. It excludes allocations attributed
to other PIDs and excludes WDDM shared GPU memory, which the benchmark reports
separately. An out-of-process Foundry helper would also be excluded, although
the model runtime in this benchmark is loaded into `livehime_translate.exe`.

The runner reports two non-empty CUDA allocators after generation. Both have the
generic name `Cuda`. This document identifies them by lifecycle:

- **Initializer-owning CUDA allocator:** populated during model initialization.
- **Request-time CUDA allocator:** empty after initialization and grows during
  generation.

The names describe observed behavior, not an explicit allocator identifier.

## A/B Summary

| Metric | Default BFC initializers | `kSameAsRequested` | Device-allocator initializers |
|---|---:|---:|---:|
| Foundry process GPU memory | 2434.14 MiB | 2434.14 MiB | **1734.14 MiB** |
| llama.cpp process GPU memory | 1616.16 MiB | 1616.16 MiB | 1616.16 MiB |
| Foundry-minus-llama gap | 817.98 MiB | 817.98 MiB | **117.98 MiB** |
| Initializer-owning allocator total | 2048.00 MiB | 2047.00 MiB | **1346.68 MiB** |
| Direct reserved initializers | 0 MiB | 0 MiB | **1169.68 MiB** |
| Initializer-owning BFC regions | 2048.00 MiB | 2047.00 MiB | **177.00 MiB** |
| Initializer-owning BFC slack | 856.28 MiB | 734.67 MiB | **176.97 MiB** |
| Initializer-owning internal fragmentation | 22.01 MiB | 142.62 MiB | **0 MiB** |
| Request-time BFC region after measured request | 129.00 MiB | 129.00 MiB | 129.00 MiB |
| Foundry measured latency | 263 ms | 207 ms | 250 ms |
| llama.cpp measured latency | 73 ms | 70 ms | 72 ms |

The latency values are single samples and are included only to detect gross
regressions. The process-memory and allocator values are the focus of this
analysis.

## Default BFC Initializer Allocation

The original model configuration does not set
`session.use_device_allocator_for_initializers`, so its default value is `0`.
ORT places GPU initializers in planned memory backed by normal BFC allocation.

### Initializer-owning allocator

The post-cached-request checkpoint is:

```text
total_allocated_bytes=2147483648
reserved_bytes=0
bfc_region_bytes=2147483648
bytes_in_use=1249611776
bytes_requested_in_use=1226528584
arena_slack_bytes=897871872
internal_fragmentation_bytes=23083192
num_reserves=0
num_arena_extensions=11
```

Converted to MiB:

| Component | MiB |
|---|---:|
| Requested live allocation | 1169.71 |
| Internal fragmentation | 22.01 |
| Free reusable BFC slack | 856.28 |
| **BFC backing regions** | **2048.00** |

The 2048 MiB value is the aggregate size of this allocator's BFC backing
regions, not proof of one 2048 MiB allocation.

### Whole-process decomposition

| Component | MiB |
|---|---:|
| Initializer-owning CUDA allocator | 2048.00 |
| Request-time CUDA allocator | 129.00 |
| CUDA context, libraries, and allocations outside the registered arenas | 257.14 |
| **Foundry WDDM dedicated memory** | **2434.14** |

```text
2048.00 MiB initializer-owning allocator
+ 129.00 MiB request-time allocator
+ 257.14 MiB CUDA/driver/other
-----------------------------------------
= 2434.14 MiB Foundry process GPU memory
```

The distinction between 2048.00 MiB and 2177.00 MiB is scope:

```text
2048.00 MiB one initializer-owning CUDA allocator
+ 129.00 MiB second request-time CUDA allocator
-------------------------------------------------
= 2177.00 MiB all ORT-tracked CUDA allocations
```

## Why the 856.28 MiB Slack Remains

The slack is free and reusable by the same BFC allocator; it is not a leaked
live allocation. WDDM still counts it because ORT retains the underlying CUDA
regions.

The default BFC strategy is `kNextPowerOfTwo`. As demand grows, the arena
obtains progressively larger regions. `BFCArena::Shrink()` can release a
region only when every chunk in that region is unused. A long-lived initializer
in a region prevents the entire region from being returned to CUDA even if
other chunks in it are free.

The before/after checkpoints show that the initializer-owning arena had no
eligible empty region:

```text
Before shrink: total_allocated_bytes=2147483648 arena_slack_bytes=897871872
After shrink:  total_allocated_bytes=2147483648 arena_slack_bytes=897871872
```

Keeping the slack is a performance tradeoff. It avoids later `cudaMalloc`
calls, but it cannot reduce process GPU memory while the regions remain owned
by BFC.

## `kSameAsRequested` A/B

The controlled configuration added only:

```json
"arena_extend_strategy": "1"
```

The GenAI provider bridge requires the numeric value `1` for
`kSameAsRequested`.

| Initializer-owning allocator metric | `kNextPowerOfTwo` | `kSameAsRequested` | Delta |
|---|---:|---:|---:|
| BFC backing regions | 2048.00 MiB | 2047.00 MiB | -1.00 MiB |
| Requested live memory | 1169.71 MiB | 1169.71 MiB | 0 MiB |
| Internal fragmentation | 22.01 MiB | 142.62 MiB | +120.60 MiB |
| Free BFC slack | 856.28 MiB | 734.67 MiB | -121.60 MiB |
| Total above requested | 878.29 MiB | 877.29 MiB | -1.00 MiB |
| Foundry WDDM memory | 2434.14 MiB | 2434.14 MiB | **0 MiB** |

`kSameAsRequested` changed the shapes of BFC regions but did not solve the
lifetime problem. It moved approximately 121 MiB from free slack into internal
fragmentation. The default 128 MiB `max_dead_bytes_per_chunk` threshold allows
BFC to consume a larger free chunk without splitting off a smaller remainder.

Long-lived initializers still pinned all eligible regions, so process memory
did not change.

## Device-Allocator Initializer A/B

The successful configuration added only:

```json
"session.use_device_allocator_for_initializers": "1"
```

This makes ORT allocate initializers through `Reserve()` instead of placing
them in normal BFC regions.

### Process-level comparison

| Memory component | Default BFC initializers | Device-allocator initializers | Change |
|---|---:|---:|---:|
| Persistent initializer storage | Included in model BFC | 1169.68 MiB reserved | Separated |
| Initializer-owning BFC regions | 2048.00 MiB | 177.00 MiB | **-1871.00 MiB** |
| Request-time BFC region | 129.00 MiB | 129.00 MiB | 0 MiB |
| All ORT-tracked CUDA allocation | 2177.00 MiB | **1475.68 MiB** | **-701.32 MiB** |
| CUDA/driver/other | 257.14 MiB | 258.46 MiB | +1.32 MiB |
| **Foundry process GPU memory** | **2434.14 MiB** | **1734.14 MiB** | **-700.00 MiB** |
| llama.cpp process GPU memory | 1616.16 MiB | 1616.16 MiB | 0 MiB |
| **Foundry-minus-llama gap** | **817.98 MiB** | **117.98 MiB** | **-700.00 MiB** |

The device-initializer process equation is:

```text
1169.68 MiB persistent reserved initializers
+ 177.00 MiB initializer-owning transient BFC regions
+ 129.00 MiB request-time BFC region
+ 258.46 MiB CUDA/driver/other
------------------------------------------------------
= 1734.14 MiB Foundry process GPU memory
```

### Initializer-owning allocator details

| Metric | Default BFC initializers | Device-allocator initializers |
|---|---:|---:|
| Total allocated | 2048.00 MiB | **1346.68 MiB** |
| Reserved initializer memory | 0 MiB | **1169.68 MiB** |
| BFC backing regions | 2048.00 MiB | **177.00 MiB** |
| Total requested live memory | 1169.71 MiB | 1169.71 MiB |
| Live BFC memory excluding reserves | 1191.72 MiB | **0.03 MiB** |
| Internal fragmentation | 22.01 MiB | **0 MiB** |
| Free BFC slack | 856.28 MiB | **176.97 MiB** |
| `Reserve()` calls | 0 | **808** |
| BFC arena extensions | 11 | **5** |

The two scopes reconcile as follows:

```text
1169.68 MiB reserves + 177.00 MiB BFC = 1346.68 MiB
    initializer-owning CUDA allocator

1346.68 MiB initializer-owning CUDA allocator
+129.00 MiB request-time CUDA allocator
------------------------------------------------
=1475.68 MiB all ORT-tracked CUDA allocations
```

## `Reserve()` Accounting and Reuse

`BFCArena::Reserve()` calls the underlying device allocator directly and
updates:

```text
bytes_in_use
bytes_requested_in_use
reserved_bytes
num_reserves
total_allocated_bytes
```

A live reserved allocation is not available to normal BFC `Alloc()` calls and
is not released by `Shrink()`. When its owner calls `Free()`, ORT directly frees
the device allocation and subtracts it from `reserved_bytes` and
`total_allocated_bytes`.

This behavior is appropriate for model weights: they remain live for the
session lifetime and have no useful opportunity for request-time reuse. Moving
them outside BFC prevents them from pinning regions intended for transient
allocation.

## Shrink Behavior

| Shrink result | Default BFC initializers | Device-allocator initializers |
|---|---:|---:|
| Initializer-owning transient BFC reclaimed | 0 MiB | 176 MiB |
| Request-time BFC reclaimed | 96 MiB | 96 MiB |
| Total immediately reclaimed | 96 MiB | **272 MiB** |
| Persistent saving after the measured request from shrink itself | 0 MiB | 0 MiB |
| Persistent saving from initializer placement | 0 MiB | **700 MiB** |

The measured request regrew the transient regions:

```text
Request-time arena: 129 MiB -> 33 MiB after shrink -> 129 MiB after request

Device-initializer model BFC:
177 MiB -> 1 MiB after shrink -> 177 MiB after request
```

Therefore, `Shrink()` exposes how much memory is transiently releasable but
does not provide a lasting request-to-request saving for this workload. The
persistent 700 MiB improvement comes from initializer placement.

## Interpretation

The validated measurements support these conclusions:

1. The legacy 8 MiB `MatMulNBits` scratch allocation is not the dominant source
   of the memory gap.
2. Changing only the BFC growth strategy does not reduce process memory.
3. The dominant issue is mixing persistent initializers with transient BFC
   allocations. Long-lived weights pin oversized regions.
4. Allocating initializers directly reduces Foundry WDDM memory by 700 MiB and
   reduces the Foundry-minus-llama gap from 817.98 MiB to 117.98 MiB.
5. The remaining approximately 118 MiB includes differences in runtime
   buffers, CUDA library state, allocator behavior, and implementation details
   outside the registered BFC arenas.

The effective lifetime split is:

```text
Persistent exact allocation:
    model weights and initializers

Reusable transient BFC allocation:
    activations, temporary tensors, and dynamic workspaces

Memory-pattern allocation:
    stable per-request activation and workspace lifetimes
```

The device-allocator initializer configuration is the first tested change that
closes most of the coherent Foundry-versus-llama.cpp memory gap.

## Earlier 130-Sample Memory-Stability Diagnostic

The device-allocator initializer configuration was also run over the complete
130-sample corpus after one warmup and one post-warmup shrink. Retained
process-local WDDM memory was sampled synchronously after requests 1, 10, 25,
50, 100, and 130. The harness divides bytes by `1MB` in PowerShell, so values
named `*_mb` in its JSON output are binary MiB.

| Request | llama.cpp dedicated GPU | Foundry dedicated GPU | Foundry minus llama.cpp |
|---:|---:|---:|---:|
| 1 | 1616.16 MiB | 1734.14 MiB | 117.98 MiB |
| 10 | 1616.16 MiB | 1734.14 MiB | 117.98 MiB |
| 25 | 1616.16 MiB | 1734.14 MiB | 117.98 MiB |
| 50 | 1616.16 MiB | 1734.14 MiB | 117.98 MiB |
| 100 | 1616.16 MiB | 1734.14 MiB | 117.98 MiB |
| 130 | 1616.16 MiB | 1734.14 MiB | 117.98 MiB |

The process-wide sampled maxima were also 1616.16 MiB for llama.cpp and
1734.14 MiB for Foundry. Neither runtime accumulated dedicated GPU memory over
the corpus:

```text
Foundry retained growth, request 1 -> 130: 0.00 MiB
llama.cpp retained growth, request 1 -> 130: 0.00 MiB
```

This validates that the 700 MiB saving from direct initializer allocation is
persistent across varied inputs. It also shows no cumulative growth from
shape-specific memory patterns, generator state, or CUDA arena fragmentation
for this corpus.

This diagnostic run is not a reproduction of the earlier Chinese-language
quality benchmark. The relevant protocol differences are:

| Dimension | Earlier quality benchmark | This memory-stability run |
|---|---|---|
| GPU | RTX 4070 SUPER, SM89 | RTX 5090 Laptop, SM120 |
| Driver | 610.74 | 610.62 |
| ORT / GenAI | ORT 1.28 development build / GenAI 0.15.0 development build | Coherent rebased source build / GenAI commit `ed5f4e87` |
| Script warmups | 3 | 1 |
| Initializer allocation | Default BFC placement | `session.use_device_allocator_for_initializers=1` |
| Explicit arena shrink | None | Once after warmup |
| llama.cpp mode | `use_hy_model=false` | `use_hy_model=true` |
| Foundry prompt | Strict English-only translation prompt | Short `只输出译文` prompt |
| Generation settings | Temperature 0.7, top-p 0.6, top-k 20, repetition penalty 1.05, maximum 100 tokens | SDK defaults except maximum 128 tokens |

The model payloads are equivalent in size, but the runtime and generation
protocols are not. In particular, direct initializer allocation is the
intentional source of the 700 MiB Foundry memory reduction, while the prompt
and generation-setting differences change the number of decoded tokens and
therefore latency.

The latency results from this diagnostic run are:

| Metric | llama.cpp | Foundry |
|---|---:|---:|
| Successful requests | 130 | 130 |
| Average latency | 53.45 ms | 1178.83 ms |
| P50 | 52 ms | 584 ms |
| P90 | 66 ms | 2056 ms |

Foundry produced Chinese text in 129 of 130 outputs and produced much longer
responses than llama.cpp. Many requests therefore decoded substantially more
tokens, including a 13.1-second maximum. These timing values characterize this
diagnostic prompt/model behavior, not an equal-output translation-performance
comparison. The result establishes only that dedicated GPU memory remains flat
across 130 requests under the direct-initializer configuration.

## Protocol-Matched 130-Sample Reproduction

A second 130-sample A/B reproduces the workload documented in
`测试环境及相关参数说明.md` on the current RTX 5090 Laptop host. Unlike the
earlier diagnostic above, this run uses:

- three script warmups plus Foundry's internal load warmup;
- llama.cpp `use_hy_model=false`;
- the strict English-only prompts for both engines;
- temperature 0.7, top-p 0.6, top-k 20, repetition penalty 1.05, and a
  100-token maximum;
- no explicit arena `Shrink()`; and
- the same coherent ORT, ORT GenAI, Foundry runner, model payloads, and sample
  order in both initializer configurations.

The current C++ Foundry SDK serializes `ChatSettings.top_k` as numeric metadata,
while the managed bridge requires metadata values to be strings. Therefore
top-k 20 and the SDK-unexposed repetition penalty 1.05 were supplied through
temporary model-config overrides. Temperature, top-p, and maximum tokens were
set through `ChatSettings`. The model config was restored after both runs to
SHA-256
`01EEF87C41EE4D059BBAAB16700F9F506C7309BE4A31D43AE185AABD115233DE`.

This is a protocol reproduction, not an exact reproduction of the historical
RTX 4070 SUPER numbers. The current host uses an RTX 5090 Laptop GPU, driver
610.62, SM120, and the coherent rebased source runtime described above.

### Current-hardware baseline

| Metric | llama.cpp Q4 | Foundry INT4, default BFC initializers |
|---|---:|---:|
| Successful requests | 130 | 130 |
| Average latency | 54.31 ms | 89.86 ms |
| P50 / P90 | 53 / 67 ms | 85 / 121 ms |
| Process CPU average | 3.76% | 3.66% |
| Working-set maximum | 1505.88 MiB | 1107.62 MiB |
| Private-memory maximum | 2251.24 MiB | 3637.15 MiB |
| Process dedicated GPU maximum | 1616.16 MiB | 2380.14 MiB |
| Process shared GPU maximum | 88.73 MiB | 76.73 MiB |
| Process GPU-utilization maximum | 68% | 64% |
| Heuristic quality score | 90.23 | 89.27 |

All 130 Foundry requests succeeded. No Foundry output was empty and none
contained Chinese text. The average latency and quality score are close to the
historical RTX 4070 SUPER results of 92.38 ms and 89.15, respectively, which
confirms that the earlier 1178.83 ms result was caused by the mismatched prompt
and generation protocol rather than by direct initializer allocation.

### Initializer-placement A/B under the matched protocol

| Foundry metric | Default BFC initializers | Device-allocator initializers | Difference |
|---|---:|---:|---:|
| Average latency | 89.86 ms | 89.60 ms | -0.26 ms |
| P50 / P90 | 85 / 121 ms | 84 / 120 ms | -1 / -1 ms |
| Working-set maximum | 1107.62 MiB | 1014.33 MiB | -93.29 MiB |
| Private-memory maximum | 3637.15 MiB | 2854.38 MiB | -782.77 MiB |
| Process dedicated GPU maximum | 2380.14 MiB | 1662.14 MiB | **-718.00 MiB** |
| Heuristic quality score | 89.27 | 89.38 | +0.11 |

Relative to the 1616.16 MiB llama.cpp process:

```text
Default initializer gap: 2380.14 - 1616.16 = 763.98 MiB
Direct initializer gap:  1662.14 - 1616.16 =  45.98 MiB
Gap reduction:                                    718.00 MiB
```

The default configuration retained 2380.14 MiB at requests 1, 10, 25, 50,
and 100 and in the post-generation snapshot after all 130 requests. The
device-allocator configuration retained 1662.14 MiB at requests 1, 10, 25, 50,
100, and 130. Neither configuration accumulated dedicated GPU memory across
the corpus.

The 718 MiB reduction is slightly larger than the 700 MiB reduction in the
single-sample shrink experiment because this protocol uses longer prompts,
three warmups, and no post-warmup shrink. The conclusion is unchanged:
separating session-lifetime initializers from transient BFC allocation removes
nearly all of the ORT-versus-llama.cpp process-memory gap without a measurable
latency or output-quality regression.

### Initialization versus inference-time growth

The older `legacy_matmul_nbits_workspace_preallocation_analysis.md` reported
that llama.cpp retained more dedicated GPU memory immediately after
initialization, while ORT/Foundry grew more during inference. Its exact
ORT/Foundry values came from the superseded binary-incoherent deployment and
must not be reused. However, the relationship can be evaluated from the
coherent protocol runs because the harness records process-local WDDM
checkpoints before initialization, after initialization, and after all
generation.

The lifecycle values for default BFC initializer placement are:

| Dedicated GPU memory | llama.cpp | Foundry, default BFC initializers | Foundry minus llama.cpp |
|---|---:|---:|---:|
| Post-initialization | 1582.16 MiB | 2304.14 MiB | +721.98 MiB |
| Post-generation | 1616.16 MiB | 2380.14 MiB | +763.98 MiB |
| Growth after initialization | 34.00 MiB | 76.00 MiB | +42.00 MiB |

With default BFC initializer placement, the recalled relationship does **not**
hold: Foundry already retains substantially more dedicated GPU memory after
initialization. Foundry still grows 42 MiB more than llama.cpp during the
subsequent workload.

The lifecycle changes after enabling
`session.use_device_allocator_for_initializers=1`:

| Dedicated GPU memory | llama.cpp | Foundry, device-allocator initializers | Foundry minus llama.cpp |
|---|---:|---:|---:|
| Post-initialization | 1582.16 MiB | 1428.13 MiB | **-154.03 MiB** |
| Post-generation | 1616.16 MiB | 1662.14 MiB | **+45.98 MiB** |
| Growth after initialization | 34.00 MiB | 234.01 MiB | **+200.01 MiB** |

For this configuration, the recalled relationship is confirmed with coherent
binaries: llama.cpp retains 154.03 MiB more immediately after initialization,
but Foundry retains 200.01 MiB more additional memory during the subsequent
warmup and inference workload. This reverses the ordering by the end of
generation, leaving Foundry 45.98 MiB above llama.cpp.

A fresh 130-sample repeat run reproduced every dedicated-memory lifecycle
value in the table exactly. Its results are stored in:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
LIFECYCLE_DEVICE_INITIALIZERS_20260904_1453
```

| Fresh repeat-run metric | llama.cpp | Foundry, device-allocator initializers |
|---|---:|---:|
| Successful requests | 130/130 | 130/130 |
| Average translation latency | 53.20 ms | 88.95 ms |
| P50 / P90 latency | 53 / 64 ms | 86 / 119 ms |
| Dedicated GPU memory peak | 1616.16 MiB | 1662.14 MiB |

Both engines were already at their post-generation dedicated-memory values by
request 1 and remained unchanged at requests 10, 25, 50, 100, and 130:
1616.16 MiB for llama.cpp and 1662.14 MiB for Foundry. The repeat therefore
confirms that the 234.01 MiB Foundry growth occurs during initialization of
generation state and the three script warmups, not by accumulating memory
across the 130 measured requests.

#### Allocation-level attribution of the 234.01 MiB growth

A follow-up run enabled `ORT_ARENA_DIAGNOSTICS=1` and delayed the diagnostic
checkpoint until completed call 133: three script warmups plus 130 measured
requests. The `before_shrink` snapshot therefore represents the same retained
state as the untraced post-generation measurement:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
LIFECYCLE_DEVICE_INITIALIZERS_ARENA_TRACE_REGISTERED_20260904_1515
```

The first-level attribution is:

| Source of new retained dedicated memory | Growth |
|---|---:|
| Generation-time expansion of the decoder/model-session arena; free at the final checkpoint | **168.00 MiB** |
| ORT GenAI global device-allocator arena: requested live allocations | **14.27 MiB** |
| ORT GenAI global device-allocator arena: internal fragmentation | **3.88 MiB** |
| ORT GenAI global device-allocator arena: reusable slack | **46.85 MiB** |
| CUDA/GenAI/WDDM growth outside ORT-tracked BFC arenas | **1.01 MiB** |
| **Whole-process WDDM growth** | **234.01 MiB** |

```text
234.01 MiB = 168.00 + 14.27 + 3.88 + 46.85 + 1.01
```

The underlying arena-capacity counters are:

| Retained component | Post-initialization | Before shrink | Capacity growth |
|---|---:|---:|---:|
| Direct initializer reserves | 1169.68 MiB | 1169.68 MiB | 0 MiB |
| Decoder/model-session arena, normal BFC regions | 1.00 MiB | 169.00 MiB | **168.00 MiB** |
| ORT GenAI global device-allocator arena, normal BFC regions | 0 MiB | 65.00 MiB | **65.00 MiB** |
| **Total ORT-tracked CUDA allocation** | **1170.68 MiB** | **1403.68 MiB** | **233.00 MiB** |
| Other process-level WDDM dedicated memory | 257.45 MiB | 258.46 MiB | **1.01 MiB** |
| **Whole-process WDDM dedicated memory** | **1428.13 MiB** | **1662.14 MiB** | **234.01 MiB** |

The initializer reserves do not grow. The largest transient allocation in the
decoder/model-session arena occurs during generation and is exactly 128 MiB,
matching the legacy `MatMulNBits` dequantization workspace. Arena extension
granularity turns that and the other generation-time temporary allocations
into 168 MiB of additional retained BFC regions. At the end of generation, the
arena has returned to its original live allocation level; the added 168 MiB is
then completely free arena capacity. Calling `BFCArena::Shrink()` releases all
168 MiB and reduces process WDDM dedicated memory from 1662.14 MiB to
1494.14 MiB.

The second arena is ORT GenAI's process-global CUDA device allocator. During
`EnsureDeviceOrtInit()`, GenAI creates a trivial CUDA session solely to obtain
an allocator whose lifetime extends beyond an individual model or generator.
GenAI reuses that allocator for CUDA-resident model inputs and outputs,
scoring/sampling tensors, and KV-cache tensors. It is separate from the CUDA
arena owned by the real decoder `InferenceSession`.

This GenAI device-allocator arena retains 65 MiB. At the final checkpoint it
contains:

| Arena at the final pre-shrink checkpoint | BFC capacity | Requested live | Internal fragmentation | Free/reusable slack | Reclaimed by shrink |
|---|---:|---:|---:|---:|---:|
| Decoder/model-session arena | 169.00 MiB | 0.03 MiB | 0 MiB | 168.97 MiB | **168.00 MiB** |
| ORT GenAI global device-allocator arena | 65.00 MiB | 14.27 MiB | 3.88 MiB | 46.85 MiB | **0 MiB** |

The decoder/model-session arena already had 1 MiB of capacity at
post-initialization. Generation expanded it by 168 MiB. At the final checkpoint
it contained the same approximately 0.03 MiB live allocation and 168.97 MiB
slack. Consequently, the entire additional 168 MiB was free by then and could
be released as complete unused regions. The remaining original 1 MiB region
was retained.

This arena holds GenAI-owned generation state and reusable capacity. Its
diagnostic maximum single allocation is 31.77 MiB. The allocator counters do
not identify individual tensors, so the 14.27 MiB live portion cannot be
separated further into KV cache, logits, and other generator buffers without
tensor-level instrumentation. The 65 MiB does not grow across requests and
cannot be reclaimed while its regions still contain live allocations. The
remaining 1.01 MiB is outside ORT's tracked BFC arenas and falls within
CUDA/GenAI/WDDM process-accounting overhead.

`Growth after initialization` is:

```text
post_generation dedicated GPU memory - post_initialize dedicated GPU memory
```

It is retained process-level WDDM growth, not the sum of allocation requests
made by the runtime. It covers the three script warmups plus all 130 measured
translations. Foundry's `post_initialize` checkpoint is taken after Foundry's
internal model-load warmup, whereas llama.cpp does not perform that same
internal warmup. The milestones are therefore useful for process lifecycle
accounting but do not isolate one identical kernel boundary.

The larger 234.01 MiB Foundry growth with direct initializers does not negate
the optimization. Direct initializer allocation reduces Foundry's
post-initialization footprint by 876.01 MiB relative to default BFC placement
(`2304.14 - 1428.13`), while post-generation memory falls by 718.00 MiB
(`2380.14 - 1662.14`). Part of the saved initialization footprint is therefore
consumed by transient generation state, but the final process footprint
remains much lower.

### Static workspace-preallocation A/B under the matched protocol

The matched protocol was also run with default BFC initializer placement and
only this session option added:

```text
session.enable_static_workspace_preallocation=1
```

This option is disabled by default. Merely compiling the workspace declaration
and lookup code does not enable it. A bounded trace proved both expected
phases for the 8 MiB legacy `MatMulNBits` workspace:

```text
[workspace_plan_lookup] state=frame_returned_null ... planned_bytes=8388608
[legacy_workspace_check] source=scratch_fallback ... required_bytes=8388608
[workspace_plan_lookup] state=success ... planned_bytes=8388608
[legacy_workspace_check] source=preallocated ... required_bytes=8388608
```

The first run of a previously unseen memory-pattern signature records the
pattern and therefore falls back to scratch. A later cache hit retrieves the
planned workspace from the memory-pattern backing buffer. In the bounded first
512 execution-frame trace, 23 lookups recorded new patterns and 489 were cache
hits. These are execution-frame counts, not translation-request counts.

Performance was measured in a separate untraced run:

| Foundry metric | Static preallocation disabled | Static preallocation enabled | Difference |
|---|---:|---:|---:|
| Successful requests | 130 | 130 | 0 |
| Average latency | 89.86 ms | 89.30 ms | -0.56 ms |
| P50 / P90 | 85 / 121 ms | 84 / 116 ms | -1 / -5 ms |
| Working-set maximum | 1107.62 MiB | 1082.65 MiB | -24.97 MiB |
| Private-memory maximum | 3637.15 MiB | 3635.55 MiB | -1.60 MiB |
| Process dedicated GPU maximum | 2380.14 MiB | 2378.14 MiB | -2.00 MiB |
| Process shared GPU maximum | 76.73 MiB | 76.73 MiB | 0 MiB |
| Heuristic quality score | 89.27 | 89.50 | +0.23 |

The enabled run retained 2378.14 MiB at requests 1, 10, 25, 50, 100, and
130, so it did not accumulate GPU memory. Static workspace preallocation is
functionally active, but its process-level effect is small for this workload:
the workspace is only 8 MiB and the scratch allocation already reuses BFC
capacity. The observed 0.56 ms average-latency and 2 MiB GPU-memory differences
are small enough to be run-to-run noise rather than evidence of a material
performance or memory improvement.

## Result Artifacts

Default BFC initializers:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_SCRATCH_SHRINK_VS_LLAMA_WARMUP1_SAMPLE1_VALID
```

`kSameAsRequested`:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_SCRATCH_KSAMEASREQUESTED_SHRINK_VS_LLAMA_WARMUP1_SAMPLE1_VALID
```

Device-allocator initializers:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_SCRATCH_DEVICE_INITIALIZERS_SHRINK_VS_LLAMA_WARMUP1_SAMPLE1
```

Device-allocator initializers, 130-sample stability run:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_SCRATCH_DEVICE_INITIALIZERS_SHRINK_VS_LLAMA_WARMUP1_SAMPLE130_VALID
```

Protocol-matched default BFC initializers:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_DEFAULT_INITIALIZERS_WARMUP3_SAMPLE130_VALID
```

Protocol-matched device-allocator initializers:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_DEVICE_INITIALIZERS_WARMUP3_SAMPLE130_CORRECTED
```

Protocol-matched static workspace preallocation, untraced performance run:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_STATIC_WORKSPACE_PREALLOC_UNTRACED_WARMUP3_SAMPLE130_VALID
```

Protocol-matched static workspace preallocation, traced path-verification run:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_STATIC_WORKSPACE_PREALLOC_WARMUP3_SAMPLE130_VALID
```

The single-sample diagnostic directories contain:

- `results.json`, `results.csv`, and `summary.md`;
- Foundry and llama.cpp stdout/stderr;
- the exact experimental `genai_config.json`;
- source-build hashes; and
- an `ab-comparison.json` with derived allocator accounting.

The protocol-matched directories contain `comparison.json`,
`comparison.csv`, `summary.md`, separate `q4_raw` and `foundry_raw`
subdirectories, `genai_config.effective.json`, and
`source-build-manifest.json`.

The active model configuration was restored after every experiment. Its
SHA-256 is:

```text
01EEF87C41EE4D059BBAAB16700F9F506C7309BE4A31D43AE185AABD115233DE
```
