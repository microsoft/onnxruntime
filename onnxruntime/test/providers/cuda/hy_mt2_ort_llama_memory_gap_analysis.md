# HY-MT2 ORT Versus llama.cpp Memory Gap Analysis

## Status and Scope

This document uses one benchmark scenario as the source of truth:

- coherent source-built ORT and ORT GenAI binaries;
- RTX 5090 Laptop GPU, driver 610.62, SM120;
- CUDA 12.8 and cuDNN 9.5;
- llama.cpp tag `b10156`;
- three script warmups;
- 130 matched translation samples;
- the same strict English-only prompt and generation settings;
- serial, isolated llama.cpp and Foundry processes; and
- no arena shrink before or during measurement.

Older one-warmup/one-sample shrink diagnostics are intentionally excluded.
They used different prompt, generation, and shrink boundaries and must not be
combined with the results in this document.

The runtime versions used by the benchmark are:

- ORT branch: `chilo/static-workspace-preallocation-test`
- ORT commit: `e648c9c2fe684082d0c1b786049215ee2dbbd570`
- ORT GenAI commit: `ed5f4e87147731e5b07810f9f5c90103b3603cdf`
- CUDA EP SHA-256:
  `F6785E36707EAD7DB10A38EEDE4CAF7555AED7D43F45596413D6AAC08125AB9D`
- model config restored SHA-256:
  `01EEF87C41EE4D059BBAAB16700F9F506C7309BE4A31D43AE185AABD115233DE`

## Selected Benchmark Protocol

Each engine runs in a fresh process. llama.cpp completes and exits before
Foundry starts, avoiding GPU and CPU contention.

The protocol is:

1. Start the translation process.
2. Capture a pre-initialization process-local WDDM snapshot.
3. Initialize the model.
4. Capture the post-initialization snapshot.
5. Run three script warmups.
6. Run the same 130 measured translation requests.
7. Capture retained-memory checkpoints at requests 1, 10, 25, 50, 100, and
   130.
8. Capture the post-generation snapshot.
9. Stop the process.

The generation settings are temperature 0.7, top-p 0.6, top-k 20, repetition
penalty 1.05, and a 100-token maximum. Foundry also performs its internal
model-load warmup before the harness records `post_initialize`.

## Memory Accounting

All values use binary MiB (`1 MiB = 1,048,576 bytes`).

| Term | Meaning |
|---|---|
| `total_allocated_bytes` | All CUDA memory tracked by one allocator: direct reserves plus BFC backing regions |
| `reserved_bytes` | Direct allocations made through `Reserve()` |
| BFC region capacity | `total_allocated_bytes - reserved_bytes` |
| Requested live memory | Bytes requested by allocations that are still live |
| Internal fragmentation | Allocated chunk capacity above requested live bytes |
| BFC slack | Free reusable capacity inside retained BFC regions |
| WDDM dedicated memory | Total dedicated GPU memory attributed to the whole translation process PID |

WDDM dedicated memory includes ORT, ORT GenAI, Foundry Local Core, CUDA
contexts, CUDA libraries, model weights, KV cache, activations, workspaces,
and library-internal buffers. It is therefore larger than the sum of
ORT-tracked allocator totals.

Two CUDA BFC allocators are relevant:

| Allocator | Owner and role |
|---|---|
| Decoder/model-session allocator | Owned by the real decoder `InferenceSession`; holds direct initializer reserves and graph activation/workspace regions |
| ORT GenAI global device allocator | Obtained through GenAI's trivial CUDA session; used for GenAI-owned device inputs, outputs, scoring/sampling tensors, and KV-cache tensors |

## Protocol-Matched Results

### Default initializer placement

| Metric | llama.cpp Q4 | Foundry INT4 |
|---|---:|---:|
| Successful requests | 130 | 130 |
| Average latency | 54.31 ms | 89.86 ms |
| P50 / P90 | 53 / 67 ms | 85 / 121 ms |
| Process CPU average | 3.76% | 3.66% |
| Working-set maximum | 1505.88 MiB | 1107.62 MiB |
| Private-memory maximum | 2251.24 MiB | 3637.15 MiB |
| Process dedicated GPU maximum | 1616.16 MiB | 2380.14 MiB |
| Process shared GPU maximum | 88.73 MiB | 76.73 MiB |
| Heuristic quality score | 90.23 | 89.27 |

All 130 Foundry requests succeeded with no empty output or Chinese leakage.

### Device-allocator initializer A/B

The only allocator change is:

```text
session.use_device_allocator_for_initializers=1
```

| Foundry metric | Default BFC initializers | Device-allocator initializers | Difference |
|---|---:|---:|---:|
| Average latency | 89.86 ms | 89.60 ms | -0.26 ms |
| P50 / P90 | 85 / 121 ms | 84 / 120 ms | -1 / -1 ms |
| Working-set maximum | 1107.62 MiB | 1014.33 MiB | -93.29 MiB |
| Private-memory maximum | 3637.15 MiB | 2854.38 MiB | -782.77 MiB |
| Process dedicated GPU maximum | 2380.14 MiB | 1662.14 MiB | **-718.00 MiB** |
| Heuristic quality score | 89.27 | 89.38 | +0.11 |

Relative to llama.cpp:

```text
Default initializer gap: 2380.14 - 1616.16 = 763.98 MiB
Direct initializer gap:  1662.14 - 1616.16 =  45.98 MiB
Gap reduction:                                    718.00 MiB
```

Direct initializer allocations use `BFCArena::Reserve()`, which calls the
underlying CUDA allocator directly. ORT tracks these allocations in
`reserved_bytes`, but they bypass normal BFC regions and bins. They remain
live for the model session and cannot be reused for transient requests.

This separation prevents long-lived weights from pinning large BFC regions
that are also expected to serve transient graph allocations.

## Initialization Versus Inference-Time Growth

| Dedicated GPU memory | llama.cpp | Foundry, default initializers | Foundry, device initializers |
|---|---:|---:|---:|
| Post-initialization | 1582.16 MiB | 2304.14 MiB | 1428.13 MiB |
| Post-generation | 1616.16 MiB | 2380.14 MiB | 1662.14 MiB |
| Growth after initialization | 34.00 MiB | 76.00 MiB | 234.01 MiB |

With direct initializer allocation, Foundry starts 154.03 MiB below
llama.cpp, grows 200.01 MiB more during the subsequent workload, and finishes
45.98 MiB above llama.cpp.

Both engines reach their final retained dedicated-memory level by measured
request 1 and remain flat through request 130:

| Request | llama.cpp | Foundry with device initializers |
|---:|---:|---:|
| 1 | 1616.16 MiB | 1662.14 MiB |
| 10 | 1616.16 MiB | 1662.14 MiB |
| 25 | 1616.16 MiB | 1662.14 MiB |
| 50 | 1616.16 MiB | 1662.14 MiB |
| 100 | 1616.16 MiB | 1662.14 MiB |
| 130 | 1616.16 MiB | 1662.14 MiB |

The 234.01 MiB therefore appears while establishing generation state and
running the three warmups. It does not accumulate across measured requests.

## Allocation-Level Attribution of the 234.01 MiB

A diagnostic repeat used the same selected protocol and delayed the arena
checkpoint until completed call 133:

```text
3 warmups + 130 measured requests = 133 completed calls
```

The diagnostic first captured the unmodified `before_shrink` state. It then
called `Shrink()` only to test which retained regions were fully unused and
reclaimable. The shrink result is attribution evidence, not a second benchmark
scenario.

The growth is:

| Source of new retained dedicated memory | Growth |
|---|---:|
| Generation-time expansion of decoder/model-session arena, free at final checkpoint | **168.00 MiB** |
| GenAI global allocator requested live memory | **14.27 MiB** |
| GenAI global allocator internal fragmentation | **3.88 MiB** |
| GenAI global allocator reusable slack | **46.85 MiB** |
| CUDA/GenAI/WDDM growth outside ORT-tracked BFC allocators | **1.01 MiB** |
| **Whole-process WDDM growth** | **234.01 MiB** |

```text
234.01 MiB = 168.00 + 14.27 + 3.88 + 46.85 + 1.01
```

The corresponding allocator capacities are:

| Component | Post-initialization | End of selected workload | Growth |
|---|---:|---:|---:|
| Direct initializer reserves | 1169.68 MiB | 1169.68 MiB | 0 MiB |
| Decoder/model-session BFC regions | 1.00 MiB | 169.00 MiB | **168.00 MiB** |
| ORT GenAI global allocator BFC regions | 0 MiB | 65.00 MiB | **65.00 MiB** |
| **Total ORT-tracked CUDA allocation** | **1170.68 MiB** | **1403.68 MiB** | **233.00 MiB** |
| Other process-local dedicated memory | 257.45 MiB | 258.46 MiB | **1.01 MiB** |
| **Whole-process WDDM dedicated memory** | **1428.13 MiB** | **1662.14 MiB** | **234.01 MiB** |

### Decoder/model-session arena

The largest transient generation-time allocation was exactly 128 MiB,
matching the legacy `MatMulNBits` dequantization workspace. Together with
other temporary allocations and BFC growth granularity, it expanded this
arena by 168 MiB.

At the final checkpoint:

```text
BFC capacity:       169.00 MiB
Requested live:       0.03 MiB
Free BFC capacity:  168.97 MiB
```

The arena already had 1 MiB of capacity after initialization. The complete
additional 168 MiB consisted of unused regions at the final checkpoint.
`Shrink()` released those regions and reduced process WDDM dedicated memory
from 1662.14 MiB to 1494.14 MiB. The original 1 MiB region remained.

### ORT GenAI global device allocator

At the final checkpoint:

```text
BFC capacity:           65.00 MiB
Requested live:         14.27 MiB
Internal fragmentation: 3.88 MiB
Reusable slack:         46.85 MiB
```

This arena contains GenAI-owned generation state and reusable capacity.
Because its regions still contain live allocations, `Shrink()` cannot release
them as complete regions. The allocator counters do not identify individual
tensors, so separating the 14.27 MiB live portion into KV cache, logits, and
other generator buffers requires tensor-level instrumentation.

## Static Workspace Preallocation

The selected protocol was also run with:

```text
session.enable_static_workspace_preallocation=1
```

| Foundry metric | Disabled | Enabled | Difference |
|---|---:|---:|---:|
| Average latency | 89.86 ms | 89.30 ms | -0.56 ms |
| P50 / P90 | 85 / 121 ms | 86 / 121 ms | +1 / 0 ms |
| Working-set maximum | 1107.62 MiB | 1108.03 MiB | +0.41 MiB |
| Private-memory maximum | 3637.15 MiB | 3635.55 MiB | -1.60 MiB |
| Process dedicated GPU maximum | 2380.14 MiB | 2378.14 MiB | -2.00 MiB |
| Heuristic quality score | 89.27 | 89.50 | +0.23 |

A bounded trace confirmed that the first unseen shape uses scratch and a
matching cached shape later retrieves the planned workspace. The process-level
differences are small enough to be normal run-to-run variation. Dynamic
scratch already reuses BFC capacity, so static workspace preallocation does
not materially change retained process memory for this workload.

## Conclusions

1. The protocol-matched 130-sample scenario is the only benchmark scenario
   used for numerical conclusions.
2. Device-allocator initializers reduce Foundry dedicated GPU memory by
   718 MiB without a measurable latency or quality regression.
3. The remaining Foundry-versus-llama.cpp gap is 45.98 MiB.
4. Foundry does not accumulate dedicated GPU memory across the 130 requests.
5. The 234.01 MiB post-initialization growth consists of 168 MiB of free
   decoder-arena regions, 65 MiB in the GenAI global allocator, and 1.01 MiB
   outside ORT-tracked BFC allocators.
6. The diagnostic shrink proves reclaimability but is not part of the selected
   benchmark measurement.

## Result Artifacts

Default initializer protocol:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_DEFAULT_INITIALIZERS_WARMUP3_SAMPLE130_VALID
```

Device-initializer protocol:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_DEVICE_INITIALIZERS_WARMUP3_SAMPLE130_CORRECTED
```

Fresh device-initializer repeat:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
LIFECYCLE_DEVICE_INITIALIZERS_20260904_1453
```

End-of-workload arena attribution:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
LIFECYCLE_DEVICE_INITIALIZERS_ARENA_TRACE_REGISTERED_20260904_1515
```

Static-workspace untraced performance run:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_STATIC_WORKSPACE_PREALLOC_UNTRACED_WARMUP3_SAMPLE130_VALID
```

Static-workspace path-verification trace:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_STATIC_WORKSPACE_PREALLOC_WARMUP3_SAMPLE130_VALID
```
