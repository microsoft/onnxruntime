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

## Real Model Input Shapes

The 130 measured Foundry requests do not use the synthetic sequence length 64
from `hy_mt2_benchmark_test.cc`. Tokenizing the exact chat template,
translation instruction, and source text with ORT GenAI's model tokenizer
gives prompt lengths from 36 to 43 tokens:

| Prompt tokens `P` | Sample count | Physical shape of each KV key/value tensor |
|---:|---:|---|
| 36 | 2 | `[1, 4, 136, 128]` |
| 37 | 32 | `[1, 4, 137, 128]` |
| 38 | 45 | `[1, 4, 138, 128]` |
| 39 | 26 | `[1, 4, 139, 128]` |
| 40 | 15 | `[1, 4, 140, 128]` |
| 41 | 7 | `[1, 4, 141, 128]` |
| 42 | 2 | `[1, 4, 142, 128]` |
| 43 | 1 | `[1, 4, 143, 128]` |

The mean prompt length is 38.42 tokens. These lengths include the complete
Chinese translation instruction and the chat-template begin, user, and
assistant tokens.

Let `P` be the prompt length and `k` be a one-based decode model-call index.
The real decoder inputs are:

| Model tensor | Prefill model call | Decode model call `k` |
|---|---|---|
| `input_ids` INT64 | `[1, P]` | `[1, 1]` |
| `attention_mask` INT64 | `[1, P]` | `[1, P + k]` |
| Each past key/value FP16 tensor | `[1, 4, P + 100, 128]` | `[1, 4, P + 100, 128]` |
| Logical past length represented in the shared KV buffer | 0 | `P + k - 1` |
| `logits` FP16 output | `[1, P, 120818]` | `[1, 1, 120818]` |

There are 32 layers and one key plus one value tensor per layer, for 64 KV
input tensors. `past_present_share_buffer=true`, so the physical KV shape is
allocated once to:

```text
P + max_completion_tokens = P + 100
```

The shape does not grow during decoding. Only the logical past length and
attention-mask length advance. Present outputs alias the same shared backing
buffers, so the past and present shapes must not be counted as separate KV
allocations. Across these prompts, the complete 64-tensor FP16 KV allocation
is approximately 8.50 to 8.94 MiB.

A direct CUDA input trace for measured sample 1 (`P = 42`) confirmed:

```text
Prefill:
  input_ids:            [1, 42]
  attention_mask:       [1, 42]
  each KV key/value:    [1, 4, 142, 128]

First decode model call:
  input_ids:            [1, 1]
  attention_mask:       [1, 43]
  each KV key/value:    [1, 4, 142, 128]
```

Re-encoding the stored translation text reconstructs 5 to 18 generated
content tokens for the default-BFC run and 5 to 19 for the device-initializer
run. These sampled output lengths change the number of decode model calls,
but not the per-call tensor-shape rule above. The terminal EOS token is not
included in the stored translation text. This ONNX graph does not expose a
separate `position_ids` input.

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

## Three-Configuration Memory Summary

The process rows below come from the primary paired protocol runs. The ORT
allocator rows come from matched diagnostic repeats that captured
`post_initialize` and the end-of-workload `before_shrink` state. llama.cpp
does not use ORT's allocators, so allocator-specific rows are not available
for that process.

For Foundry, the only configuration difference is:

```text
Default BFC initializers:       session.use_device_allocator_for_initializers=0
Device-allocator initializers:  session.use_device_allocator_for_initializers=1
```

| Metric | llama.cpp Q4 CUDA | Foundry default BFC initializers | Foundry device-allocator initializers |
|---|---:|---:|---:|
| **Configuration and performance** |  |  |  |
| Initializer/weight placement | llama.cpp native allocator | Decoder BFC arena | Direct CUDA reserves through `BFCArena::Reserve()` |
| Successful measured requests | 130 | 130 | 130 |
| Average latency | 54.31 ms | 89.86 ms | 89.60 ms |
| P50 / P90 latency | 53 / 67 ms | 85 / 121 ms | 84 / 120 ms |
| Heuristic quality score | 90.23 | 89.27 | 89.38 |
| **Whole-process memory lifecycle** |  |  |  |
| Pre-initialization dedicated GPU | 12.87 MiB | 12.87 MiB | 12.87 MiB |
| Weights loaded / post-initialization dedicated GPU | **1582.16 MiB** | **2304.14 MiB** | **1428.13 MiB** |
| Inference-time dedicated GPU growth | +34.00 MiB | +76.00 MiB | +234.01 MiB |
| Final retained dedicated GPU after inference | **1616.16 MiB** | **2380.14 MiB** | **1662.14 MiB** |
| Dedicated GPU peak | 1616.16 MiB | 2380.14 MiB | 1662.14 MiB |
| Final dedicated GPU gap versus llama.cpp | baseline | +763.98 MiB | **+45.98 MiB** |
| Shared GPU peak | 88.73 MiB | 76.73 MiB | 76.73 MiB |
| Process working-set maximum | 1505.88 MiB | 1107.62 MiB | 1014.33 MiB |
| Process private-memory maximum | 2251.24 MiB | 3637.15 MiB | 2854.38 MiB |
| **ORT allocator state after model initialization** |  |  |  |
| Direct initializer reserves | N/A - not ORT/BFC instrumented | 0 MiB | **1169.68 MiB** |
| Decoder/model-session BFC capacity | N/A - not ORT/BFC instrumented | **2049.00 MiB** | 1.00 MiB |
| ORT GenAI global allocator BFC capacity | N/A - not ORT/BFC instrumented | 0 MiB | 0 MiB |
| Total ORT-tracked CUDA allocation | N/A - not ORT/BFC instrumented | **2049.00 MiB** | **1170.68 MiB** |
| Other process-local dedicated GPU memory | N/A - allocator split unavailable | 255.14 MiB | 257.45 MiB |
| **ORT allocator state after inference, before diagnostic shrink** |  |  |  |
| Direct initializer reserves | N/A - not ORT/BFC instrumented | 0 MiB | **1169.68 MiB** |
| Decoder/model-session BFC capacity | N/A - not ORT/BFC instrumented | **2049.00 MiB** | 169.00 MiB |
| Decoder requested live memory | N/A - not ORT/BFC instrumented | **1169.71 MiB** | 0.03 MiB |
| Decoder internal fragmentation | N/A - not ORT/BFC instrumented | 13.43 MiB | 0 MiB |
| Decoder reusable BFC slack | N/A - not ORT/BFC instrumented | **865.86 MiB** | **168.97 MiB** |
| Decoder capacity reclaimable as complete unused regions | N/A - not ORT/BFC instrumented | 0 MiB | **168.00 MiB** |
| Largest decoder allocation observed | N/A - not ORT/BFC instrumented | 128.00 MiB | 128.00 MiB |
| ORT GenAI global allocator BFC capacity | N/A - not ORT/BFC instrumented | 65.00 MiB | 65.00 MiB |
| GenAI global allocator requested live memory | N/A - not ORT/BFC instrumented | 14.27 MiB | 14.27 MiB |
| GenAI global allocator internal fragmentation | N/A - not ORT/BFC instrumented | 3.88 MiB | 3.88 MiB |
| GenAI global allocator reusable BFC slack | N/A - not ORT/BFC instrumented | 46.85 MiB | 46.85 MiB |
| Total ORT-tracked CUDA allocation | N/A - not ORT/BFC instrumented | **2114.00 MiB** | **1403.68 MiB** |
| Other process-local dedicated GPU memory | N/A - allocator split unavailable | 266.14 MiB | 258.46 MiB |
| ORT-tracked growth after initialization | N/A - not ORT/BFC instrumented | +65.00 MiB | +233.00 MiB |
| Growth outside ORT-tracked allocators | N/A - allocator split unavailable | +11.00 MiB | +1.01 MiB |

The post-initialization checkpoint is the closest process-level measurement
to "weights loaded." It includes CUDA context and library state and, for
Foundry, its internal model-load warmup; it is not a weights-only
measurement.

The "other process-local" rows are residuals calculated as whole-process
WDDM dedicated memory minus the ORT allocator total from a matched diagnostic
repeat. They include CUDA libraries, CUDA context state, Foundry Local Core,
and normal run-to-run variation outside ORT's tracked allocators.

Default BFC placement does not mean that 2049 MiB of model data is live. The
decoder arena is already 2049 MiB after initialization and remains the same
size through inference. At the final checkpoint it contains 1169.71 MiB of
requested live allocations, 13.43 MiB of internal fragmentation, and
865.86 MiB of reusable slack. Long-lived model allocations prevent
`Shrink()` from releasing any complete region.

Direct initializer allocations bypass normal BFC regions and bins while
remaining tracked in `reserved_bytes`. This reduces the decoder arena from
2049 MiB to 1 MiB after initialization. Generation later expands it to
169 MiB, but 168 MiB is completely unused and reclaimable at the final
checkpoint.

Relative to llama.cpp:

```text
Default initializer gap: 2380.14 - 1616.16 = 763.98 MiB
Direct initializer gap:  1662.14 - 1616.16 =  45.98 MiB
Gap reduction:                                    718.00 MiB
```

All configurations reach their final retained dedicated-memory level by
measured request 1. llama.cpp and device-initializer Foundry remain exactly
flat through request 130. Default-BFC Foundry is flat through request 100 and
has the same value in the post-generation snapshot; its request-130 process
sample raced process exit and is unavailable. The measured growth therefore
establishes reusable generation state during the warmups and first request;
it does not accumulate across the 130 requests.

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
5. Default BFC placement retains 865.86 MiB of reusable decoder-arena slack
   because live model allocations prevent release of a complete region.
6. The 234.01 MiB device-initializer post-initialization growth consists of
   168 MiB of free decoder-arena regions, 65 MiB in the GenAI global
   allocator, and 1.01 MiB outside ORT-tracked BFC allocators.
7. The diagnostic shrink proves reclaimability but is not part of the selected
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

Per-sample Foundry input shapes:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
REBASED_PROTOCOL_DEFAULT_INITIALIZERS_WARMUP3_SAMPLE130_VALID\
hy_mt2_130_sample_model_input_shapes.csv
```

Default-BFC end-of-workload allocator attribution:

```text
C:\Users\lochi\hy_mt2_wayne_repro\benchmark\translate_benchmark_results\
LIFECYCLE_DEFAULT_BFC_ARENA_TRACE_REGISTERED_20260911
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
