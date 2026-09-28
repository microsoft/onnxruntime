# Phi-4-mini memory diagnostics and logits-allocation findings

Prepared September 28, 2026 from saved September 14-25 measurements.
This is a historical evidence contribution for the ORT `benchmark` branch,
not a production code change or a new benchmark of current upstream releases.
No measurements were rerun to prepare it.

## Scope and reading guide

The investigation started with an ORT GenAI versus llama.cpp comparison,
then used source-built ORT arena diagnostics to distinguish allocation
requests, live arena allocations, retained capacity and sampled GPU memory.

- [Earlier runtime comparison](evidence/earlier-runtime-comparison.csv):
  selected columns from all 45 original runs, not an equal-output-length
  or quantizer-matched runtime comparison.
- [Allocation before/after](evidence/memory-before-after.csv): all 12
  controlled measurement rows, including device and pinned-host snapshots.
- [Parity checks](evidence/parity-checks.json): first-token logits and
  generated-token comparison results and hashes.
- [Model/runtime manifest](evidence/model-manifest.json): pinned artifacts,
  effective configurations, saved input hashes and historical library hashes.
- [Source provenance](evidence/source-provenance.json): input report hashes,
  transformations and limits of this compact bundle.
- [Reproduction protocol](REPRODUCING.md): offline review versus fresh GPU
  replay, required external inputs and measurement precautions.

No model weights, prompt contents, raw logits/tokens, profiling traces,
binaries or machine-specific paths are included.

## 1. Earlier ORT GenAI versus llama.cpp comparison

The September 14 exploratory matrix used one NVIDIA A100-SXM4-80GB,
Linux, driver 580.105.08, Python 3.11.16, ORT GenAI CUDA 0.15.2,
ORT GPU 1.30.0 and llama-cpp-python 0.3.35. It ran three repetitions
per runtime/configuration/context in separate processes, with NVML
sampling requested at 10 ms. The output cap was 64 tokens.

ORT used Phi-4-mini ONNX INT4 RTN block 32; llama.cpp used Q4_K_M GGUF.
All offloadable llama.cpp layers were reported on GPU. ORT checked CUDA
configuration and initializer residency; this was not proof that every
shape/control node ran on CUDA. The ORT "device" variant changed
`session.use_device_allocator_for_initializers` from 0 to 1, not the
logits-allocation implementation.

**Every memory value in this table is sampled whole-device NVML MiB.**
Values are medians of three runs; TTFT is milliseconds. All 45 rows
completed. Context labels are capacity settings, not actual input lengths.

| Context | Actual input tokens | Runtime / initializer allocator | Output tokens | Inference peak MiB | TTFT ms | Decode tokens/s |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2048 | 1801 | ORT / baseline | 64 | 8656.12 | 477.373 | 202.338 |
| 2048 | 1801 | ORT / device | 64 | 8450.12 | 417.994 | 202.792 |
| 2048 | 1801 | llama.cpp | 37 | 4332.12 | 462.766 | 165.372 |
| 4096 | 3596 | ORT / baseline | 64 | 11984.12 | 541.534 | 177.093 |
| 4096 | 3596 | ORT / device | 64 | 12288.12 | 553.829 | 175.501 |
| 4096 | 3596 | llama.cpp | 64 | 4636.12 | 758.893 | 167.995 |
| 8192 | 7189 | ORT / baseline | 64 | 19412.12 | 856.935 | 158.647 |
| 8192 | 7189 | ORT / device | 64 | 18952.12 | 915.855 | 160.187 |
| 8192 | 7189 | llama.cpp | 59 | 5246.12 | 1425.944 | 149.190 |
| 16384 | 14376 | ORT / baseline | 64 | 33104.12 | 1584.027 | 134.683 |
| 16384 | 14376 | ORT / device | 64 | 33288.12 | 1609.984 | 132.732 |
| 16384 | 14376 | llama.cpp | 40 | 6838.12 | 3349.070 | 119.318 |
| 32768 | 28747 | ORT / baseline | 64 | 60626.12 | 3311.018 | 98.711 |
| 32768 | 28747 | ORT / device | 64 | 60936.12 | 3492.421 | 100.878 |
| 32768 | 28747 | llama.cpp | 45 | 10022.12 | 9802.428 | 88.063 |

This motivated allocation attribution, not a general ranking of runtimes:
different quantizers, early EOS and unequal completion counts prevent a
strict equal-work comparison. These runs do not establish cross-runtime
output quality or parity. Whole-device readings can include unrelated
device activity. Exact build commits and complete effective settings for
this older matrix were not captured as fully as for the later experiment;
do not retroactively assign the later manifest's settings to these runs.

## 2. Allocation owner and root cause

In the historical GenAI multi-token FP16/BF16 logits-extraction path:

1. The model produced logits covering the prompt.
2. Extraction selected the last relevant token for each batch/beam.
3. GenAI eagerly allocated the FP32 destination using the full prompt shape.
4. The existing Cast helper detected the shape mismatch, released that
   destination and allocated the much smaller last-token destination.

At 28,747 input tokens, the redundant request was:

```text
1 batch x 28,747 positions x 200,064 vocabulary entries x 4 bytes
= 23,004,959,232 bytes (21.425 GiB)

Required last-token destination:
1 x 1 x 200,064 x 4 = 800,256 bytes
```

The historical correction removed the eager destination allocation and
let the existing Cast helper allocate/reuse compatible storage. It did
not remove the model's raw prompt logits, change numerical operations,
enable chunking or change allocator policy.

Two instrumented unchunked captures, at 1,801 and 28,747 saved input tokens,
recorded successful large creation, Cast shape mismatch, completed owner
reset and small replacement. Returned pointers, requested sizes and BFC
address ranges correlated the long-input request with a 32 GiB arena
extension. The evidence establishes a GenAI allocation call site, not a
CUDA kernel/workspace owner, arena allocation ID or exact cudaFree event.
Releasing the owning OrtValue does not imply returning backing memory to CUDA.

**Upstream status:** the later inspected GenAI main at
`7f750ddbcbfb4c93d99d23d595c40294904f1b06` already included
[PR #2607](https://github.com/microsoft/onnxruntime-genai/pull/2607),
commit `dd7ada04695e424029c0ba7328e6434e751c5905`, which changes the
allocation shape from `shape_` to `shape_last`. This report does not
claim the historical oversized allocation is still missing a fix there.
No GenAI production patch is included in this ORT contribution.

## 3. Controlled historical allocation-fix results

Both variants used ORT benchmark commit
`94e76459dc9a759888707aa430d51a95d878da8f` and GenAI base
`ed5f4e87147731e5b07810f9f5c90103b3603cdf`. "Patched" means that
historical GenAI build with the redundant allocation removed, not a newer
release. Actual baseline/patched library hashes are in the manifest.

The ONNX model revision is `fc04c8f93df696602fd9f300a30d1bf2e3081347`,
variant `gpu/gpu-int4-rtn-block-32`. Environment: one A100-SXM4-80GB,
CUDA runtime 13.0.48, cuDNN 9.12.0, driver 580.105.08.

Settings: batch 1, beams 1, chunk size 0, greedy generation
(`do_sample=false`, top_k=1, top_p=1, temperature=1), shared past/present
cache, default arena and initializer device-allocator option 0.
Saved input lengths were 1,801 and 28,747; max_length was 1,865 and
28,811 respectively. Both variants generated 64 tokens. Full effective
settings and input hashes are recorded in the manifest.

| Nominal context | Variant | Process NVML peak MiB | Whole-device NVML peak MiB | BFC capacity after generator cleanup MiB |
| ---: | --- | ---: | ---: | ---: |
| 2048 | Baseline | 8002 | 8778.125 | 7555 |
| 2048 | Patched | 5954 | 6730.125 | 5507 |
| 32768 | Baseline | 59848 | 60624.125 | 59401 |
| 32768 | Patched | 27082 | 27858.125 | 26635 |

These are independent column medians from three fresh-process repetitions
per condition, not a single representative run. The long-input process
peak reduction is 32,766 MiB, while the redundant request is 21.425 GiB.
Arena growth granularity/reuse can make these differ; the entire peak
reduction is not attributed exclusively to that tensor.

Definitions:

- Process NVML: GPU memory charged to the measured process.
- Whole-device NVML: all sampled memory used on the GPU.
- BFC capacity: backing memory retained by the arena.
- BFC live: bytes assigned to live arena allocations.
- BFC slack: retained capacity minus live allocations, not automatically
  a leak or immediately reclaimable GPU memory.
- MiB and GiB use powers of 1024; they are not decimal MB/GB.

At nominal 32K, median post-cleanup live bytes corresponded to 3,288.39 MiB
for both variants, while capacity fell from 59,401 to 26,635 MiB. The model
remained alive. All recorded pinned-host capacity/live/slack values were
zero; this does not mean the process used no host memory.

### Correctness

Four Phase A captures compared original and allocation-fixed unchunked
execution at both input lengths. All 200,064 FP32 first-token logits
(800,256 bytes per vector) matched byte-for-byte, and all 64 generated IDs
matched. All 12 Phase B runs matched their corresponding Phase A token
reference. Complete saved Phase A logits were also rechecked offline
while preparing this report; no inference was run.

This demonstrates preservation of the original unchunked behavior on
those inputs. It is not proof of absolute model quality, cross-runtime
parity or chunked/unchunked equivalence. Compact hashes alone are not
substitutes for comparing raw captures.

### Measurement and timing limits

Phase B used no diagnostic GetLogits copying, verbose attribution tracing,
profiling or shrinking. Arena registration was enabled on both variants;
snapshots were outside the timed interval. NVML sampling requested 10 ms,
and only queries wholly inside inference contributed to peaks. Complete
query timing and live library identities were retained in the raw evidence,
which is not distributed here. No foreign GPU process was observed in the
controlled runs, but sampling cannot rule out activity between observations.

Saved timing summaries reported median TTFT of 3,133.19 versus 2,574.37 ms
at nominal 32K, and decode rates of 100.70 versus 100.42 tokens/s.
TTFT ran from before AppendTokens through first-token CUDA completion,
excluding initialization; decode covered the remaining 63 tokens.
These are historical observations, not a causal profiling result or a
performance promise: there were three repetitions, fixed baseline-then-
patched ordering and no inference warmup. Timing summaries come from the
hashed source report; the compact memory CSV does not contain timing rows.
Do not combine these timings with the earlier package-runtime matrix.

## 4. Separate correctness work and unfinished questions

The chunking investigation identified a distinct ORT fused GQA path
that used cache offsets rather than supplied explicit rotary positions.
At absolute position 4096, the historical Phi graph supplied position
8192 but that path consumed row 4096. A separately tested guard corrected
the new-token index. No such guard is added by this documentation PR.

Correcting the index did not establish full-model chunked/unchunked
parity. In the bounded 4097-token historical confirmation, chunk512 versus
unchunked logits still differed (maximum absolute difference 17.484375).
Sampled old K entries retained their append-time treatment. Intended
LongRoPE model/export transition semantics and cached-history handling
remain unresolved; re-rotating stored K alone is not assumed sufficient.
Ownership or deferral requires team agreement.

Other unfinished work:

- The proposed shrink-after-request/prefill/decode/cleanup matrix and
  concurrency safety are not established as complete. No production shrink
  policy is recommended.
- Additional model families, larger contexts, batch/concurrent serving
  behavior and output quality were not established by these focused results.
- Small batch/beam unit-test coverage does not constitute multi-request
  full-model performance validation.
- Causal TTFT profiling and performance on current upstream remain follow-ups.

This report preserves useful historical findings without claiming that the
entire broader investigation is complete or blocking a separate narrow
production correctness review on unrelated profiling/model experiments.
