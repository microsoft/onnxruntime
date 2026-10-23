# Native WebGPU queue uploads

ONNX Runtime automatically uses WebGPU `Queue.WriteBuffer` for eligible
CPU-to-GPU uploads up to and including **1 MiB (1,048,576 bytes)**. No
environment-variable toggle or provider option is required.

The upload API is byte-oriented and writes at **destination offset zero**.
An upload may cover a prefix of a larger buffer. Queue uploads require a byte
count divisible by four and no larger than 1 MiB, an unmapped destination with
`CopyDst` usage, and ordinary (non-capture, non-replay) command recording. No new
tensor element-type restriction is introduced, and the host source pointer
need not be aligned.
Zero-byte uploads copy no data; mapped destinations are still unmapped.

Mapped-at-creation destinations retain their direct host-copy/unmap path.
Larger transfers, unaligned byte counts, and graph capture/replay retain the
existing staging path. That path rounds the GPU copy size up to four bytes;
padding bytes are not part of the logical tensor and remain unspecified.
Queue uploads do not add support for nonzero destination offsets or unaligned
GPU copy offsets.

## Ordering and lifetime

Before a queue upload, ORT submits any work already recorded on the caller's
command timeline. This is necessary when a previously recorded dispatch or
copy still reads the destination: recording a command does not yet order it
before `WriteBuffer`. Subsequent commands submitted to the same queue observe
the upload in queue order.

`WriteBuffer` copies the host source data before returning. The caller may
immediately modify or release the host allocation. Dawn owns any staging
allocation and retains resources until its transfer completes. The
destination still follows the existing ORT allocation and stream ownership
rules; do not destroy it or race writes to the same tensor.
Each session's recording timeline still requires serialization. Independent
sessions sharing tensors still require caller-provided dependency ordering;
the flush and queue write are not a device-wide atomic transaction.

Submitting prior commands is **not a CPU wait for GPU completion**. Queue
uploads remove no readback, completion, error-scope, device-loss, or session
synchronization checks. Queue-write validation failures use the existing
WebGPU error-reporting path; they are not silently retried as staging copies.

## Why this can help

The single-upload staging path creates a fresh mapped staging buffer, copies
host bytes, unmaps it, records `CopyBufferToBuffer`, and finishes/submits an
encoder for every upload. `WriteBuffer` lets Dawn use its dynamic uploader,
reuse staging allocations, select backend-specific host-visible fast paths,
and batch pending transfers with later queue work. It may still submit work
internally, so fewer ORT encoders do not imply an identical reduction in
backend submissions.

The 1 MiB limit is conservative: the original A100/Vulkan measurements consistently
improved both host and completion-fenced latency through that size, with and
without preceding GPU reads. An uncapped queue-write experiment regressed
completion-fenced transfers at 4 MiB + 4 bytes. Sizes between 1 MiB and that
regression were not validated, so they remain on staging rather than
extrapolating the small-upload improvement.
The limit is also below the pinned Dawn dynamic uploader's 4 MiB ring size;
larger uploads can require dedicated staging rather than allocation reuse.

`WriteBuffer` is a core WebGPU API, not a Vulkan-only extension. Its queue and
host-source lifetime guarantees apply to Dawn's Vulkan, D3D12, and Metal
backends. A100/Vulkan correctness has been exercised locally; macOS/Metal and
Windows/D3D12 native CI must still validate default routing, ordering,
threshold boundaries, and concurrency. Backend performance, memory pressure,
and external Dawn hosts require platform-specific validation before raising
the size limit.

Correctness coverage is part of `onnxruntime_provider_test`, including the
`WebGpuQueueUploadTest` cases, typed byte/lifetime tests,
deferred-dispatch and copy ordering, and independent concurrent transfers.
These run with the native WebGPU test suites in Windows and macOS CI.
Performance benchmarks and model-generation comparisons are not CI tests.

## Interaction with batched input uploads

Main includes #33491's batched CPU-to-GPU input transfers. Eligible lists with
at least two inputs use `UploadBatch`, packing copies into staging chunks of
up to 16 MiB. They do not call the single-upload fast path, even when each
input is smaller than 1 MiB. Queue uploads neither replace nor duplicate that
batching optimization. Oversized individual inputs retain the staging
fallback.

Single-input transfers, streamless plugin copies, and lists that cannot be
batched can still reach `Upload`. Their eligible small transfers benefit
independently from `WriteBuffer`. A batch followed by a single upload, or the
reverse, must preserve any recorded reads before overwriting the destination.
`WebGpuDataTransferTest.BatchedAndSingleUploadsPreserveRecordedReaders`
covers both transitions and immediate reuse of the host source.

## Main-based A100 validation (2026-10-23)

### Controlled comparison

The baseline is main **`7ec270791d92a789db39c868e565d05ce7d1654b`**,
including #33491 (`07ac1719bf07e0f56ab8f63ef56e0e9ea6263009`). The after revision
is merge **`a39796931090f04eb95344022dd44b9bb0bd6e9e`**, which combines that main
with PR #33551's `9485746837564924892c3ab8cb5f1289bc5dc6e9`. The additional
mixed-upload regression test does not change production code.

Main advanced during measurement to
`db2039d05cb92218ec5567d39b5a61af308fe4b1`, adding only a macOS CI timeout
increase (#33564). Merge `5380f451e49de41eeaa7fc69fc745812e3b2f3a4` also
integrates that revision. A full diff confirms that runtime sources,
dependencies, and build inputs are identical to the measured main
revision; the CI-only change does not require repeating the comparison.

Both variants used the same NVIDIA A100-SXM4-80GB, Vulkan backend, NVIDIA
580.159.04 driver, pinned dependencies, Release/Ninja configuration, and
disabled LTO and leak checking. Main was built from a clean detached
worktree. After artifacts recompiled `buffer_manager.cc` and linked it
against the exact same main objects and archives; this is the only
production translation unit that differs. Changed test objects were
rebuilt separately. Native static WebGPU was used for microbenchmarks;
shared WebGPU plugins were used for generation and plugin tests.

Generation used the same fresh main CPU ORT core for both plugins, not an
older experimental ORT binary. A WebGPU-enabled Python build on main
failed to import with an unresolved
`WebGpuProviderFactoryCreator::Create(ConfigOptions const&)` symbol; using
the identical freshly built CPU core plus the explicit plugin avoided
changing unrelated code. Loaded core/plugin paths and SHA-256 hashes were
checked through `/proc/self/maps`. The unchanged GenAI runtime was built
from `fe0badb408398df6b6a9bf0557a45912549434d0`.

### Single uploads and staging fallback

The native harness calls `DataTransfer::CopyTensors` with one tensor or
eight tensors. Sizes are per tensor. Every case verifies the uploaded
bytes, any preceding GPU-read snapshots, the selected path, and staging
and submission counts. Nine paired process runs alternate before/after
order, with eight warmup uploads per case. The initial sweep uses
128 iterations through 64 KiB, 32 through 1 MiB + 4 B, and eight above that.
Longer single-upload controls use 1,024 iterations at 64 KiB, 256 at 1 MiB,
and 32 at 16 MiB, again with nine paired runs.

The following values are **before / after medians, in microseconds per
call**. Host timing includes GPU completion via mapped timestamp readback.
GPU timestamps bracket the queue timeline, including CPU feeding and idle
gaps; they are not isolated DMA duration. The start marker is submitted
before the host timer starts, so the two intervals have different
boundaries and must not be equated. Timestamp quantization is disabled.

| Bytes | Host, standalone | GPU timeline, standalone | Host, preceding read | GPU timeline, preceding read |
|---|---:|---:|---:|---:|
| 64 | 1101.44 / 8.30 | 1101.15 / 7.40 | 1098.00 / 49.54 | 1097.99 / 48.78 |
| 256 | 1100.47 / 6.29 | 1100.31 / 9.22 | 1098.50 / 36.10 | 1098.37 / 35.19 |
| 1,024 | 1115.96 / 6.43 | 1115.82 / 9.28 | 1103.81 / 32.15 | 1103.59 / 31.24 |
| 4,096 | 1107.61 / 6.65 | 1107.49 / 9.50 | 1117.70 / 32.64 | 1117.65 / 31.56 |
| 16,384 | 1114.50 / 7.82 | 1114.37 / 10.59 | 1114.78 / 32.73 | 1114.70 / 31.61 |
| 65,536 | 1141.87 / 46.44 | 1141.84 / 46.32 | 1140.08 / 83.12 | 1140.07 / 101.51 |
| 262,144 | 1312.11 / 191.77 | 1311.58 / 200.99 | 1316.81 / 149.92 | 1315.90 / 142.43 |
| 1,048,576 | 1869.92 / 658.37 | 1869.73 / 658.82 | 1890.56 / 349.59 | 1890.43 / 717.51 |
| 1,048,580 | 1886.63 / 1837.84 | 1885.95 / 1837.73 | 1895.84 / 1852.41 | 1895.17 / 1852.22 |
| 4,194,308 | 4292.89 / 4250.22 | 4294.91 / 4246.40 | 4327.22 / 4234.65 | 4326.66 / 4235.90 |
| 16,777,216 | 15601.60 / 15508.90 | 15596.30 / 15509.60 | 15027.10 / 14787.80 | 15024.50 / 14784.80 |

The longer controls replace the corresponding initial-sweep rows above;
both sets of raw samples are retained. Most small-upload GPU timeline
IQR/median values are 0.3-3.3%. The 256 KiB standalone after case is 6.9%,
and 64 KiB with preceding reads remains variable at 19.2% even in the
longer control; do not interpret that case as a precise latency guarantee.
Its median paired completed-host reduction is still 92.7%. At 16 MiB,
median paired host changes are 1.20% and 0.29% improvements, within
roughly 2-4% per-variant variation; no meaningful large-upload regression
was established. Near-equal staging timings are not another queue-write
gain.

Path checks confirm that all after single uploads through 1 MiB use
`WriteBuffer`; larger cases use staging in both variants. The cap therefore
preserves the existing large-upload path rather than extending the small
upload result to unvalidated sizes.

Separate diagnostic builds count actual Vulkan `vkQueueSubmit` calls as
well as ORT submissions; instrumented latency is not used in these tables.
For 128 standalone 64 B uploads, before creates 128 staging buffers and
makes 128 ORT submissions; after makes 128 queue writes with no staging
buffers or ORT submissions. Actual Vulkan submissions are 129 / 1,
including the final timestamp marker. With preceding GPU reads, both make
128 ORT and 129 Vulkan submissions: ordering still requires flushing the
readers, while after removes staging allocation. For the original
32-iteration standalone 1 MiB case, after has no ORT submissions but Dawn
makes two internal Vulkan submissions plus the final marker. Submission
claims must distinguish these layers.

### Batched input control

Successful eight-input batches select `UploadBatch` in both builds and
make **zero queue writes**. An initial mixed single/batch sweep appeared
to improve some unchanged batch cases by 10-14%. Repeating batches alone
in fresh processes removed the different preceding single-upload histories;
only this nine-pair isolated control is used below.

| Bytes per input | Host, standalone, before / after (us) | Host, preceding read, before / after (us) |
|---|---:|---:|
| 64 | 1109.78 / 1110.91 | 1115.50 / 1111.83 |
| 4,096 | 1149.88 / 1142.66 | 1135.63 / 1123.37 |
| 65,536 | 1487.19 / 1491.37 | 1506.73 / 1509.87 |
| 1,048,576 | 7874.50 / 7979.99 | 8010.79 / 8055.50 |
| 4,194,308 | 31408.30 / 31241.50 | 31381.80 / 31133.40 |
| 16,777,216 | 124452.00 / 122427.00 | 122918.00 / 121992.00 |

Median paired host changes for these controls range from a 0.63% slowdown
to a 1.63% improvement; GPU timeline IQR/median ranges from 0.8% to 5.6%.
Staging allocations, tensor copies, batch calls, and submissions match.
Eight inputs of 4 MiB + 4 B require three staging chunks per call in both
builds. There is **no additional batched-path benefit attributable to
Queue.WriteBuffer**, and no duplicated upload. The initial apparent batch
gain must not be counted as an incremental optimization.

### Qwen 3.8 generation

The target is the same Qwen 3.8 27B INT4 paged-attention export, with the
genuine DFlash2 width-3 drafter (four query rows, three drafts). Short is
86 prompt tokens plus 128 generated tokens; long is 1,315 plus 512.
Graph capture is disabled, Vulkan/basic validation is identical, and
experimental upload/SIMT environment flags are removed. Each
variant/mode/workload receives one full-length warmup followed by five
paired runs, reversing execution order on alternate pairs. Separate
persistent workers load the models before timing; their requests execute
serially on the GPU. Timing includes an explicitly checked GPU completion
fence, but excludes model loading.

| Mode | Workload | Before median (s) | After median (s) | Latency reduction |
|---|---|---:|---:|---:|
| Baseline | Short | 5.174488 | 5.002032 | 3.33% |
| Baseline | Long | 22.820360 | 21.270567 | 6.79% |
| DFlash2 width 3 | Short | 11.750336 | 11.457950 | 2.49% |
| DFlash2 width 3 | Long | 54.657887 | 52.972959 | 3.08% |

Reductions above compare medians; medians of the five individual paired
reductions are 3.39%, 6.26%, 2.50%, and 3.08%, respectively. These are
observed incremental results **after #33491**, not the combined gain
against older main or a universal performance guarantee. Baseline trials
include outliers; all raw samples are retained.

Before/after generated token IDs and all speculative-counter deltas match
exactly for every pair and across repetitions. Width 3 accepts/evaluates
84/103 drafts in 42 completed verification rounds for short, and 323/443
in 188 rounds for long. Baseline has no speculation. Width-3 short also
matches baseline tokens, but width-3 long diverges from baseline at
generated token index 55 **in both builds**. This pre-existing
baseline/speculative difference is not fixed or introduced by this PR.
Width 3 is slower than baseline in this clean-main setup; older timings
from combined experimental branches are not a controlled comparison.

Separate instrumented generation runs confirm the remaining opportunity:

| Case | Single staging calls, before / after | Queue writes, after | Batch calls, both | ORT submissions, before / after | Vulkan submissions, before / after |
|---|---:|---:|---:|---:|---:|
| Baseline short | 256 / 0 | 256 | 129 | 33571 / 33443 | 33571 / 33443 |
| Baseline long | 1028 / 0 | 1028 | 515 | 134769 / 134255 | 134769 / 134255 |
| Width-3 short | 264 / 0 | 264 | 89 | 14822 / 14778 | 14822 / 14778 |
| Width-3 long | 1148 / 2 | 1146 | 383 | 64568 / 64377 | 64568 / 64377 |

These counts include completion-fence work but exclude model loading.
Queue-write counts cover this `Upload` path, not existing uniform-buffer
writes. Batched tensor counts and bytes are unchanged. Width-3 long
retains two larger single-upload staging calls because of the 1 MiB cap.

### Correctness and reproduction

| Suite | Before | After |
|---|---|---|
| Native `*WebGpu*:*WebGPU*` | 965 passed, 3 skipped | 980 passed, 3 skipped |
| Plugin modern `*WebGpu*:*WebGPU*` | 278 passed, 4 skipped | 278 passed, 4 skipped |
| Repository-selected plugin legacy recording/concurrency | 46 passed | 46 passed |

Each suite also has two disabled tests; none failed. Native skips cover
unsupported packed split-K selection and an adapter-index build
configuration. Plugin skips cover unsupported segmented-buffer/aliased
state cases. Upload, deferred-reader ordering, independent concurrency,
capture/replay fallback, and the new alternating batch/single regression
all ran successfully. Initial plugin failures were missing `mul_1.onnx`
fixtures in the targeted build directory, resolved by copying repository
testdata and rerunning both variants; they were not upload defects.
C++ lintrunner and `git diff --check` also passed.

The complete local experiment is retained at
`/workspace/onnxruntime/build/webgpu-queue-main-20261023/`.
`provenance.json` identifies source/model/workload revisions and hashes.
Build-command JSON and variant manifests retain exact compiler/linker
commands and artifact hashes. `summary.json`, raw microbenchmark JSONL,
generation token IDs/counters, diagnostic traces, test XML/logs, and
`test-commands.json` / `plugin-test-commands.json` retain the results.
The latter command file records the corrected, passing plugin runs.

Recorded benchmark entry points, after the paired builds and with the
repository virtual environment activated, are:

```bash
R=/workspace/onnxruntime/build/webgpu-queue-main-20261023
python "$R/reproduction/run_micro.py"
python "$R/reproduction/run_micro.py" --diagnostic --pairs 1
python "$R/reproduction/run_micro.py" --batch-only
python "$R/reproduction/relink_micro.py"
python "$R/reproduction/run_micro.py" --single-stability
python "$R/reproduction/generation.py" --trace
python "$R/reproduction/generation.py"
python "$R/reproduction/plugin_tests.py"
python "$R/reproduction/summarize.py"
```

All commands completed successfully. These archived scripts use fixed
experiment/model paths and refuse to overwrite trial directories; a
rerun needs fresh output paths. GPU timing runs must execute serially,
without concurrent builds, correctness suites, or diagnostic generation.

### Recommendation and remaining validation

**Merge the bounded optimization with the updated main integration and
mixed-upload regression test, after required CI and review.** Keep the
1 MiB threshold and make no device-specific changes. #33491 already
optimizes successful multi-input batches, but single/non-batchable
transfers remain common and the controlled generation comparison still
improves. No redundant upload or new correctness failure was found;
batched-path timing noise must not be claimed as another gain.

Local performance evidence is A100/Vulkan only. New-head Windows/D3D12
and macOS/Metal CI, including default routing, ordering, and concurrency,
remain required. PR approval is also required. The old head's Android
failure occurred while downloading/unpacking NDK 28.0.13004108
(`Error on ZipFile unknown archive`), before compilation; its WebGPU
arm64 Debug job was cancelled. Those stale checks are not evidence
against the new implementation, nor a substitute for new-head CI.
