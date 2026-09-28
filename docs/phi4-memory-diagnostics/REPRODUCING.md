# Reviewing and reproducing the historical evidence

## What can be checked from this contribution

This directory contains compact evidence, not model weights, captured
vectors or a turnkey inference application. It does not download anything
or automatically run GPU work.

From this directory:

```sh
sha256sum --check SHA256SUMS
```

This checks package integrity, not whether a model run is correct.
For an offline result review:

1. Group `evidence/memory-before-after.csv` by `context` and `variant`.
   Each group has repetitions 1, 2 and 3. Compute column medians, retaining
   the distinction between process NVML, whole-device NVML and BFC capacity.
2. Confirm that both context pairs in `evidence/parity-checks.json` have
   matching logits hashes, 800,256-byte FP32 vectors and 64 matching output
   IDs. Check that all 12 recorded Phase B token hashes match the appropriate
   Phase A reference. This checks report consistency, not the omitted raw
   comparisons.
3. Group the 45 earlier-runtime rows by context/runtime/allocator. They have
   three repetitions each. Keep actual input/output counts visible when
   computing medians; most llama.cpp conditions stopped before 64 tokens.
4. Use `evidence/model-manifest.json` for saved model/runtime hashes,
   input hashes and effective configurations. The older exploratory matrix
   is not assigned the later controlled experiment's configuration.

The original Phase A binary pairs were compared in full, without a
tolerance. During this preparation both complete logits pairs were again
compared byte-for-byte and against their recorded hashes. Raw generated
token files were not rechecked in this preparation; token claims retain
the recorded original comparison and digest/count consistency checks.

## Requirements for an exact historical GPU replay

**This compact bundle alone is insufficient for exact replay.** It omits
the original native capture/measurement application, saved input-token
vectors, baseline/patched runtime builds and raw reference outputs.
Access and sharing approval for those external artifacts must be arranged
separately; no public archive link is implied.

Use the following as the protocol, not as a claim that replay has been
performed from this documentation:

1. Pin ORT to benchmark commit
   `94e76459dc9a759888707aa430d51a95d878da8f`, GenAI to
   `ed5f4e87147731e5b07810f9f5c90103b3603cdf`, and the model to
   `fc04c8f93df696602fd9f300a30d1bf2e3081347`.
   Reproduce the historical baseline and allocation-only GenAI variant
   separately. Using GenAI main after #2607 is not the historical baseline.
2. Obtain the original saved 1,801/28,747 input-token vectors and verify
   their hashes using the original serialization. A prompt of similar
   length is not an equivalent input.
3. Match the recorded effective model/search settings. The manifest contains
   the configurations and hashes of their original serialized files.
   Reformatting JSON changes file hashes; distinguish byte identity from
   semantic configuration equality.
4. Build GenAI against the matching custom ORT headers/libraries. For the
   native application, select the intended library using `ORT_LIB_PATH`
   and a matching `LD_LIBRARY_PATH`. Verify the actual loaded GenAI,
   CUDA companion, ORT and provider library paths/hashes, rather than
   assuming the environment variable selected them.
5. Set `ORT_ARENA_DIAGNOSTICS=1` before model creation. The branch's
   `OrtApi_DebugLogAndShrinkGpuArenas` experimental API is explicitly
   called by a diagnostic native application, not automatically by every
   GenAI application. Capture after initialization, after generation and
   after per-request generator cleanup while the model is alive.
   **Use `shrink=false` for this measurement protocol.**
6. Separate correctness captures from measurements. In Phase A, capture
   the full first-token logits after AppendTokens and before generation,
   and all 64 output IDs. In Phase B, omit diagnostic logits copying and
   verbose attribution/profiler instrumentation. Take allocator snapshots
   outside timing and apply identical CUDA completion barriers to both
   variants.
7. Run fresh processes serially, with three repetitions per condition,
   baseline then patched within context/repetition, short context before
   long context. The original controlled protocol had no inference warmup.
   A different ordering or warmup policy is a new experiment and must be
   labeled as such.
8. Use an approved idle GPU and resource budget. The historical controlled
   runs used a 300-second watchdog, pre-run interference checks, process
   and whole-device NVML sampling requested at 10 ms, and recorded actual
   sampling windows. Reserve sufficient memory for the baseline, which
   peaked near 60,624 MiB whole-device on an 80 GB A100.
   Do not introduce automatic retries, shrinking or concurrent workloads
   into the historical comparison.
9. Compare complete logits and token vectors, record every failure/skip,
   and retain loaded identities plus raw sampling/allocator evidence.
   New results must be labeled with their own exact build and artifact
   identities, not appended to the historical CSV as original measurements.

The observed historical request and its small replacement should be
correlated with allocation events in an attribution run, separate from
performance measurements. Arena capacity alone does not identify an
allocation owner, exact CUDA free or kernel workspace.

## Scope of reproducibility

Model graph, external weights and tokenizer/config hashes were recomputed
during the original evidence packaging. Model revisions were recovered from
saved download metadata, not newly resolved for these measurements.
The earlier GGUF revision/hash in the manifest belongs only to the initial
ORT/llama.cpp comparison; that GGUF was not rehashed during packaging and
was not used for the GenAI allocation-fix experiment.

Do not claim a new current-main benchmark, current-main memory saving,
full chunking correctness, causal TTFT improvement, shrink safety or
concurrency validation from this historical protocol.
