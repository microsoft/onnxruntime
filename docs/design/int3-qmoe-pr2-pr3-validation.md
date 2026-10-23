# INT3 QMoE PR2/PR3 Validation

Validated on 2026-10-23, NVIDIA A100, CUDA 13, branch
`jiafa/int3-qmoe-pr2-pr3`, based on main `96dd9658a6`.

## Implementation

This combines the CUDA correctness baseline and packed-decode/private-prepack
work. The canonical QMoE v1 model format and ONNX dependency are unchanged.
The three FC1/FC2 pairs are 3/3, 3/4 and 4/3, with FP16/BF16 and blocks
32/64/128. Prefill uses bounded direct-packed execution, not optimized GEMM.

Initializer weights use provider-private version-1 K32 bit planes, retaining
12 bytes per 32 INT3 values. Kernel-owned cache entries record shape, type,
width, block size, layout version and device identity. Initializer scales and
INT3 zero points are validated during prepack. Consumed initializers are
released; dynamic inputs retain canonical layout and per-run validation.
No private bytes are exported, and no global cross-session cache is introduced.

## Correctness

- A fresh provider/test build in `jiafa-dev` at
  `/tmp/int3-pr2-pr3-clean/Release` compiled ORT and dependencies from source
  with consistent CUDA 13 headers/runtime. Existing dependency source trees
  were reused, not cached ORT objects.
- Formal `QMoEInt3CudaTest.*`: **9 passed, 0 failed, 0 skipped** after final
  runtime formatting. Tests cover all-constant/all-dynamic matrices, partial
  prepack, malformed content, same-session valid/invalid/valid mutations,
  initializer release, session recreation, empty rows, malformed scale shapes
  and exact scratch boundaries.
- Standalone row-local loader: **80 cases passed**. Dual projection:
  **216 cases passed**, repeated three times. Both standalone suites passed
  Compute Sanitizer with zero reported errors.
- The build used local ONNX source `ee3ccbd2b2344299d3a4506c2954a47b2181a485`
  rather than the repository-pinned `e1ac8e74b620bbdc7ba07f0440806bd93f651a28`.
  This is a build-provenance limitation, not a new INT3 dependency. Default-pin
  CI qualification remains required; no ONNX gitlink is changed.
- Concurrency, CUDA graph capture and broad deployment-GPU qualification were
  not established by these tests.

Reproduction in the configured container:

```bash
ninja -C /tmp/int3-pr2-pr3-clean/Release -j8 \
  onnxruntime_provider_test onnxruntime_providers_cuda
cd /tmp/int3-pr2-pr3-clean/Release
./onnxruntime_provider_test --gtest_filter='QMoEInt3CudaTest.*'
```

## Performance

See [Session analysis](int3-qmoe-pr3-session-analysis.md), its raw JSON and
the committed reproduction harness. At H2048/I768/E128/k8, rows1, block64,
six width/type cases completed 72 fresh Sessions and 2,160 measured calls with
exact finite nonzero output agreement.

Constant/private-cache steady median latency was 0.469-0.486 ms versus
0.660-0.736 ms for all-dynamic canonical inputs (1.37-1.54x ratio). These
different validation/cache paths do not isolate a universal layout speedup.
Prepack-inclusive initialization cost was 649-892 ms. Sampled resident device
increments were 774-834 MiB for constant mode versus 264-312 MiB for dynamic
mode, including feeds and allocator retention, not peak memory. Initialization
cost and memory overhead are material trade-offs, not hidden by steady timings.

[Kernel-only raw results](int3-qmoe-pr3-kernel-results.jsonl) separately compare
canonical and private layout and record conversion cost and coexisting bytes.
These results do not establish whole-model acceleration or PR4 acceptance.
The `int3-qmoe-cuda-baseline-results.*` files preserve historical #33519
full-model evidence; they are not new full-model measurements of this revision.