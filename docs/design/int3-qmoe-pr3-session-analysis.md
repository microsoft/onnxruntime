# PR2/PR3 QMoE Session Performance

Measured 2026-10-23T04:00:53Z; performance analysis only.

## Result

All six requested cases completed with exact finite nonzero output agreement across three modes and four fresh processes/mode: 72 Sessions and 2,160 measured decode calls. No speed-goal or full-model claim is made.

H=2048, I=768, E=128, k=8, rows=1, block=64. FP16 and BF16; bit pairs 3/3, 3/4, 4/3. Input [1,1,H], router [1,E], interleaved SwiGLU, normalized routing, explicit packed zero points, no biases; deterministic codes cover the full bit range, scale=1/64. Graphs are matched by raw tensor hashes in the measurement JSON.

Modes: `constant` caches all six weight/metadata initializers; `dynamic_metadata` keeps weights constant but feeds scales/zero points; `dynamic_all` feeds all six. Input/router are GPU feeds in every mode. No per-run weight transfers. Output is copied to CPU in every mode.

## Session Timings

Milliseconds; medians. Steady median pools 120 samples/mode/case. Raw per-process samples and launch positions are retained.

| Bits / Dtype | Mode | Feed Setup | Init (Prepack-Inclusive) | Feed + Init | First Run | Steady |
|---|---|---:|---:|---:|---:|---:|
| 3/3 fp16 | constant | 0.505 | 654.340 | 654.856 | 3.994 | 0.477865 |
| 3/3 fp16 | dynamic_metadata | 19.160 | 530.511 | 549.671 | 7.327 | 0.727500 |
| 3/3 fp16 | dynamic_all | 223.339 | 11.680 | 235.231 | 4.661 | 0.735065 |
| 3/3 bf16 | constant | 0.496 | 649.411 | 649.906 | 3.967 | 0.478972 |
| 3/3 bf16 | dynamic_metadata | 20.479 | 588.362 | 608.314 | 7.361 | 0.732334 |
| 3/3 bf16 | dynamic_all | 248.904 | 11.557 | 260.349 | 4.671 | 0.736311 |
| 3/4 fp16 | constant | 0.501 | 781.645 | 782.160 | 5.617 | 0.468943 |
| 3/4 fp16 | dynamic_metadata | 20.844 | 661.362 | 682.206 | 7.955 | 0.689370 |
| 3/4 fp16 | dynamic_all | 281.849 | 11.416 | 293.265 | 4.733 | 0.695225 |
| 3/4 bf16 | constant | 0.511 | 778.098 | 778.612 | 5.259 | 0.476989 |
| 3/4 bf16 | dynamic_metadata | 21.150 | 686.191 | 707.342 | 8.042 | 0.693036 |
| 3/4 bf16 | dynamic_all | 287.506 | 11.317 | 298.918 | 4.705 | 0.690511 |
| 4/3 fp16 | constant | 0.507 | 883.305 | 883.800 | 3.985 | 0.478046 |
| 4/3 fp16 | dynamic_metadata | 22.225 | 923.958 | 946.243 | 7.530 | 0.662764 |
| 4/3 fp16 | dynamic_all | 308.545 | 11.287 | 319.817 | 4.575 | 0.660145 |
| 4/3 bf16 | constant | 0.508 | 892.141 | 892.650 | 4.036 | 0.485890 |
| 4/3 bf16 | dynamic_metadata | 21.738 | 885.721 | 907.529 | 7.353 | 0.667799 |
| 4/3 bf16 | dynamic_all | 273.847 | 11.514 | 285.495 | 4.593 | 0.664067 |

## Validation And Layout

Positive validation-path deltas are `dynamic_metadata - constant` and include feed bookkeeping and synchronization. Both retain private weights. Routing is unchanged; no routing savings are claimed.

| Bits / Dtype | Validation-Path Wall Delta (ms) | Dynamic-All / Constant Steady Ratio |
|---|---:|---:|
| 3/3 fp16 | 0.249635 | 1.538x |
| 3/3 bf16 | 0.253362 | 1.537x |
| 3/4 fp16 | 0.220427 | 1.483x |
| 3/4 bf16 | 0.216047 | 1.448x |
| 4/3 fp16 | 0.184719 | 1.381x |
| 4/3 bf16 | 0.181909 | 1.367x |

Read-only existing kernel benchmark, same nominal decode shape; not rerun here, CUDA-event timings only:

| Bits / Dtype | Canonical (ms) | Private (ms) | GPU Transform (ms) |
|---|---:|---:|---:|
| 3/3 fp16 | 0.291430 | 0.281958 | 16.426 |
| 3/3 bf16 | 0.294195 | 0.282778 | 16.101 |
| 3/4 fp16 | 0.234957 | 0.226048 | 14.657 |
| 3/4 bf16 | 0.236288 | 0.228045 | 13.694 |
| 4/3 fp16 | 0.233933 | 0.235008 | 13.693 |
| 4/3 bf16 | 0.237568 | 0.238131 | 13.719 |
| 3/3 fp16 | 0.025446 | 0.025139 | 0.031 |
| 3/3 bf16 | 0.025446 | 0.025037 | 0.025 |
| 3/3 fp16 | 0.026266 | 0.026214 | 0.025 |
| 3/3 bf16 | 0.026522 | 0.026266 | 0.025 |

Private layout is not uniformly faster in that isolated benchmark (4/3 is slightly slower). New Session results show a benefit from the combined cached control path, not proof of universal layout acceleration. Init overhead is substantial and must be amortized; these runs do not establish a full-model break-even point.

## Memory

MiB; medians of per-process measurements. Init delta is after-init minus before-init. Resident delta is steady minus before-feeds. Device figures are sampled device-wide residency, not peaks. Host RSS is whole-process high-water RSS.

| Bits / Dtype | Mode | Init GPU Delta | Steady GPU Delta Including Feeds | Peak Host RSS |
|---|---|---:|---:|---:|
| 3/3 fp16 | constant | 762.0 | 774.0 | 680.6 |
| 3/3 fp16 | dynamic_metadata | 740.0 | 774.0 | 666.4 |
| 3/3 fp16 | dynamic_all | 12.0 | 264.0 | 433.3 |
| 3/3 bf16 | constant | 770.0 | 782.0 | 680.9 |
| 3/3 bf16 | dynamic_metadata | 740.0 | 774.0 | 666.4 |
| 3/3 bf16 | dynamic_all | 12.0 | 264.0 | 433.9 |
| 3/4 fp16 | constant | 786.0 | 798.0 | 704.4 |
| 3/4 fp16 | dynamic_metadata | 636.0 | 670.0 | 690.9 |
| 3/4 fp16 | dynamic_all | 12.0 | 288.0 | 433.9 |
| 3/4 bf16 | constant | 786.0 | 798.0 | 705.1 |
| 3/4 bf16 | dynamic_metadata | 764.0 | 798.0 | 690.9 |
| 3/4 bf16 | dynamic_all | 12.0 | 288.0 | 433.9 |
| 4/3 fp16 | constant | 822.0 | 834.0 | 776.8 |
| 4/3 fp16 | dynamic_metadata | 788.0 | 822.0 | 763.2 |
| 4/3 fp16 | dynamic_all | 12.0 | 312.0 | 481.9 |
| 4/3 bf16 | constant | 810.0 | 822.0 | 777.6 |
| 4/3 bf16 | dynamic_metadata | 724.0 | 758.0 | 763.2 |
| 4/3 bf16 | dynamic_all | 12.0 | 312.0 | 481.7 |

Constant-session residency includes ORT arena retention and temporary-allocation reuse; it must not be equated to private packed payload size. Raw JSON includes before-feeds, before-init, after-init, after-first, after-steady and after-session-destroy snapshots.

## Reproduce

From the worktree root on the host, with `jiafa-dev` running and the accuracy mount visible at the same path:

```bash
python3 onnxruntime/test/python/quantization/qmoe_packed_reference/session_perf.py \
  --work /datadisks/disk1/jiafa/accuracy/int3-session-perf-full-scratch \
  --output docs/design/int3-qmoe-pr3-session-results.json
python3 onnxruntime/test/python/quantization/qmoe_packed_reference/session_perf_report.py \
  docs/design/int3-qmoe-pr3-session-results.json \
  --prefix docs/design/int3-qmoe-pr3-session-analysis
```

For a quick probe append `--quick --hidden 128 --intermediate 64 --experts 8 --top-k 2 --repeats 2 --warmup 2 --samples 3` and use a different output path. The runner links a standalone C++17 client using the clean provider-test link dependencies; provider library symlinks are created only in scratch. It does not build or modify runtime. Generated fixtures/raw worker files remain in scratch; final JSON is self-contained.

Dependencies: host Python 3.12, NumPy/ONNX/ml_dtypes (versions below), Docker CLI, git, nvidia-smi; container g++, Ninja, CUDA 13 headers/runtime, cuDNN 9, clean ORT static archives/shared CUDA provider. Existing MLAS assembly emits a GNU-stack linker warning, but linking and execution succeed. Building target `onnxruntime` was not possible in this static configuration; pybind was neither required nor built.

## Provenance

- Worktree HEAD: `96dd9658a6c6d3ac60c0c03b1ead1cea1ba9f1f1`; dirty source identities are SHA256-recorded, not inferred from HEAD.
- Build: `/tmp/int3-pr2-pr3-clean/Release`, Release, shared core OFF, CUDA provider ON; clean core/provider build only.
- ORT 1.32.0; NVIDIA A100-SXM4-80GB; UUID hex `739515b262b4f90086a250f55272a810`; runtime/driver API versions 13000/13000.
- Local ONNX source: `ee3ccbd2b2344299d3a4506c2954a47b2181a485`; differs from pinned `160000 commit e1ac8e74b620bbdc7ba07f0440806bd93f651a28	cmake/external/onnx`. CMake uses `FETCHCONTENT_SOURCE_DIR_ONNX=/datadisks/disk1/jiafa/accuracy/onnxruntime/cmake/external/onnx`; declared deps archive is ONNX v1.23.2.
- Host dependency versions: `{"ml_dtypes": "0.5.3", "numpy": "2.4.2", "onnx": "1.19.1"}`.
- Measurement SHA256: `e615e55a88d771f75dfca99aae9f9e4bf6a6863c7cb2e72df4b53f831ca735df`.

Clean artifact SHA256 identities:

```text
03349564a3c516767df8b847813b3e451c4d1ded502c62b3ebc78106d166973f  /datadisks/disk1/jiafa/accuracy/int3-session-perf-full-scratch/session_perf
78e89885d9f3aca8b0e51a5f4c2a75167ed8c77e67f05d79073dbb707bc99709  /tmp/int3-pr2-pr3-clean/Release/libonnxruntime_session.a
4b430b18ab02b31e0f299858999c25026178528740ac592ca95905f5cf9bd9cc  /tmp/int3-pr2-pr3-clean/Release/libonnxruntime_providers_cuda.so
9836a8dcc066e6e3ca2c02cd150052134aa02484e90e3f003c84a6a4f713c3d3  /tmp/int3-pr2-pr3-clean/Release/CMakeCache.txt
```

Source dependencies: `qmoe_int3.cc` (`PrePackInt3` and `ComputeInt3`) controls private-layout caching, metadata validation and routing; `qmoe_int3.cu` implements kernels; `moe_quantization.cc` dispatches QMoE; `make_cuda_session_smoke.py` supplies the canonical row packer; `session_perf.cc` and `session_perf.py` implement this experiment. Existing `qmoe_int3_benchmark.cu` and its JSONL are read-only context. Full paths/hashes, tensor hashes, launch order and raw timing samples are retained in the results JSON; generated analysis JSON includes definitions/limitations and contextual kernel data.

## Limits

- One QMoE node, synthetic deterministic repeated-pattern nonzero weights, fixed input/router, rows=1; not a full model, prefill, MMLU, or quality evaluation.
- Constant and dynamic graphs use identical numerical tensors but different control paths. No historical runtime or old full-model baseline is used.
- Init is synchronized Session construction: model file read/parse, CPU initialization, allocation, host-to-device initializer copies, private transforms, validation and synchronization. It is prepack-inclusive, not an isolated prepack timer.
- GPU feed setup (disk read, host allocation and H2D copies) is outside init and reported separately; graph generation, CUDA context/provider setup and process startup are outside both. Total setup is feed setup plus init, not process startup latency.
- First run includes lazy loading, allocations, validation if dynamic, routing, compute and small CPU output copy. Steady is synchronized CPU wall time per Run, not CUDA event kernel time. Session optimization is disabled and CPU EP fallback forbidden.
- Dynamic metadata retains private weight layouts but bypasses cached metadata validation. Its delta includes validation kernels, host synchronization and dynamic-feed bookkeeping; it is not a pure validation-kernel timer. Layout effects are only isolated by the separate existing kernel benchmark.
- All three modes execute the same SoftmaxTopK routing. No routing speedup or launch-count reduction was independently measured; no routing savings are claimed.
- cudaMemGetInfo snapshots are device-wide, synchronized sampled residency, not allocator counters or allocation peaks. Baseline is after CUDA/provider setup. Before-feed to steady includes device feeds and ORT arenas; after-destroy retains device feeds. Other GPU processes can perturb deltas. No NVML per-process or transient GPU-peak sampling was performed.
- Peak host RSS uses Linux getrusage and covers the process, including model loading and feed staging; it is not incremental host prepack memory.
- Four fresh processes per mode, alternating forward/reverse order; 10 warmups and 30 samples/process. Pooled medians and per-process medians are descriptive, not confidence intervals. Clocks are not locked, file cache is not flushed, and init/first-run variability is retained in raw data.
- Correctness means exact finite nonzero output agreement across modes/repetitions, not an independent mathematical oracle or new runtime correctness test.
- Clean core/provider build only, not whole CI. Static Session archives and the clean shared CUDA provider are linked without pybind. ONNX source override differs from the pinned gitlink; no global headers, runtime sources or dependencies were edited by this experiment.
- Existing kernel results are read-only contextual data supplied before this run, not rerun here and not proven to share the exact measured binary. Their GPU-event timings cannot be subtracted from new Session wall timings.
- No full-model throughput, memory or target-performance completion claim follows from these results.
