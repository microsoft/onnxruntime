# Opt-in TLWU-compatible MoE decode subset

Adapted from TLWU routing `3e09dced1b`/guards `1f9387d872` and finalize
`e011e19a66`, against baseline `bfd45538f44fe0e72dbb08350ae98d84bed67f53`.
No quantization, weights, GEMV reduction order, Generator runtime, or new PDL
changes are included. The original raw packed NVFP4 GEMV object is unchanged.

## Flags and fallbacks

Both flags default to **0**, enable only for value **1**, and are read once per
process/template instantiation. Set them before loading the CUDA runtime.

* `ORT_NVFP4_DECODE_FUSED_ROUTING`: fuse softmax/top-k and expert maps only for
  raw row-major NVFP4 GEMV on SM90+, 512 experts, 1–4 rows, aligned FP16/BF16
  logits, and top-k 1/2/4/6/8/10. Routing snapshots and unsupported shapes use
  the original route/map path. Lower-index ties, the original exp-sum tree,
  sequential normalization, and expert/token map ordering are preserved.
* `ORT_NVFP4_DECODE_SINGLE_PASS_FINALIZE`: use 320 threads and preload expert
  vectors for 1–4 rows, hidden dimension 2560, top-k 10, and 16-bit input/output.
  Accumulation remains in the original top-k order. This flag affects the
  **shared MoE finalizer**, including other MoE formats matching these guards;
  it is not restricted to NVFP4. Unsupported shapes retain their old launch.

These are small-row/decode specializations, not grouped-prefill pipelining.
Normal 8190-token prefill does not satisfy their row guards. Do not infer a
prefill gain from the small, noisy TTFT difference below.

## Engine qualification

H200 GPU5 only; GenAI **Engine**, batch 1, greedy, requested prompt 8192
(actual 8190), exactly 512 output tokens, MTP width 1, scheduler 8192, CUDA
graphs enabled, decoder and MTP GEMM autotuning **off**, no prefix cache.
Sparse grouped-head/async-KV and existing packed-prefill/SwiGLU/routing-scratch/
tiled-down flags were enabled identically on both sides. Rejected scale
prefetch and experimental register prefetch were off/absent.

Final isolated provider SHA256:
`cb34bf8a67e15c494060d07bfd578d0ff54233e9b041159b026c36105c65bd41`.
Model: `/var/tmp/asonawane-qwen38-long-context-selected-logits-engine`.
Runtime: `build/tlwu-decode/runtime-safe` in the isolated MoE worktree.
Python: `/home/asonawane/.conda/envs/flash/bin/python`.

Each invocation had one warmup plus three measured requests:

| Invocation, in order | TTFT ms | Decode TPS | E2E TPS |
| --- | ---: | ---: | ---: |
| `actual_safe_control_reverse_clean` | 1148.305 | 117.503 | 93.138 |
| `actual_safe_repeat_candidate` | 1145.007 | 120.856 | 95.287 |
| `actual_safe_repeat_control` | 1145.716 | 117.731 | 93.326 |

Pooled per-request means: decode **117.617 → 120.856 TPS (+2.75%)**,
E2E **93.232 → 95.287 TPS (+2.20%)**. Both bracketed controls agree within
0.20%; each candidate measured sample exceeds every control sample.
All 12 complete streams are identical, output SHA256
`d6a43167140ec287056c54e331c66cd20b3949b90f44ea0c0a15c1b9ff4ccad6`.
Each request proposed 281 drafts, accepted 229, and had zero MTP failures.

456 standalone FP16/BF16 bitwise parity cases cover ties, normalization,
maps, bias/nonlocal experts, and eager plus CUDA graph replay; six dispatch
fallback guards pass. Selected existing integration tests: 94 passed,
4 skipped. Safe standalone memcheck: zero errors; racecheck: zero hazards.

## Rejected experiments and scope of evidence

The all-port register-prefetch repeat produced a different complete sequence;
that experimental GEMV source is excluded from this integration. Its initial
larger speedup is not an output-equivalent claim. Autotune-on results are not
the final performance claim. Earlier autotune-off controls varied, so they
are retained but not pooled into the final clean bracket. One earlier reverse
control overlapped standalone GPU tests; its timing is explicitly excluded.

The previous prefill scale pipeline was a production E2E nonwin and is not
integrated. A historical ~71 E2E TPS run used scheduler 1024, graphs off, and
different tuning settings; it is not comparable to this Engine configuration
or true cached-8K runs.

All raw JSON/logs, per-request CSV, settings/runtime/source hashes, sanitizer
logs, and the safe patch are retained under the isolated worktree's
`build/tlwu-decode/`; see `safe-final-manifest.json` and
`safe-engine-all-requests.csv`. The private build helper uses read-only baseline
objects; `build_tlwu_decode.py --safe --reference-build <baseline-build>`
excludes GEMV entirely. `qualify_tlwu_decode.py --reference-build
<baseline-build>` runs the safe standalone suite by default.

Recommendation: integrate this narrow **default-off** subset for further
Engine workload qualification; do not enable register prefetch or the rejected
prefill pipeline. Original source integration does not replace any original
provider binary and requires a coordinated rebuild.
