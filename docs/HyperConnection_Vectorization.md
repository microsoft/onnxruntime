# Experimental FP16 HyperConnection vectorization

`ORT_HYPER_CONNECTION_VECTORIZE=1` opts into eight-feature CUDA PreMix and
PostMix consumer kernels ported from `a98d870911`. It defaults off. Set it
before process startup; the flag is cached on first launcher use.

This preserves the Engine API, model graphs, projection/sigmoid boundaries,
branch accumulation order and final FP16 rounding. Only FP16 streams with
FP16 gates are eligible, with hidden width divisible by eight and 16-byte
aligned vector buffers. PreMix requires at least 256 rows; PostMix requires
at least 64 rows and at most four branches. Other types, FP32 gates, small
decode/MTP rows and unsupported shapes use the unchanged scalar kernels.

The broader upstream-style dispatch was rejected: vectorized FP32 gates
changed rounding in a PreMix test, and small PreMix vector launches were
slower. These eligibility restrictions must not be broadened without new
exact-bit and performance qualification.

GPU4 qualification used the real FP16 branches=4, hidden=2560 geometry,
including rows 1/2/3/8192, gate layouts and FP16/FP32 gate types, optional
PostMix stream matrices, cancellation/extreme-finite inputs, unsupported
shape/type fallbacks and the 64/256-row dispatch boundaries. The isolated
test tools and raw reports are in the sibling worktree:

* `asonawane-opt-projection/tools/experimental_hyperconnection/`
* `asonawane-opt-projection/hyperconnection-artifacts/`

The isolated build compiles only these two CUDA sources and relinks a
separate provider using retained Release inputs, with at most four nvcc
threads. No original build or external weight copy is required.

Engine A/B/A/B used the selected-logits long-context model, an 8192-token
target prompt (8190 actual), two draft tokens, exactly 512 output tokens,
one excluded warmup and three measured requests per fresh process.
Prefix caching and GEMM autotuning were disabled; other optimization flags
were identical. All 12 measured requests and four warmups matched their
complete token hashes.

Mean TTFT was 1200.315 → 1159.453 ms (3.40% lower); mean E2E throughput was
95.909 → 97.064 TPS (1.21% higher). This is a bounded uncached 8K result, not
a universal or cached-prefix claim. Decode kernels are intentionally
unchanged and individual decode timing varied; no decode speedup is
attributed to this flag.

The final bounded provider passed 142 bitwise eager/captured operator cases.
Its default-unset flag also matched A on the eight actual-graph geometry
cases. A final fresh-process Engine check added one warmup and three
measured requests; all four again matched the same 512-token stream and
measured TTFT was 1166.912 ms. In total, 15 measured requests and five
excluded warmups matched. The original CUDA provider was not rebuilt;
the tested isolated provider and full provenance are retained in
`hyperconnection-artifacts/qualification.json` and the associated raw
reports. Enabling the flag in the original runtime requires rebuilding
that provider from the integrated sources.
