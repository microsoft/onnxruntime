# WebGPU MatMulNBits small-M selector experiment

## Recommendation and status

**Experimental candidate, measured on A100/Vulkan only; not a production-readiness claim.**
The [correctness follow-up](matmul_nbits_small_m_correctness.md) reproduces the width-3 split,
isolates its numerical cause, and shows that both old and candidate width-7 sequences also
differ from baseline greedy. No speculative state or verification-row bug was found in that
reproduction, but exact greedy-output equivalence is not established.

Keep `kMinMForTileOptimization = 4`. After the existing subgroup eligibility check succeeds,
prefer regular MatMulNBits for the FP16-result, FP16-input, wave32 `16x16x16` configuration when:

```text
4 <= M <= 8 && uint64_t(M) * N <= 128 * 1024
```

This configuration otherwise computes a 128-row tile. The output-element budget includes
the hidden projections and fused attention QKV, but excludes the very wide output head.
At M=8 it also excludes the MLP gate/up projection, where subgroup was slightly faster.
No K condition was needed for the measured cases.

The preference is not an eligibility override. It can replace only an already-eligible subgroup
dispatch, skips the intervening DP4A and wide-tile choices to reach regular MatMulNBits, and leaves
all other fallback decisions unchanged. Other subgroup configurations, FP32, INT2/INT8, FP32
accumulation requests, M=1 decode, and M>=9 retain their existing selection. No shader implementation,
FP16 GEMV, GatedDeltaNet, or convolution replay code was changed.

## Existing selection and candidate rules

The existing priority in `ApplyMatMulNBits` is subgroup matrix, DP4A, wide tile, then regular.
The shared minimum-M constant is used by all three optimized choices; declining only subgroup
does not necessarily select regular.

Subgroup eligibility requires INT4/INT8, block32, M>=4, K divisible by 32, N divisible by 64,
one effective batch, and no indirect weight index. FP16 subgroup accumulation is declined when
FP32 accumulation is requested. Apple additionally requires accuracy level 4. Feature discovery
must find a matching subgroup-matrix configuration and a supported subgroup size; variable-size
adapters require subgroup-size control. These checks and the configuration preference order are
unchanged.

DP4A requires accuracy level 4, block size divisible by 32, four activation components, K divisible
by 128, N divisible by 16, subgroups, and a non-Apple adapter. Its M>=4/contiguous-weight condition
has existing FP32-output and Qualcomm exceptions. Wide tile requires block32, INT4/INT8, M>=4,
no indirect weight index, and four activation/weight components. These guards are unchanged too.

| Candidate | Definition on the eligible FP16/INT4 cases |
| --- | --- |
| A | Existing selector; subgroup starts at M=4. |
| B | Global minimum-M cutoff 9 reference: regular for M=4-8. |
| C | Prefer regular for M=4-8 and N<=12288; otherwise existing selection. |
| D, implemented | Prefer regular for M=4-8 and M*N<=131072; otherwise existing selection. Apply only to the eligible FP16 wave32 `16x16x16` configuration. |

An initial D prototype used M*N<=98304. Verification traces exposed the current export's fused
QKV with N=14336, whose M=8 regular path was also faster. The final, round 128K budget includes
it and the independently measured N=16384 boundary without including N=17408 at M=8.
B/C and the initial prototype were isolated benchmark selectors, not changes to the shared cutoff.

## Measurement setup

Measurements on 2026-10-20 used NVIDIA A100-SXM4-80GB, driver 580.159.04, WebGPU/Vulkan,
FP16 activations/scales, INT4/block32 real Qwen 3.8 weights, and accuracy level 4.
Eight representative weight matrices were extracted/reused from the existing exports, including
the fused QKV in the current DFlash2 export. Activations were deterministic synthetic FP16 values.

Each kernel case had at least three seconds of excluded conditioning, at least ten warmups,
and 80 GPU-timestamp samples. Tables report medians of the summed dispatch durations, including
subgroup activation-layout/prepack time. Inputs/outputs were GPU-resident and CPU fallback was
disabled. Host submission and readback time are not included. Repeated weights are warm-cache
microbenchmarks, not a simulation of all layer weights streaming through cache.

Clock samples were recorded without changing shared GPU clock settings. Most kernel cases ran
at 1410 MHz SM / 1593 MHz memory. Auto boost used lower SM clocks for some short regular cases
and the wide-head regular path (approximately 1275-1410 MHz); these cases were not normalized
to a fictitious fixed clock. All sampled measured end-to-end intervals were 1410/1593 MHz,
with GPU temperatures between 31 and 42 C.

The current WebGPU plugin and C++ tests were rebuilt. Isolated variants reused identical kernel
objects and changed only the selector object, plus the same existing diagnostic timestamp bridge.
The end-to-end runs reused the existing compatible ORT core/GenAI runtime and Engine benchmark
harness, with freshly linked WebGPU plugins. Loaded plugin paths and SHA-256 hashes were checked
through `/proc/self/maps`. Diagnostic profiling was disabled during wall-clock trials.

## Kernel timings and selected paths

All values are **microseconds, M=4 / M=8**. `R` is `MatMulNBits`; `S` includes
`SubgroupMatrixMatMulLayout` plus `SubgroupMatrixMatMulNBits`. These program names were verified
in timestamp traces, including the final implementation, rather than inferred from numerical output.

| Projection (K, N) | A: S / S | B: R / R | C: timing and path | D final: timing and path |
| --- | ---: | ---: | --- | --- |
| MLP down (17408, 5120) | 1233.5 / 1232.5 | 253 / 464 | 253 R / 463 R | 253 R / 463 R |
| MLP gate/up (5120, 17408) | 441 / 444 | 244 / 456 | 441 S / 446 S | 244 R / 444 S |
| Linear-attention QKV (5120, 10240) | 369 / 364 | 151 / 280 | 151 R / 280 R | 151 R / 280 R |
| Linear-attention output (6144, 5120) | 379 / 387 | 97 / 173 | 97 R / 173 R | 97 R / 173.5 R |
| Attention Q, unfused (5120, 12288) | 378 / 379 | 178 / 331 | 177 R / 331 R | 177 R / 331 R |
| Attention K/V, unfused (5120, 1024) | 319 / 319 | 39 / 48 | 39 R / 48 R | 40 R / 47 R |
| Attention QKV, current fused export (5120, 14336) | 432 / 434 | 204.5 / 379 | 432 S / 437 S | 205 R / 379 R |
| Output head (5120, 248320) | 2575 / 2564 | 3044 / 6055 | 2559 S / 2583 S | 2573.5 S / 2583 S |

B fixes the repeated hidden projections but makes the M=8 head about 2.36x slower, and makes
M=8 gate/up about 2.7% slower. C avoids those regressions but leaves the substantial M=4 gate/up
and fused-QKV gains unused. D captures those gains without a shape lookup table or device-name rule.
Small differences between repeated timings of an unchanged shader are measurement variation.

In separate final Engine diagnostics, each full target verification contained 369 INT4 matrix
dispatches. At width 3/M=4, D selected 368 regular and one subgroup dispatch (the head).
At width 7/M=8, it selected 240 regular and 129 subgroup dispatches (128 gate/up plus the head).
The initial prefill remained subgroup, and the drafter also benefited where its fixed eight-row
projections satisfied the rule; end-to-end results are not target-only speedups.

## Focused DFlash2 end-to-end results

The existing short workload had 86 prompt tokens and generated 128 tokens. Each configuration
had one full-length warmup and three measured runs, greedy decoding, prefix caching disabled,
and graph capture disabled. Total generation latency includes prefill and decode, but excludes
model loading and Engine startup. Diagnostic runs were separate.

| Draft width | A median seconds | D median seconds | A/D speedup | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| 3 | 14.8138 | 6.7346 | 2.200x | 54.54% |
| 7 | 13.0670 | 10.6250 | 1.230x | 18.69% |

Measured samples were A/w3 `[14.7713, 14.8138, 14.8596]`, D/w3
`[6.7602, 6.7235, 6.7346]`, A/w7 `[13.0980, 13.0670, 13.0218]`, and D/w7
`[10.6250, 10.6611, 10.6186]`. No DFlash2 failures, disablements, or admission misses occurred.

**These are not identical-work verification-only ratios.** Width-3 greedy outputs first differed
at zero-based output index 67; completed verification rounds changed from 42 to 38 and accepted
draft tokens from 84 to 89. Width-7 final output IDs matched A for all 128 tokens, but verification
rounds still changed from 32 to 33 and accepted drafts from 95 to 94.

The initial 96K-budget prototype measured 7.0471 s / 10.4093 s at widths 3 / 7. The final rule's
width-7 wall time was about 2.1% higher than that prototype despite the faster isolated fused-QKV
choice; the prototype also produced different output IDs and acceptance. These short, grouped
runs do not establish a statistically optimal threshold. The 128K rule is preferred as a simple
candidate supported by the per-shape measurements, not as a proven global end-to-end optimum.

## Correctness, boundaries, and regression checks

All 16 real-weight cases were compared with A and an independent FP64 dequantization/matmul
reference. Changed outputs were not bitwise identical: relative L2 differences from A reached
0.7102%, and the largest absolute difference was 0.1015625. The regular outputs were closer to
the FP64 reference in every changed case, with relative L2 error 0.1074-0.1226%. The M=8 gate/up
and both head cases retained bitwise-identical outputs.

The final routing harness ran 29 cases per selector with full WebGPU validation and independent
numerical references. It covered M=1,3,4,5,6,7,8,9, both sides of each aligned N budget boundary,
N=65, non-block-aligned K, higher-rank activations, block64/block128, FP32, INT2/INT8, explicit
FP32 accumulation, and zero points/bias. Unchanged cases had identical program lists and
bitwise-identical outputs. `MatMulComputeHelper` flattens leading activation dimensions for 2D
weights, so `[2,4,K]` correctly exercises effective M=8 rather than a two-batch eligibility rejection.

Twelve non-Qwen synthetic shapes were also timed at M=4/8. Examples, A to D in microseconds:

| K, N | M=4 | M=8 |
| --- | ---: | ---: |
| 1024, 1024 | 94 to 27 | 95 to 32 |
| 4096, 4096 | 264 to 60 | 264 to 101 |
| 4096, 11008 | 287.5 to 133 | 288 to 244 |
| 4096, 16384 | 362 to 188 | 364 to 350 |
| 8192, 8192 | 590 to 188 | 591 to 349 |
| 1024, 32768 | 120 to 111 | 121 to 121 |

No changed-path regression was observed in this sweep. The ineligible K=96/N=65 control retained
wide tile and varied from 38 us to 38.5/39 us; the unchanged M=8 head varied by about 0.7%.
Host-only tests verified that wave64, `8x16x16`, `8x8x8`, and FP32 configurations decline the new
preference. Those are guard tests, **not performance validation on AMD, Apple, Intel, Qualcomm,
Metal, D3D12, browsers, or other GPUs**.

Nine focused C++ tests passed, including both existing subgroup tile tests, the new preference
and numerical boundary tests, accuracy-4 FP16, large-K accumulator overflow, FP32 accumulation
options, and subgroup configuration selection. Commands used for final build/checks:

```bash
source /workspace/venv/bin/activate
cmake --build build/webgpu-pipeline-lifetime/Release --target onnxruntime_provider_test --parallel 32
export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd build/webgpu-pipeline-lifetime/Release
./onnxruntime_provider_test --gtest_filter='WebGpuMatMulNBitsTest.*:SubgroupMatrixConfigTest.*:MatMulNBits.Float16_4b_Accuracy4:MatMulNBits.Float16_AccumulatorPrecisionOption_AllPaths:MatMulNBits.Float16_LargeK_AccumulatorOverflow'
```

The build and nine-test command passed. `git diff --check` and changed-line
`git clang-format --diff --binary /workspace/venv/bin/clang-format HEAD` also passed.
Local raw profiles, clocks, binary manifests, benchmark runners, and results are retained under
`build/nbits-small-m-selector/results/`.

Before production adoption, validate the same configuration on other adapters, broaden prompt
lengths/workloads and speculative acceptance checks, and assess model-quality sensitivity to
the existing kernels' different FP16 reduction orders. Keep this as an experimental selector
candidate until that evidence exists; do not promote a universal M=9 cutoff.
