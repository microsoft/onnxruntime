# QMoE FP4 Profiling Experiments

This file records QMoE FP4 profiling results so future kernel and dispatch changes can be compared against a stable baseline.

## 2026-06-14 FP4 (MXFP4) Dequant-Fallback Decode Bottleneck: SM90 (H200), GPT-OSS-20B, Batch 1

### Motivation

End-to-end benchmarking showed the exported FP4 (MXFP4) QMoE gpt-oss-20b decoding
~50x slower than the INT4 model on the same H200 (decode ~4.85 tps / 206 ms/tok
vs INT4 ~253 tps / 3.95 ms/tok). The expectation was that because H200 (SM90)
lacks the native FP4 tensor-core path, FP4 weights are dequantized to BF16/FP16
and run through the dense A16 grouped-GEMM, so FP4 should be at most a small
constant slower than INT4 -- not 50x. This experiment profiles the kernels to
locate the real cost.

### Setup

- Machine/GPU: 1x NVIDIA H200, SM90 (Hopper), `CUDA_VISIBLE_DEVICES=0`.
- ONNX Runtime build: `~/onnxruntime/build/cu130/Release` (built with
  `onnxruntime_USE_FP4_QMOE=ON`; `-DUSE_FP4_QMOE -DENABLE_FP4`).
- onnxruntime-genai: `benchmark/python/benchmark_e2e.py`.
- Models: `~/gptoss20b_fp4_qmoe` (quant_type=fp4, block_size=32) and
  `~/gptoss20b_int4_qmoe_bs32_ropefix` (quant_type=int, 4-bit, block_size=32).
- Nsight Systems: `~/cuda13.0/bin/nsys` 2025.3.2.
- CUDA graph forced OFF (so individual dequant + GEMM kernels are attributable).
- Shapes: gpt-oss-20b -- hidden 2880, intermediate 2880, 24 MoE layers,
  32 experts, top-k 4, fused interleaved SwiGLU.
- Driver script: `~/onnxruntime/profile_fp4_vs_int4_decode.sh`
  (prompt 64, gen 64, repeat 1, warmup 1, `nsys profile --trace=cuda,nvtx`).
- Report: `nsys stats --report cuda_gpu_kern_sum`.

Command template:

```bash
TMPDIR=/tmp/qmoe_fp4_profile/nsystmp \
~/cuda13.0/bin/nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/qmoe_fp4_profile/fp4_decode \
  python benchmark/python/benchmark_e2e.py -i ~/gptoss20b_fp4_qmoe -e cuda \
    -b 1 -l 64 -g 64 -r 1 -w 1 --use_random_tokens --chat_template "{input}" \
    -mn gpt-oss-20b -pr fp4
~/cuda13.0/bin/nsys stats --report cuda_gpu_kern_sum fp4_decode.nsys-rep
```

### End-To-End Decode Throughput (CUDA graph OFF)

| Model | Prefill tps | Decode tps | Decode ms/tok | Ratio vs INT4 |
|-------|-------------|------------|---------------|---------------|
| INT4 (int, 4-bit) | 5570.2 | 252.9 | 3.95 | 1.0x |
| FP4 (MXFP4) | 302.3 | 4.85 | 206.0 | 52.2x slower |

### Top GPU Kernels (decode, nsys `cuda_gpu_kern_sum`)

FP4 -- a single kernel dominates:

| Time % | Total (s) | Instances | Avg (ms) | Kernel |
|--------|-----------|-----------|----------|--------|
| 78.2 | 25.72 | 6144 | 4.187 | `QMoEDequantizeFp4WeightsKernel<__half>` |
| 1.2 | 0.41 | 624 | 0.658 | `cutlass GemmUniversal GroupProblemShape` (A16 dense GEMM) |
| 1.2 | 0.39 | 1248 | 0.313 | `cutlass MoeFCGemm MmaMultistage` |
| ... | ... | ... | ... | remaining A16 grouped-GEMM tiles |

INT4 -- no all-expert dequant kernel; time is spread across the fused
mixed-precision `MoeFCGemm ... DqMmaMultistage` GEMMs (dequant fused *inside*
the GEMM) plus small GEMV/activation kernels:

| Time % | Total (s) | Instances | Avg (ms) | Kernel |
|--------|-----------|-----------|----------|--------|
| 9.8 | 0.36 | 2496 | 0.144 | `MoeFCGemm ... DqMmaMultistage` (fused dequant+GEMM) |
| 8.0 | 0.29 | 1248 | 0.236 | `MoeFCGemm ... DqMmaMultistage` |
| ... | ... | ... | ... | (no separate dequant kernel) |
| 1.6 | 0.06 | 7560 | 0.008 | `MatMulFloatInt4Kernel` |
| 1.5 | 0.06 | 3024 | 0.018 | `moe_gemv_interleaved_swiglu_kernel` |

### Root Cause

The FP4 dequant fallback (`moe_quantization.cc`, `ComputeInternal`, the
`(is_fp4 && use_fp4_dequant_fallback_)` branch, lines ~924-967) dequantizes the
**entire** MXFP4 weight tensor for **all 32 experts** into a BF16/FP16 scratch
buffer on **every forward pass**, then hands the dense weights to the A16
grouped-GEMM runner:

```cpp
size_t fc1_bytes = num_experts * fc1_n * fc1_k * element_size;  // ALL experts
dequant_fc1_weights = GetScratchBuffer<void>(fc1_bytes, ...);
LaunchQMoEDequantizeFp4Weights(... fc1 all experts ...);        // every Compute()
LaunchQMoEDequantizeFp4Weights(... fc2 all experts ...);
```

`QMoEDequantizeFp4WeightsKernel` is called 48 times per token
(24 layers x {fc1, fc2}) at ~4.19 ms each = **~201 ms/token**, which is
essentially the *entire* measured 206 ms/token decode latency.

Two compounding problems, both independent of FP4 arithmetic:

1. **Dequantizes all experts, not just the top-k.** Decode routes 4 of 32
   experts per token, but the fallback materializes all 32. ~8x wasted work.
2. **Materializes full BF16 weights to HBM, then re-reads them in the GEMM
   (un-fused).** Analytical per-MoE-layer weight traffic:
   FP4 fallback ~2190 MB (398 MB FP4 read + 1593 MB BF16 write + 199 MB GEMM
   read) vs INT4 fused ~50 MB (top-k INT4 read, dequant fused in-GEMM register).
   That is **~44x** more weight memory traffic -- matching the measured 52x
   end-to-end slowdown (memory-bound decode).

So the user's intuition is correct: FP4-dequant-to-BF16 *math* is comparable to
INT4. The 50x is **not** an FP4 compute cost -- it is the fallback's
**all-expert, un-fused, every-step weight dequantization** dominating a
memory-bound decode.

### Recommendations (not yet implemented)

- **Fuse FP4 dequant into the grouped-GEMM** like INT4's `DqMmaMultistage`
  (dequantize MXFP4 in-register inside the MoE GEMM mainloop). This is the real
  fix: removes the 1593 MB/layer BF16 write + re-read and limits work to the
  routed experts. Closes nearly all of the gap.
- **Short term:** only dequantize the top-k selected experts (gather by the
  routing map) instead of all 32. Cuts dequant cost ~8x even before fusion.
- **Caching is not viable for decode** because weights stay quantized in HBM by
  design; the cost is the per-step dense materialization, not a one-time setup.
- The native FP4 tensor-core path (`use_fp4_dequant_fallback_ = sm_ < 120`) is
  gated to SM120+ (Blackwell), so H200/SM90 cannot use it; the fused-dequant
  grouped-GEMM is the only path that helps Hopper.

### Artifacts

- `~/onnxruntime/profile_fp4_vs_int4_decode.sh`
- `/tmp/qmoe_fp4_profile/{fp4,int4}_decode.nsys-rep`, `*_run.log`

### How llama.cpp and vLLM Avoid This on H200 (reference-engine survey)

Both faster engines keep MXFP4 weights **4-bit in HBM** and fuse the
dequantization **inside** the matmul mainloop. Neither ever materializes a dense
BF16/FP16 weight tensor, and both touch only the routed (top-k) experts.

**llama.cpp (CUDA, decode / batch 1) -- fused MXFP4 GEMV via integer dp4a.**
- `ggml/src/ggml-cuda/mmvq.cu` + `vecdotq.cuh::vec_dot_mxfp4_q8_1`.
- The activation row is quantized to `Q8_1` (int8 + per-block scale).
- Weights stay MXFP4. A 16-entry LUT `kvalues_mxfp4` maps each 4-bit e2m1 code
  to an int8 value (`get_int_from_table_16`), then `dp4a` (int8x4 dot) accumulates:
  ```cpp
  const int2 v = get_int_from_table_16(aux_q4, kvalues_mxfp4); // 4-bit -> int8 LUT
  sumi = ggml_cuda_dp4a(v.x, q8[l+0], sumi);                   // fused int8 dot
  sumi = ggml_cuda_dp4a(v.y, q8[l+4], sumi);
  const float d = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * d_act; // e8m0 block scale
  return d * sumi;
  ```
- No dense weight write/re-read; weight traffic = 4-bit, top-k experts only.

**vLLM (Hopper / SM90) -- fused MXFP4 grouped-GEMM, weights stay 4-bit.**
- `select_mxfp4_moe_backend` on capability 90 uses the OpenAI **Triton** MXFP4
  kernels (`gpt_oss_triton_kernels_moe.py`, StridedLayout) -- a grouped GEMM that
  dequantizes MXFP4 in the Triton mainloop.
- Alternative `MarlinMxFp4LinearKernel` (`mxfp4/marlin.py`,
  `apply_fp4_marlin_linear`) is a fused-dequant W4A16 Marlin GEMM (4-bit weights
  dequantized in-register, same principle as ORT's INT4 `DqMmaMultistage`).
- Native CUTLASS block-scaled FP4 tensor-core path is reserved for SM100+
  (`is_device_capability_family(100)`), mirroring ORT's `sm_ < 120` gate.

### Conclusion: ORT needs a fused FP4 path, not the all-expert dequant fallback

The user's hypothesis is confirmed by both engines: the win is **fusing FP4
dequant into the GEMM/GEMV** so weights stay 4-bit and only top-k experts are
read -- exactly what ORT already does for INT4. ORT's INT4 fused machinery is
the natural host:

- **GEMV (decode, batch 1):** `onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemv/`.
  The in-register weight decode is `details.h::I2FConverter` /
  `FastInterleavedAndBiasedNumericArrayConverter` (INT4/INT8 -> half/bf16). Add
  an **FP4 (e2m1) converter** (16-entry LUT `{0,.5,1,1.5,2,3,4,6}` with sign,
  like llama.cpp's `kvalues_mxfp4`, or a `prmt`/bit-twiddle convert) plus the
  per-block **e8m0** scale, and register `FP16Fp4*/BF16Fp4*` `KernelType`s in the
  `KERNEL_TYPE_TRAITS_REGISTRY`. Then relax `is_moe_gemv_supported` so MXFP4
  (block_size 32) routes to GEMV.
- **GEMM (prefill / batched):** route MXFP4 through the existing
  `MoeFCGemm ... DqMmaMultistage` fused-dequant grouped-GEMM (the INT4 path)
  with an e2m1+e8m0 dequant functor, instead of constructing the dense A16 runner.
- **Net effect (projected):** removes the 1.6 GB/layer BF16 write+re-read and the
  ~8x all-expert waste, bringing FP4 decode close to INT4 (INT4 ~253 tps vs FP4
  ~4.85 tps today). This is the "special GEMV kernel for FP4" needed.

### Proposed ORT FP4 GEMV (W4A16) vs llama.cpp (W4A8): Accuracy and Performance

The proposed ORT path reuses the INT4 `fpA_intB_gemv` machinery (FP16/BF16
activations, MXFP4 weights dequantized in-register) -- i.e. **W4A16**.
llama.cpp's `vec_dot_mxfp4_q8_1` quantizes the activation to int8 (`Q8_1`) and
uses integer `dp4a` -- i.e. **W4A8**. The single design difference is the
activation precision.

| | ORT proposed (W4A16) | llama.cpp (W4A8) |
|---|---|---|
| Weight decode | e2m1 -> FP16/BF16 in-register | e2m1 -> int8 LUT in-register |
| Activation | FP16/BF16 (exact) | int8, Q8_1 per 32-block (lossy) |
| Inner product | FP16/BF16 FMA, FP32 accum | int8 `dp4a`, int32 accum |
| e8m0 block scale | exact pow-2 multiply | exact pow-2 multiply |

#### Accuracy -- ORT W4A16 is strictly more accurate

- Both decode **weights exactly**: e2m1 values `{0, +/-0.5, +/-1, +/-1.5, +/-2,
  +/-3, +/-4, +/-6}` times a power-of-two e8m0 scale are dyadic, so they are
  lossless in FP16, BF16, and as llama.cpp's int8 LUT.
- The difference is the **activation**. W4A16 keeps activations full precision,
  so the result is bit-equivalent to "dequantize weights then matmul" (the MXFP4
  reference). W4A8 adds a per-block int8 rounding error on the activation that
  compounds with depth.
- Net: the proposed ORT path matches a BF16 reference; llama.cpp trades a small
  amount of activation precision for integer throughput.

#### Performance -- equal on the decode MoE op; llama.cpp ~10-17% faster e2e

- Batch-1 decode MoE GEMV is **memory-bound on the 4-bit weights**. Both engines
  read the same bytes (top-k experts, 4-bit, no BF16 materialization), so the
  FP4-GEMV lower bound is essentially identical. The fixed ORT path should land
  next to ORT INT4 today.
- Reference e2e decode: ORT INT4 **253-271 tps**, llama.cpp MXFP4 **297 tps**,
  vLLM ~278 tps, ORT FP4 today (broken fallback) **4.85 tps**.
- So even after the fix llama.cpp likely stays **~10-17% ahead**, but that
  residual gap is **mostly not the FP4 kernel** -- it is framework/host overhead
  (per-op kernel launches, separate layernorm/RoPE/activation kernels, the genai
  Python loop) and attention. `dp4a` (W4A8) only pulls ahead in **compute-bound**
  regimes (prefill / larger batch), where it roughly doubles MMA throughput.

#### How to close the gap

1. **Cut host/launch overhead first (biggest lever).** The MoE GEMV equalizes,
   so the gap is elsewhere. Enable CUDA graphs for decode and fuse
   layernorm + RoPE + SwiGLU to drop per-token launch count toward llama.cpp's
   fused graph. Confirm with a host-overhead / GPU-idle-ratio profile.
2. **Add an opt-in W4A8 `dp4a` FP4 GEMV** mirroring llama.cpp for the
   **prefill / batched** path where it genuinely wins. Keep W4A16 as the accurate
   default for decode; expose A8 as a "fast" mode. ORT already has a wfp4afp8
   (W4A8) concept for SM100+ -- extend a `dp4a` variant down to SM90.
3. **Tune the GEMV for gpt-oss shapes** (2880x2880, top-4, 32 experts): 128-bit
   vectorized weight loads, e2m1 LUT in constant memory, double-buffered weight
   tiles. Reuse the INT4 tile-shape probe methodology already in this doc.
4. **Maximize weight coalescing in the expert gather** (row-to-expert map) so the
   4-bit reads hit peak HBM bandwidth -- the actual decode ceiling.
5. **Keep FP32 accumulation** (ORT default) for accuracy; it is free vs
   llama.cpp's int32 accum because decode is memory-bound.

**Bottom line:** W4A16 FP4 GEMV is the right default -- more accurate than
llama.cpp and on par on the decode MoE op. To beat the remaining ~10-17% e2e
gap, attack host/launch overhead and add an opt-in `dp4a` W4A8 path for prefill.

## 2026-06-14 Phase 0: Top-K-Only Dequant Fallback (active-expert mask)

### Motivation

The first short-term recommendation above ("only dequantize the top-k selected
experts instead of all 32") is the lowest-risk, no-new-files change. Decode
routes 4 of 32 experts per token, so the all-expert FP4 dequant fallback wastes
~8x of its work on experts that the grouped-GEMM never reads. This phase skips
the untouched experts while keeping the existing dequant-to-dense path.

### Change

- `qmoe_kernels.cu`: added `QMoEBuildActiveExpertMaskKernel` (builds a per-expert
  0/1 mask from the `expert_indices` routing table, length `num_rows * top_k`)
  and `LaunchQMoEBuildActiveExpertMask` (memset 0 + launch). Added a
  `const int* expert_active` parameter to `QMoEDequantizeFp4WeightsKernel<T>`
  with an early `return` when `expert_active[expert] == 0`. Threaded the param
  through `LaunchQMoEDequantizeFp4WeightsImpl<T>` and both
  `LaunchQMoEDequantizeFp4Weights` overloads (half + bf16).
- `qmoe_kernels.h`: matching declarations.
- `moe_quantization.cc`: in the FP4 fallback branch, allocate an
  `expert_active` scratch buffer (`num_experts` ints), build the mask from
  `expert_indices` (already computed by the top-k softmax), and pass it to all 4
  `LaunchQMoEDequantizeFp4Weights` calls (fc1/fc2 x fp16/bf16).
- Safe because grouped-GEMM only reads experts with assigned rows; skipped
  experts leave stale scratch the GEMM never touches.

### Accuracy (gate -- run BEFORE perf)

`onnxruntime/test/python/transformers/test_qmoe_fp4_cuda.py`: **15/15 PASS**
(`Ran 15 tests OK`). Covers token_counts (8/16/32/64/128), 4 and 8 experts,
SiLU and SwiGLU, FP16 and BF16, top-4 routing -- exercises the active-expert
mask. Max abs diff vs torch MXFP4 reference unchanged (e.g. FP16 SiLU
`max_diff=0.001953`, BF16 SwiGLU `max_diff=0.0625`), well within
`atol=0.12` (fp16) / `0.15` (bf16). Output is bit-identical to the all-expert
fallback.

### End-To-End Decode Throughput (CUDA graph OFF, same driver as baseline)

| Model | Decode tps | Decode ms/tok | vs INT4 |
|-------|------------|---------------|---------|
| INT4 (int, 4-bit) | 250.5 | 3.99 | 1.0x |
| FP4 baseline (all-expert) | 4.85 | 206.0 | 52.2x slower |
| **FP4 Phase 0 (top-k mask)** | **12.11** | **82.58** | **20.7x slower** |

Phase 0 lifts FP4 decode **~2.5x** (206.0 -> 82.58 ms/tok; 4.85 -> 12.11 tps).

### Top GPU Kernels (decode, nsys `cuda_gpu_kern_sum`)

| Time % | Total (s) | Instances | Avg (ms) | Kernel |
|--------|-----------|-----------|----------|--------|
| 58.3 | 10.01 | 6144 | 1.629 | `QMoEDequantizeFp4WeightsKernel<__half>` |
| 2.4 | 0.42 | 624 | 0.666 | `cutlass GemmUniversal GroupProblemShape` (A16 dense GEMM) |
| 2.3 | 0.39 | 1248 | 0.313 | `cutlass MoeFCGemm MmaMultistage` |
| ... | ... | ... | ... | remaining A16 grouped-GEMM tiles |

Dequant total dropped **25.72 s -> 10.01 s** and avg **4.187 -> 1.629 ms**
(grid still launches over all experts and early-returns, so the instance count
stays 6144; the skipped experts no longer do the FP4 decode + BF16 HBM write).

### Remaining Gap and Next Step

Dequant still dominates at **58.3%** of GPU time. Even limited to top-k experts,
the fallback **materializes dense BF16 weights to HBM and re-reads them in the
GEMM** (the un-fused round-trip is the second, larger problem from the root-cause
analysis). The ~20x residual vs INT4 is this dense weight traffic. The real fix
remains the **fused FP4 GEMV/GEMM** (weights stay 4-bit, dequant in-register),
implemented in the following phases. Phase 0 is a safe, accuracy-neutral
intermediate win that ships independently.

### Artifacts

- `/tmp/qmoe_fp4_profile/{fp4,int4}_decode.nsys-rep`, `*_run.log`,
  `phase0_profile.log`
- Provider lib under test: `~/ort_home_cu130/lib/libonnxruntime_providers_cuda.so`
  (pre-change backup: `*.pre_phase0`).

## 2026-06-14 Phase 1: MXFP4 (e2m1) In-Register GEMV Converter (foundation)

### Motivation

The real fix is a **fused FP4 GEMV** that keeps weights 4-bit in HBM and decodes
e2m1 in-register inside the MoE GEMV mainloop -- mirroring the INT4
`fpA_intB_gemv` path that already wins on decode. The foundational, independently
testable piece is the in-register **e2m1 -> half/bf16 converter** that plugs into
the existing `dequantize<>()` dispatch in
`fpA_intB_gemv/dispatcher.h`. Phase 1 adds that converter and the supporting
type plumbing **additively** (no existing kernel behavior changes; nothing is
routed to it yet).

### Change (additive, not yet wired into dispatch)

`onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemv/details.h`:
- New `Fp4DetailsW { kElemBits = 4; }` -- same 4-bit storage as `Int4DetailsW`
  but classified via an opt-in `IsFp4Weight<T>` trait (defaults to false; only
  `Fp4DetailsW` specializes it to true). This keeps the integer weight
  descriptors free of FP4-specific members.
- New `Fp4I2FConverter<AType>` (AType = half | bf16): decodes e2m1 codes packed
  two per byte (low nibble = even element) into half/bf16 using the magnitude LUT
  `{0, .5, 1, 1.5, 2, 3, 4, 6}` indexed by `code & 0x7` with sign bit `code & 0x8`
  -- identical to `DecodeFp4E2M1` in `qmoe_kernels.cu`. All e2m1 values are
  exactly representable in fp16/bf16, so the convert is lossless.
- `ConverterWrapper<Details>` now selects `Fp4I2FConverter` when
  `IsFp4Weight<TypeDetailsW>::value`, else the existing integer `I2FConverter`
  (via `std::conditional_t`). The INT4/INT8 paths resolve exactly as before.

The packed nibble order the converter expects matches the existing
`QMoERepackFP4ColToRowKernel` output (`[expert, n, k/2]` row-major, two K-values
per byte, even K = low nibble) -- i.e. the non-interleaved `ColumnMajor`
(`kInterleave = 1`) GEMV weight layout. This is the layout Phase 3 will produce
in PrePack.

### Accuracy (gate)

Standalone CUDA harness `/tmp/fp4_converter_test/test_fp4_converter.cu`
(`nvcc -arch=sm_90`) runs `Fp4I2FConverter<half>` and `Fp4I2FConverter<bf16>`
over a 256-code stream exercising all 16 e2m1 codes in both nibble positions and
compares against `DecodeFp4E2M1`:

```
[fp16] PASS (256/256 match)
[bf16] PASS (256/256 match)
ALL PASS
```

In-tree compile verified: incremental `ninja onnxruntime_pybind11_state`
recompiles `moe_gemv.cu` (which transitively includes the modified `details.h`)
and links `libonnxruntime_providers_cuda.so` cleanly. The INT4/INT8 GEMV paths
are unchanged (no perf run -- Phase 1 adds no active kernel).

### Remaining Integration (Phases 2-4, hardware-iterative)

To turn the converter into an end-to-end win, the following must be wired and
verified numerically against the dequant-fallback reference (recommended via an
env-gated opt-in path so the proven Phase 0 fallback stays the default until the
GEMV is proven bit-accurate):

- **Phase 2 (scales):** build a per-expert `[k/32, n]` half/bf16 scale tensor
  `scale[g, n] = exp2(e8m0[n, g] - 127) * global[expert]` -- a transpose +
  e8m0-decode + global-fold of the MXFP4 block scales `[expert, n, k/32]` into
  the groupwise (GroupSize = 32) layout the GEMV indexes
  (`real_offset_k / GroupSize * n + real_offset_n`).
- **Phase 3 (weight + scale layout via PrePack):** in the `weights_prepacked = 0`
  PrePack hook, run `LaunchQMoERepackFP4ColToRow` for the weights and the Phase 2
  scale-combine kernel, caching packed buffers in members.
- **Phase 4 (dispatch + routing):** add an `Fp4` `KernelDetails`
  (`ColumnMajor`, `UseInterleavedConverter = false`, `Fp4DetailsW`), a
  `tryLaunchMoeGemvFp4` sibling of `tryLaunchMoeGemvIntSymmetric[InterleavedSwiGLU]`
  in `moe_gemm/moe_kernels.cu`, extend `is_moe_gemv_supported` for FP4
  block_size 32, and route SM<120 FP4 decode (`rows <= 8`) to the fused GEMV
  (reusing the INT MoE prologue that builds `expert_first_token_offset` /
  `permuted_row_to_expert`). Keep the Phase 0 fallback for unsupported
  shapes/regimes. Verify with the python FP4 accuracy test and an
  `ORT_DISABLE_MOE_GEMV` on/off parity check, then profile.

### Artifacts

- `/tmp/fp4_converter_test/test_fp4_converter.cu` (+ compiled `test_fp4_converter`)
- `/tmp/fp4_phase1_ninja.log` (in-tree compile log)

## 2026-06-14 Phase 2: MXFP4 Scale-Combine Kernel for the Fused GEMV (foundation)

### Motivation

The fused FP4 MoE GEMV (Phases 3-4) indexes its scales groupwise along K, then N
(`real_offset_k / GroupSize * n + real_offset_n`, GroupSize = 32) and expects them
in the activation dtype (half/bf16), already multiplied by the per-expert global
scale. The MXFP4 weights instead ship two separate scale tensors: e8m0 block
scales `[experts, n, k_blocks]` (uint8 power-of-two codes, `k_blocks = k / 32`)
and a per-expert float32 global scale `[experts]`. Phase 2 fuses the e8m0 decode,
the global-scale fold, and the `[n, k_blocks] -> [k_blocks, n]` transpose into one
kernel so the eventual GEMV reads a ready-to-use TypeA scale directly (no
dispatcher change needed).

### Change (additive, not yet wired into dispatch)

`onnxruntime/contrib_ops/cuda/moe/qmoe_kernels.cu` / `.h`:

- New `QMoECombineFp4ScalesForGemvKernel<T>` (T = half | `__nv_bfloat16`): one
  thread per `(expert, n, k_block)`; computes
  `gemv_scales[expert, g, row] = exp2(e8m0[expert, row, g] - 127) * global[expert]`
  (e8m0 code 0 -> 0). Output layout `[experts, k_blocks, n]` matches the INT MoE
  GEMV groupwise indexing.
- New `LaunchQMoECombineFp4ScalesForGemvImpl<T>` plus half / bf16 launcher
  overloads `LaunchQMoECombineFp4ScalesForGemv`, declared in the header next to
  `LaunchQMoEBuildActiveExpertMask`.

INT/FP8 paths untouched; nothing dispatches to this kernel yet.

### Accuracy (gate)

Standalone bit-exact check `/tmp/fp4_converter_test/test_fp4_scales.cu`
(`nvcc -arch=sm_90`) against a CPU reference (`exp2(code-127)*global`, transpose),
including the gpt-oss-20b shape (experts=8, n=2880, k_blocks=90):

```
[fp16]       PASS (2073600/2073600 match)
[bf16]       PASS (2073600/2073600 match)
[fp16-small] PASS (8192/8192 match)
ALL PASS
```

In-tree build (`ninja onnxruntime_pybind11_state`, real source files changed)
compiled and linked clean.

### Artifacts

- `/tmp/fp4_converter_test/test_fp4_scales.cu` (+ compiled `test_fp4_scales`)
- `/tmp/fp4_phase2_ninja.log` (in-tree compile log)

### Phase 3-4 Design: Confirmed Non-Interleaved `ColumnMajor` Layout

A code read of `moe_gemm/moe_gemv.cu` + `fpA_intB_gemv/dispatcher.h` (`epilogue`,
`swiglu_epilogue`, `warp_reduce_sum`) confirms the **non-interleaved `ColumnMajor`
(kInterleave = 1)** layout is the correct, lowest-risk target for the fused FP4
GEMV, and that it is mutually consistent with the building blocks already built and
verified:

- **Weight layout** the kernel expects for `kInterleave = 1`: `[experts, n, k/2]`
  row-major, two e2m1 codes per byte, even-K in the low nibble. This is exactly
  the output of the existing `LaunchQMoERepackFP4ColToRow` and the order the
  Phase 1 `Fp4I2FConverter` decodes (verified bit-exact).
- **Scale layout** the kernel indexes for `GroupSize = 32`, `kInterleave = 1`:
  `[experts, k/32, n]` (`real_offset_k / GroupSize * n + real_offset_n`,
  per-N stride = `kInterleave` = 1). This is exactly the Phase 2
  `LaunchQMoECombineFp4ScalesForGemv` output (verified bit-exact).
- **Reduction correctness for `kInterleave = 1`:** each of the 128 threads owns a
  strided `kStepK = 8` slice of K and accumulates partial sums for all `CtaN`
  columns. `warp_reduce_sum<1, ...>` runs the full 5-step butterfly (16,8,4,2,1)
  → full 32-lane warp sum; then `epilogue` / `swiglu_epilogue` sum across the
  `WarpNum = 4` warps via shared memory (only lane 0 writes per warp, since
  `lane_id < ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0`).
  This is a provably-correct full 128-thread K-reduction for the per-channel /
  `fc2` path. The `fc1` SwiGLU path additionally requires the fc1 columns to be
  **(gate, linear) pair-interleaved** (`out[pair]` reads `shmem[..gate_idx]` /
  `shmem[..linear_idx]` with `gate_idx = 2*pair`), which the PrePack step must
  guarantee for the MXFP4 fc1 weights -- the one remaining item to verify on
  hardware against the dequant-fallback reference.

Remaining (hardware-iterative, env-gated opt-in so Phase 0 stays default):

- **Phase 3 (PrePack):** in the `weights_prepacked = 0` hook, run
  `LaunchQMoERepackFP4ColToRow` (weights) + `LaunchQMoECombineFp4ScalesForGemv`
  (scales) once at load time, caching the packed buffers in members; ensure fc1
  emits gate/linear pair-interleaved columns.
- **Phase 4 (dispatch + routing):** add an `Fp4` `KernelDetails`
  (`ColumnMajor`, `UseInterleavedConverter = false`, `Fp4DetailsW`); a
  `tryLaunchMoeGemvFp4` sibling in `moe_kernels.cu`; extend
  `is_moe_gemv_supported` for FP4 block_size 32; and route SM < 120 FP4 decode
  (`rows <= 8`) through the MoE-runner prologue
  (`expert_first_token_offset` / `permuted_row_to_expert`) to the fused GEMV,
  keeping the Phase 0 fallback for unsupported shapes. Validate with
  `test_qmoe_fp4_cuda.py` and an `ORT_DISABLE_MOE_GEMV` on/off parity check
  BEFORE profiling.

## 2026-06-14 Phase 4a: MXFP4 ColumnMajor GEMV Launchers + Device-Math Validation

### Motivation

Before wiring the fused FP4 decode path into the MoE pipeline (Phase 3 PrePack +
Phase 4 routing), the single highest-risk unknown was whether the non-interleaved
`ColumnMajor` (kInterleave = 1) GEMV kernel -- which the INT paths never exercise
(they all use `ColumnMajorInterleaved`) -- produces bit-correct output, especially
the warp/cross-warp reduction and the SwiGLU gate/linear pairing. A layout or
reduction bug here would be a silent wrong-output failure, so it must be proven
numerically first.

### Change (additive, not yet routed)

`onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemv.cu` / `.h`:

- `Fp4ADetails<T>` + `Fp4KernelDetails<T>` =
  `KernelDetails<FP16|BF16DetailsA, Fp4DetailsW, ColumnMajor, /*UseInterleavedConverter=*/false, 64>`.
- `is_moe_gemv_fp4_supported(sm, expanded_num_rows, n, k, group_size)`: kInterleave = 1
  variant of the INT support check (n divisible by `kCtaN`, not `kCtaN*interleave`;
  StepK = 128/16 = 8; group_size == 32).
- `launch_moe_gemv_fp4_symmetric<T>` and
  `launch_moe_gemv_fp4_symmetric_interleaved_swiglu<T>` (half + bf16 explicit
  instantiations), reusing the existing `dispatch_moe_gemv[_interleaved_swiglu]_group_size`
  templates with the FP4 `KernelDetails`.

INT/FP8 paths untouched. Nothing dispatches to these launchers yet.

### Accuracy (gate)

In-tree build (`ninja onnxruntime_pybind11_state`) compiled + linked the FP4 GEMV
launcher instantiations clean.

Faithful standalone replicas of the exact `ColumnMajor` device math (indexing,
`warp_reduce_sum` full butterfly for Interleave = 1, `epilogue` / `swiglu_epilogue`
shmem cross-warp sum) vs a CPU W4A16 reference:

```
# /tmp/fp4_converter_test/test_fp4_gemv.cu  (plain GEMV, fc2 path)
[fp16 gemv]  PASS (n=512  k=512,  max_rel=0.00047)
[fp16 gemv]  PASS (n=2880 k=2880, max_rel=0.00056)
[bf16 gemv]  PASS (n=512  k=512,  max_rel=0.0039)
[bf16 gemv]  PASS (n=2880 k=2880, max_rel=0.0038)
ALL PASS

# /tmp/fp4_converter_test/test_fp4_swiglu_gemv.cu  (fc1 SwiGLU path)
[fp16 swiglu] PASS (inter=512  k=512,  max_rel=0.0038)
[fp16 swiglu] PASS (inter=2880 k=2880, max_rel=0.011)
[bf16 swiglu] PASS (inter=2880 k=2880, max_rel=0.0039)
ALL PASS
```

Errors are at the fp16/bf16 accumulation-noise level (the reference accumulates in
float; the kernel accumulates in T). The SwiGLU replica confirms the required fc1
column ordering: output `j` reads gate = column `2*j`, linear = column `2*j+1`
(`out_base = tile_id_n * CtaN/2`, `gate_idx = 2*pair`). PrePack (Phase 3) must emit
fc1 weights with gate/linear columns pair-interleaved in this order.

### Remaining (Phase 3 PrePack + Phase 4 routing)

The fused decode path now has every device-side building block built and verified:
Fp4 converter (Phase 1), scale-combine (Phase 2), and both GEMV kernels (this phase).
The MoE prologue (`fusedBuildExpertMapsSortFirstToken`), activation expansion
(`expandInputRowsKernelLauncher`), and top-k finalize
(`finalizeMoeRoutingKernelLauncher`) are all directly-callable free/templated
functions, so a standalone SM<120 FP4 decode path (rows <= 8) can reuse them:
prologue -> expand -> fc1 `launch_moe_gemv_fp4_symmetric_interleaved_swiglu` -> fc2
`launch_moe_gemv_fp4_symmetric` -> finalize, with PrePack producing the repacked
weights ([E,n,k/2] via `LaunchQMoERepackFP4ColToRow`, fc1 gate/linear pair-interleaved)
and combined scales ([E,k/32,n] via `LaunchQMoECombineFp4ScalesForGemv`). This is
gated behind `ORT_ENABLE_FP4_GEMV` (default OFF) and validated against
`test_qmoe_fp4_cuda.py` + an `ORT_DISABLE_MOE_GEMV` parity check BEFORE profiling.

### Artifacts

- `/tmp/fp4_converter_test/test_fp4_gemv.cu` (+ compiled `test_fp4_gemv`)
- `/tmp/fp4_converter_test/test_fp4_swiglu_gemv.cu` (+ compiled `test_fp4_swiglu_gemv`)
- `/tmp/fp4_phase4a_ninja.log` (in-tree compile log)

## Phase 3 + 4 — PrePack GEMV weights/scales + standalone decode routing

Wires the Phase 1/2/4a building blocks into a complete, opt-in fused MXFP4 W4A16
decode path that replaces the all-active-expert dequant fallback for small-batch
decode shapes. Gated behind `ORT_ENABLE_FP4_GEMV` (default **OFF**); the proven
Phase 0 top-k dequant fallback remains the production default.

### Phase 3 — PrePack (`moe_quantization.{h,cc}`)

When `quant_type == "fp4"`, `use_fp4_dequant_fallback_` (SM < 120), and
`ORT_ENABLE_FP4_GEMV` is set, PrePack additionally produces the GEMV-consumed
buffers alongside (not replacing) the raw initializers the fallback still needs:

- **Weights** (inputs 2/5): `PrePackRepackFP4Weights` lays out the raw
  `[E, k, n/2]` col-major e2m1 codes into `[E, n, k/2]` row-major
  (`LaunchQMoERepackFP4ColToRow`) in `gemv_fp4_fc{1,2}_weights_`. A **local**
  `is_packed` is passed so the op-level `is_packed` stays `false` and the raw
  initializer is retained for the dequant fallback (unsupported shapes).
- **Scales**: block scales (inputs 3/6, raw `[E, n, k/32]` e8m0) and the
  per-expert global scale (inputs 15/16, float) arrive in separate PrePack calls.
  `TryBuildGemvFp4Scales(fc)` is invoked from **both** handlers and performs the
  combine (`LaunchQMoECombineFp4ScalesForGemv` -> `[E, k/32, n]` activation dtype,
  `exp2(e8m0-127) * global[e]`) exactly once, when both GPU buffers are present —
  making the build order-independent. Block-scale dims are captured at PrePack
  time (`gemv_fp4_fc{1,2}_scale_{e,n,kb}_`).

### Phase 4 — routing (`ComputeInternal`)

Before the dequant fallback, a standalone pipeline runs when all of: `is_fp4`,
`use_fp4_dequant_fallback_`, `enable_fp4_gemv_`, fused SwiGLU, all four GEMV
buffers built, `num_rows <= 256`, and `is_moe_gemv_fp4_supported` holds for both
fc1 (`n = 2*inter, k = hidden`) and fc2 (`n = hidden, k = inter`) — i.e. n,k >= 512
and `expanded_rows = num_rows * top_k` in `(0, 8]`.

Sequence (dispatched on `is_fp16_`, reusing the A16 runner's free functions):

```
fusedBuildExpertMapsSortFirstToken   # prologue: permuted maps + expert offsets
expandInputRowsKernelLauncher<T,T>   # gather permuted activations [expanded, hidden]
launch_moe_gemv_fp4_symmetric_interleaved_swiglu<T>   # fc1 + fused SwiGLU -> [expanded, inter]
launch_moe_gemv_fp4_symmetric<T>                      # fc2 -> [expanded, hidden]
finalizeMoeRoutingKernelLauncher<T,T,T>               # top-k weighted scatter-add -> output
```

Per-expert fc1/fc2 bias is applied inside the GEMV launchers; finalize bias is
`nullptr`. Scratch (`p_r2u`, `p_exp`, `p_efto`, `p_act`, `p_fc1`, `p_fc2`) comes
from `GetScratchBuffer`; `unpermuted_row_to_permuted_row` is reused from the
existing workspace. Unsupported shapes (prefill / large batch / missing buffers)
fall through to the dequant fallback. `MXFP4` group size is the intrinsic 32 — the
`block_size_` attribute is `-1` for fp4 and is **not** part of the gate (an early
`block_size_ == 32` gate was the reason the path was initially skipped; see below).

### Accuracy (gate)

New decode-shaped test `TestQMoEFP4.test_fp4_decode_swiglu_gemv` (hidden = inter =
512, 8 experts, SwiGLU, `num_tokens * top_k <= 8`) is the first case whose shape
satisfies `is_moe_gemv_fp4_supported`, so it actually exercises the fused path
under `ORT_ENABLE_FP4_GEMV=1` (a temporary `ORT_FP4_GEMV_TRACE` stderr marker
confirmed `[FP4_GEMV] taken` for all four parameterizations, then was removed):

```
# ORT_ENABLE_FP4_GEMV=1  (fused GEMV path taken)
FP4 MoE: FP16 SwiGLU tokens=1 experts=8 hidden=512 inter=512  max_diff=0.0117
FP4 MoE: FP16 SwiGLU tokens=2 experts=8 hidden=512 inter=512  max_diff=0.0156
FP4 MoE: BF16 SwiGLU tokens=1 experts=8 hidden=512 inter=512  max_diff=0.0781
FP4 MoE: BF16 SwiGLU tokens=2 experts=8 hidden=512 inter=512  max_diff=0.1016
```

All within tolerance (fp16 atol 0.12, bf16 atol 0.15). Full suite parity:

```
ORT_ENABLE_FP4_GEMV=1 : Ran 19 tests ... OK   # fused path for decode shapes
(default, unset)      : Ran 19 tests ... OK   # dequant fallback for all shapes
```

The fc1 gate/linear pair-interleaving (gate = col `2j`, linear = col `2j+1`),
bias-before-SwiGLU, and the `[E,n,k/2]` / `[E,k/32,n]` layouts are all validated
end-to-end by these passing decode tests against the dequantized-weight PyTorch
reference.

### Debugging note

Initial runs passed accuracy but silently used the fallback (the trace never
fired). A gate diagnostic showed `block=-1`: the MXFP4 group size is intrinsic
(32) and is not stored in `block_size_`. Removing the `block_size_ == 32` clause
let the path engage. Lesson: accuracy passing alone does **not** prove a new fast
path is taken — always confirm engagement (trace / profiler) before profiling.

## Phase 5 — End-to-end profiling on gpt-oss-20b (H200 / SM90)

Goal: measure whether the fused MXFP4 GEMV decode path (`ORT_ENABLE_FP4_GEMV=1`)
improves real decode throughput vs the Phase 0 dequant fallback, using
onnxruntime-genai `benchmark_e2e.py` on the gpt-oss-20b FP4 QMoE model
(hidden = inter = 2880, 32 experts, top_k = 4, 24 layers, fused SwiGLU).
Decode shape: `num_rows = 1`, `expanded = 1 * 4 = 4`; fc1 `n = 5760 / k = 2880`,
fc2 `n = 2880 / k = 2880` — all GEMV-supported.

### Fast-path engagement confirmed first

Before profiling, a temporary `ORT_FP4_GEMV_TRACE` marker (since removed) was
gated on `num_rows <= 8` to capture decode (not prefill) passes. It confirmed on
real gpt-oss-20b decode:

```
[FP4_GEMV_GATE]  ... enable=1 swiglu=1 w1=1 w2=1 s1=1 s2=1 num_rows=1 ...
[FP4_GEMV_SHAPE] expanded=4 fc1_n=5760 hidden=2880 inter=2880 sup_fc1=1 sup_fc2=1
```

All gates pass — the fused path **is** taken during decode when enabled.

### Library-sync gotcha

onnxruntime-genai's Python process loads `libonnxruntime_providers_cuda.so` from
the **onnxruntime pip package capi dir**
(`<venv>/lib/python3.X/site-packages/onnxruntime/capi/`), *not* from
`ort_home/lib/`. After every rebuild the fresh `.so` must be copied to **both**
locations (and `onnxruntime_pybind11_state.so` to the venv capi dir) or the
benchmark silently runs a stale binary. Two early "identical profile" runs were
caused by a lagging venv capi lib.

### Decode kernel attribution (nsys `cuda_gpu_kern_sum`)

- `QMoEDequantizeFp4WeightsKernel` runs in **prefill only** (`num_rows = 64`,
  GEMV-unsupported); decode never invokes it in any config.
- Decode MoE on this model already flows through `moe_gemv_*` kernels in **both**
  configs: the Phase 0 fallback dequantizes the active experts and the dense A16
  runner picks its own small-row `moe_gemv` fast path, while `ORT_ENABLE_FP4_GEMV=1`
  runs `moe_gemv` directly on the packed `[E,n,k/2]` weights. Instance counts and
  total kernel times are essentially identical between the two
  (`moe_gemv_interleaved` ~878 ms, `moe_gemv_kernel` ~478 ms across the whole
  capture in both off and on).

### Throughput results (batch 1, prompt 64, gen 64–128)

| Config                                   | CUDA graph | Token-gen throughput |
|------------------------------------------|------------|----------------------|
| `ORT_DISABLE_MOE_GEMV=1` (pure dequant+GEMM) | off    | 158.5 tps |
| default (fallback, dense `moe_gemv`)         | off    | 157.7 tps |
| `ORT_ENABLE_FP4_GEMV=1` (fused FP4 GEMV)     | off    | 159.0 tps |
| default                                      | **on** | 164.5 tps |
| `ORT_ENABLE_FP4_GEMV=1`                      | **on** | 164.7 tps |

nsys runs (gen 128) gave the same picture: off 155.5 tps vs on 153.1 tps.

### Conclusion

The fused MXFP4 GEMV path is **correct but not faster** for gpt-oss-20b decode on
H200/SM90. All configs land within ~1 % (noise) regardless of CUDA graph. Reasons:

- At `top_k = 4`, the Phase 0 fallback already dequantizes **only the active
  experts** and reaches an efficient small-row `moe_gemv`, so the theoretical win
  of the fused path (skipping the dense-weight HBM round-trip) is small.
- Batch-1 decode is dominated by fixed per-token / launch overhead and overall
  HBM traffic that is the same for both paths; the MoE GEMV cost is not the
  binding decode bottleneck here, so swapping the MoE kernel does not move e2e
  throughput.

The Phase 0 dequant fallback therefore remains the **production default**; the
fused GEMV path stays gated off behind `ORT_ENABLE_FP4_GEMV` as a validated
experiment. It may still pay off on shapes with larger `top_k`, more experts, or
batched decode where dense dequantization HBM traffic dominates — not the case
for this model.

---

## 2026-06-15 — FP4 GEMV converter: branch-free bit-math decode (compute-bound fix)

Revisited the gated `ORT_ENABLE_FP4_GEMV=1` path after the earlier "correct but
not faster" result. Fresh ncu profiling overturned the old memory-bound
assumption and produced a real decode speedup.

### Root-cause (ncu, decode, gpt-oss-20b FP4, H200/SM90)

The FP4 GEMV kernels were **compute(SM)-bound, not memory-bound**:

| Kernel (baseline)             | Duration | Compute(SM) | DRAM | Regs | Occ |
|-------------------------------|----------|-------------|------|------|-----|
| `moe_gemv_interleaved_swiglu` (fc1) | 96.8 us | **88.65 %** | 8.4 % | 75 | 33.4 % |
| `moe_gemv_kernel` (fc2)             | 51.3 us | **80.35 %** | 7.6 % | 46 | 42.6 % |

DRAM was only ~8 % (≈400 GB/s of 4.8 TB/s), so the 4× fewer weight bytes from
4-bit weights never helped — bandwidth was never the limiter. The bottleneck was
the in-register MXFP4 weight converter `Fp4I2FConverter::decode`, which used a
scalar lookup table:

```cpp
constexpr float kValues[8] = {0,0.5f,1,1.5f,2,3,4,6};
float v = kValues[code & 0x7];
return (code & 0x8) ? -v : v;
```

This compiled to a **data-dependent local-memory load + sign branch per nibble**,
inflating register pressure (75 regs/thread → spills → 33 % occupancy) and
saturating the SM pipes.

### Change (`fpA_intB_gemv/details.h`)

Replaced the LUT with a **branch-free bit-math** decode that synthesizes the
half/bf16 bit pattern directly from the 4-bit code (sign = bit3; magnitude 0..7
maps to {0,0.5,1,1.5,2,3,4,6}). Subnormal magnitudes (0,1) use a multiply,
normals (≥2) compose exponent/mantissa bits. Bit patterns are **exact** for both
`half` and `bfloat16`, so the result is numerically identical to the LUT.

### Results

Accuracy: `test_qmoe_fp4_cuda.py` **19/19 passed** with `ORT_ENABLE_FP4_GEMV=1`
and with the default (off) — bit-exact, zero accuracy change.

ncu after the fix:

| Kernel (after)                | Duration | Compute(SM) | DRAM | Regs | Occ |
|-------------------------------|----------|-------------|------|------|-----|
| `moe_gemv_interleaved_swiglu` (fc1) | 64.2 us (−34 %) | 79.9 % | 12.7 % | 54 | 47.1 % |
| `moe_gemv_kernel` (fc2)             | 38.2 us (−26 %) | 66.9 % | 10.2 % | 40 | 47.8 % |

nsys decode (gen 64, CUDA graph off), per-call averages:

| Kernel        | Before  | After   | Speedup |
|---------------|---------|---------|---------|
| fc1 GEMV      | 96.0 us | 63.5 us | 1.51×   |
| fc2 GEMV      | 53.0 us | 37.0 us | 1.43×   |

End-to-end decode throughput (`ORT_ENABLE_FP4_GEMV=1`, CUDA graph off):
**154.9 → 189.5 tps (+22 %)**.

### Conclusion

The converter was the binding decode bottleneck for the fused FP4 GEMV path. The
bit-math decode lowers register pressure, raises occupancy, and cuts both GEMV
kernels' SM time, yielding a real +22 % decode improvement. The path is still
somewhat compute-bound (fc1 ~80 % SM); further gains could come from vectorizing
the converter (half2) or raising `kCtaN`. The fix is header-only, bit-exact, and
still gated behind `ORT_ENABLE_FP4_GEMV`.

---

## 2026-06-15 — FP4 GEMV converter SWAR vectorization (negative result) + next-step diagnosis

Follow-up to the bit-math converter above. Hypothesis: the GEMV was still ~80 %
SM, so vectorizing the e2m1 decode (process both nibbles of a byte in one 32-bit
SWAR op, ~halving converter ALU) should cut more SM time.

### What was tried

Added `Fp4I2FConverter::decode2(byte)` — a SIMD-within-a-register decode that puts
the low nibble in the low 16-bit lane and the high nibble in the high 16-bit lane
of a `uint32`, then assembles both fp16/bf16 results with shared 32-bit ALU ops.
Verified bit-identical to two scalar `decode()` calls for all 256 byte values
(fp16 and bf16). Accuracy: `test_qmoe_fp4_cuda.py` 19/19 pass, on and off.

### Result — no net speedup

| Kernel   | scalar bit-math | SWAR decode2 | Δ |
|----------|-----------------|--------------|---|
| fc1 GEMV | 63.5 us | 61.6 us | −3 % |
| fc2 GEMV | 37.0 us | 39.0 us | +5 % |
| decode e2e | 189.5 tps | 188.1 tps | ~flat (noise) |

**The change was reverted** (no measurable e2e benefit, added intricate bit-math).

### Why — the bottleneck moved (ncu pipe/stall breakdown)

ncu `ComputeWorkloadAnalysis` + `WarpStateStats` on the current GEMV:

| Kernel | Compute(SM) | **ALU pipe** | Mem | Occ | Exec IPC | top stall |
|--------|-------------|--------------|-----|-----|----------|-----------|
| fc1 swiglu | 85.6 % | **90.0 %** | 25 % | 41.8 % | 2.73 | not-selected 36 %, math-throttle |
| fc2        | 66.5 % | **76.2 %** | 20 % | 43.6 % | 2.29 | short-scoreboard (smem) 33 % |

The saturated pipe is the **integer/logic ALU**, not FMA/float. After the first
fix the float decode is already a small slice; the ALU is now dominated by
**per-K-step loop, index, address and Mapper arithmetic**. The FP4 GEMV uses the
non-interleaved `ColumnMajor` layout with `AccessTypeW = int` (32-bit) and
`kStepK = 8`, so it runs **4× more K-steps** than INT4's `ColumnMajorInterleaved`
(`AccessTypeW = int4` 128-bit, `kStepK = 32`) — i.e. ~4× the integer loop
overhead. That, plus the low 42 % occupancy (which leaves the over-subscribed ALU
pipe latency exposed), is the real binding constraint. More converter micro-opt
cannot move an ALU pipe that is saturated by loop/index overhead.

### Recommended next change (the proper lever)

Widen the FP4 weight access to **128-bit** (`AccessTypeW = int4`, `kStepK = 32`)
so the K-loop runs 4× fewer trips, cutting the integer-ALU overhead that now
dominates. `kAccessNumW = 32*4/128 = 1` int4 load/step; `kAccessNumA = 32*16/128
= 4` activation float4 loads/step. The repacked weights are already `[E, n, k/2]`
row-major, so 16 contiguous bytes along K = 32 codes form a clean 128-bit load —
**no weight repack needed**. The reduction changes (`kThreadsPerInterleavedTile =
kTileSize/kStepK = 64/32 = 2` vs `64/8 = 8`), so correctness must be re-validated
with `test_qmoe_fp4_cuda.py` and the on/off parity gate. Raising occupancy (tune
`kCtaN`/threads) is a complementary lever to hide the ALU latency. The full
`kInterleave=4` interleaved layout (matching INT4, requires a weight prepack) is
the larger follow-up that also amortizes the cross-warp reduction 4×.

### Note on INT4

INT4 fc1 GEMV (~18 us) is ~38 % of the 6.9 us roofline, so there is ~2× headroom
there too. At batch-1 (M=1) it is likely reduction/occupancy-bound; a split-K
(2-pass / atomic) accumulation would expose more parallelism across the 132 SMs
for both INT4 and FP4. This is a separate, larger change.

## Experiment — wide (64-bit int2) weight load + FP32 accum (negative, reverted)

**Status:** tried, reverted. No net decode win. *Inefficient implementation*, not a
bad idea — recorded for revisiting via autotune.

**Code record (so it can be revisited):**

- Change commit: `7e9d6d7840` — *EXPERIMENT: FP4 GEMV 64-bit wide weight load
  (int2, StepK=16) + FP32 accum*
- Revert commit: `103a9e8724` — restores the clean `ColumnMajor` / bf16-accum
  baseline (identical to `3cb09b0`).

To revisit: `git show 7e9d6d7840` (or cherry-pick it onto a fresh branch).

### What was changed

Following the "widen the FP4 weight access" recommendation above, but tuned for
this shape. The 128-bit variant (`AccessTypeW = int4`, `StepK = 32`) was too
coarse: `StepK × Threads = 32 × 128 = 4096 > k = 2880`, leaving ~30 % of threads
idle and making fc1 **17 % slower** (74.3 µs). So the experiment used a **64-bit**
access instead:

- New `ColumnMajorFp4Wide` layout: `AccessTypeW = int2` (64-bit),
  `kStepK = 64/4 = 16` (vs the `ColumnMajor` baseline `kStepK = 128/16 = 8`),
  `kInterleave = 1`, identity Mapper. `StepK × Threads = 16 × 128 = 2048 < 2880`,
  so all 128 threads stay active.
- The longer per-thread accumulation chain broke bf16 accuracy (max_diff 0.1445,
  fragile), so accumulation was moved to **FP32** via a new
  `KernelDetails::kUseFloatAccum` (gated on `IsFp4Weight`), threaded through the
  `mma`/`epilogue`/`swiglu_epilogue` accumulators. That restored bf16 max_diff to
  **0.0625** (well under the 0.15 gate).

### Wall-time (nsys, gpt-oss-20b decode, H200, cuda-graph off)

| Variant | fc1 (swiglu) | fc2 | decode tps | bf16 max_diff |
|---|---|---|---|---|
| Baseline (StepK=8, bf16 accum) | **63.5 µs** | 37.0 µs | **189.5** | safe |
| 128-bit (StepK=32, fp32 accum) | 74.3 µs | 38.8 µs | ~181 | 0.0625 |
| 64-bit (StepK=16, fp32 accum) | 66.5 µs | 35.7 µs | 186.2 | 0.0625 |
| 64-bit (StepK=16, bf16 accum) | 63.9 µs | 35.5 µs | 189.3 | 0.1445 (fragile) |

### Why it didn't win — ncu (H200, `-b 1 -l 8 -g 6`)

| Metric | fc1 base → wide | fc2 base → wide |
|---|---|---|
| **ALU %** | 84.4 → **75.0** (−9.4) | 73.0 → **70.0** (−3.0) |
| SM % | 79.7 → 71.0 | 67.2 → 64.4 |
| FMA % | 23.0 → 26.9 | 20.0 → 24.9 |
| **occupancy %** | 47.1 → **38.1** (−9.0) | 47.8 → **35.0** (−12.8) |
| IPC | 2.6 → 2.7 | 2.2 → 2.4 |

The wide load **did** do what it was supposed to: the integer-ALU pipe pressure
dropped (fc1 −9.4 pts, fc2 −3.0 pts) because the K-loop runs half as many trips,
cutting per-step index/address/loop arithmetic. **But** the FP32 accumulators
(`float2` tile_acc vs the packed `half2`) raised register pressure and dropped
occupancy 9–13 pts. With fewer resident warps the freed ALU headroom can't be
filled (the over-subscribed pipe latency is no longer hidden), so wall-time is
flat. The bf16-accum 64-bit variant keeps occupancy and is ~even on time, but its
accuracy (0.1445) is too fragile to ship.

### Conclusion

Load-widening is a **mechanically sound** ALU-reduction lever (the ALU drop is
real and reproducible), but in this implementation the precision tax (FP32 accum
→ register pressure → occupancy loss) erases the gain. It is therefore an
**inefficient implementation**, not a dead idea. The right way to capture it is to
make `StepK` and the accumulation mode **per-shape autotuned knobs** (e.g. fc2 has
a shorter chain and more accuracy margin than fc1, so the win/precision trade-off
differs per kernel), paired with a register-pressure-aware accumulation strategy
(periodic FP32 flush, or smaller `CtaN`) so the wider load does not cost
occupancy. Search space for the autotuner:
`StepK ∈ {8,16,32} × Threads ∈ {64,128,256} × CtaN ∈ {4,8,16} × accum ∈ {bf16,fp32} × splitK`,
keyed on `(n, k, dtype, sm)`, gated on the accuracy check, and frozen at warmup
for cuda-graph compatibility.

---

## FP4 GEMV per-shape autotune (CtaN / Threads sweep)

After cherry-picking the QMoE GEMM↔GEMV autotune infrastructure (the
accuracy fix — FP32 accumulation + buffer-aliasing fix — plus the route/config
autotuner originally built for the INT path), the FP4 fused-SwiGLU GEMV decode
path was wired into the same per-shape config tuner.

### What is tuned

The first lever taken from the search space above is the **parallelization /
tiling** pair `{CtaN, Threads}` -- specifically the candidate set
`{kDefault (CtaN=8,Threads=128), kCtaN16 (CtaN=16), kThreads64 (Threads=64)}`.
These are **pure tiling knobs**: same reduction, same 16-bit (`AccT=T`)
accumulation, so the result is **bit-exact across all three configs**. That is
why this sweep needs **no accuracy gate** -- the profiling iterations double as
correct warmup work, and any config is safe to cache and replay.

`StepK` and the `accum` mode were deliberately left out of this first cut because
(per the negative result above) widening the load forces FP32 accumulation, which
costs occupancy — that lever needs the register-pressure-aware accumulation work
before it is worth searching.

### Implementation

- `launch_moe_gemv_fp4_symmetric<T>` and
  `launch_moe_gemv_fp4_symmetric_interleaved_swiglu<T>` now take a
  `MoeGemvConfig config` argument and dispatch `CtaN`/`Threads` as compile-time
  template params via a generic lambda. `is_moe_gemv_fp4_supported` gained a
  6-arg overload that re-checks `n % CtaNForConfig(config) == 0` so a wider
  `CtaN` is only offered when the output width divides evenly.
- In `moe_quantization.cc` the FP4 routing block builds a
  `Fp4GemvTuneKey{is_fp16, row_bucket, hidden, inter, sm}` and looks up a
  per-shape `Fp4GemvTuneResult{fc1_config, fc2_config}` cache. `row_bucket` is
  `MoeGemmProfiler::bucketM(expanded)` -- the expanded row count snapped up to a
  power of two -- so nearby row counts reuse one tune instead of re-profiling
  every distinct `expanded` value (the CtaN/Threads optima are essentially
  row-count independent in the tiny decode regime). On the first
  non-captured (warmup) call it CUDA-event-profiles each candidate for **fc1**
  and **fc2 independently** (kWarmup=3, kIters=20), caches the best, and replays
  the frozen choice thereafter.
- **cuda-graph safety:** unlike the INT autotuner (gated to prefill `rows>1`),
  FP4 GEMV is a decode path where graph capture happens, so tuning is gated on
  `cudaStreamIsCapturing` — it only profiles on non-captured calls and freezes
  the cached config for replay.
- **Gating:** FP4 GEMV stays behind `ORT_ENABLE_FP4_GEMV` (default off); the
  autotune is behind `ORT_FP4_GEMV_AUTOTUNE` (default **on** within the opt-in
  path); `ORT_FP4_GEMV_AUTOTUNE_LOG=1` logs the chosen config + timing per shape.

### Results (gpt-oss-20b FP4, H200, `-b 1 -l 512 -g 256`, cuda-graph off)

| Variant | decode tps |
|---|---|
| FP4 GEMV **ON + autotune** | **188.3** |
| FP4 GEMV ON, autotune off (kDefault) | 186.7 |
| FP4 GEMV **OFF** (dequant fallback) | 13.4 |

- Autotune over `{CtaN, Threads}` is **+0.9 %** over the fixed `kDefault` config,
  bit-exact, on the gpt-oss decode shapes (fc1 n=5760/k=2880, fc2 n=2880/k=2880).
  The win is modest because both kernels are already occupancy-limited near the
  same operating point; the `kCtaN16` candidate widens the tile but the chosen
  config is shape-dependent and the harness picks whichever profiles fastest.
- The dominant win remains the **GEMV fast path itself (~14×)** over the dequant
  fallback, which materializes **all 32 experts'** MXFP4 weights to dense HBM per
  token regardless of routing.

### Accuracy

`test_qmoe_fp4_cuda.py` is **19/19** with autotune on (default), autotune off,
and FP4 GEMV off — identical, as expected from the bit-exact tiling sweep. On a
microbenchmark of small shapes (hidden=inter=512) the tuner consistently selects
`Threads=64` for fc1/fc2, confirming the profiling path is live and per-shape.

### Conclusion

The `{CtaN, Threads}` sweep is the **safe, zero-accuracy-risk** first lever of
the autotuner: it lands a small but real, bit-exact decode gain and proves the
per-shape FP4 tune-cache + cuda-graph-safe warmup-freeze machinery end to end.
The larger predicted gains (`StepK`/`accum` load-widening) remain future work
gated on a register-pressure-aware accumulation strategy.

---

## 2026-06-16 — FP4 fc1 split-K (`kSplitK2`) autotune candidate (negative result, reverted)

Follow-up to the `{CtaN, Threads}` autotune. The earlier ncu diagnosis listed a
**split-K (2-pass) accumulation** as a way to "expose more parallelism across the
132 SMs" for the batch-1 decode GEMV, motivated by the ~42 % occupancy. The INT
GEMV already has a generic two-pass split-K SwiGLU path
(`launch_moe_gemv_splitk_twopass_swiglu` /
`dispatch_moe_gemv_splitk_twopass_swiglu_group_size`), so wiring it into the FP4
fc1 launcher as a new `kSplitK2` autotune candidate is a low-risk reuse.

### What was tried

- `moe_gemv.cu`: added a `kSplitK2` branch to
  `launch_moe_gemv_fp4_symmetric_interleaved_swiglu<T>` that dispatches to the
  existing generic two-pass split-K SwiGLU kernel (`SplitK=2`, fp32 cross-split
  partials, fused SwiGLU in pass 2) for both `half` and `bfloat16`. The launcher
  already auto-degrades to the fused kernel when `num_iters < 2`.
- `moe_quantization.cc`: split the FP4 autotune candidate list into
  `kFc1Candidates = {default, ctan16, threads64, splitk2}` (fc1 only) and
  `kFc2Candidates = {default, ctan16, threads64}`.
- No change to `is_moe_gemv_fp4_supported` (`CtaNForConfig(kSplitK2)=8`, and
  `n % 8 == 0` holds for every target shape).

### Result — split-K loses; autotune correctly rejects it

`test_qmoe_fp4_cuda.py` **19/19** on and off (the fp32 two-pass reduce stays
within the fp16/bf16 tolerance). Autotune timings on the gpt-oss-20b decode
shape (H200/SM90, fp16, expanded=4, fc1 n=5760/k=2880), `kIters=20`:

| fc1 config | ms / 20it |
|---|---|
| `default` | **1.326** ← chosen |
| `splitk2` | 1.435 (+8 %) |
| `threads64` | 1.550 |
| `ctan16` | 1.565 |

`splitk2` is faster than the other tile variants but **slower than `default`**,
so the autotuner picks `default` (no regression) and the change ships no benefit.

### Why — the GEMV grid is not SM-starved; the limiter is intra-SM

Split-K helps only when the base grid under-fills the GPU. For FP4
(`kInterleave=1`, `CtaN=8`) the fc1 grid is
`expanded_rows × n/CtaN × SplitK`. Even at batch-1 decode the base grid
(`expanded × n/8`) is already **far larger than any target GPU's SM count**:

| Shape | base CTAs (`expanded × n/8`) | SMs (H200 / 4090 / 3090 / 5080) |
|---|---|---|
| gpt-oss-20b fc1 (exp=4, n=5760) | 2880 | 132 / 128 / 82 / 84 |
| qwen3 fc1 (exp=8, n=1024) | 1024 | … |
| gemma fc1 (exp=8, n=1408) | 1408 | … |

So the grid already saturates the SMs across all targets — split-K cannot add
useful parallelism and only adds a second reduction pass + `cudaMallocAsync`
partials. The 42 % occupancy is an **intra-SM** limit (register pressure /
exposed ALU-pipe latency from the non-interleaved `kStepK=8` K-loop, per the
SWAR/wide-load diagnosis above), which split-K does not touch.

### Conclusion — reverted; pivot to the interleaved layout

Both edits were **reverted** (no benefit on any target, and a never-selected
candidate is pure binary/compile-time cost). This confirms the earlier diagnosis:
the binding constraint is the **integer-ALU loop/index overhead at low intra-SM
occupancy**, not grid under-subscription. The real lever remains the
**`kInterleave=4` interleaved FP4 layout** (mirroring INT4's
`ColumnMajorInterleaved`, 128-bit `AccessTypeW`, `kStepK=32`): 4× fewer K-trips
cut the saturated integer-ALU overhead and amortize the cross-warp reduction 4×,
which is the structural change that should move FP4 decode toward INT4.

## 2026-06-17 — FP4 GEMV interleaved (`kInterleave=4`) `ColumnMajorInterleaved` layout (negative result, reverted)

This is the structural change predicted as "the real lever" by every prior
diagnosis above: make the fused MXFP4 W4A16 GEMV mirror INT4's
`ColumnMajorInterleaved` weight layout, replacing the non-interleaved
`ColumnMajor` (`kStepK=8`) path. It was implemented end-to-end, measured, and
**reverted** — it produced **no decode speedup** and **regressed bf16 accuracy**.

### What was changed

- **`fpA_intB_gemm_preprocessors.{h,cu}`**: added `bool apply_bias_interleave =
  true` to `preprocess_weights_for_mixed_gemm_cuda`. FP4 callers pass `false` so
  the integer-only step 4 (add `+8` bias + pair-interleave) is skipped — that
  step's integer bias would corrupt the floating-point e2m1 codes. Steps 1–3
  (row-permute + subbyte-transpose + column-interleave) are layout-only and
  apply to e2m1 unchanged.
- **`moe_quantization.{h,cc}` `PrePackRepackFP4Weights`**: added an
  `interleaved_gemv_layout` flag. `false` (native WFP4AFP8) keeps the existing
  `LaunchQMoERepackFP4ColToRow` `[E,n,k/2]` layout; `true` (fused FP4 GEMV) feeds
  the raw `[E,k,n/2]` e2m1 codes per-expert through the CUTLASS fpA_intB SM80
  `W4_A16` preprocessor with `apply_bias_interleave=false`. The decode kernel
  reuses the **existing** linear `Fp4I2FConverter` + `ColumnMajorInterleaved`
  `Mapper` unchanged (steps-1–3 logical order == linear-converter output order).
- **`moe_gemv.cu`**: `Fp4KernelDetails` switched `ColumnMajor` →
  `ColumnMajorInterleaved` (with `TileSizeK=64`, `kElemBits=4` ⇒ `kInterleave=4`,
  `kStepK=32`, `kThreadsPerInterleavedTile=2`). `is_moe_gemv_fp4_supported` now
  requires `n % (CtaN*4)==0` and `k % 64==0` (complete interleaved K-tiles).
- **Scales unchanged**: the interleaved `kStepK=32` equals the MXFP4 block size,
  so the standard interleaved group-scale iterator (`GroupSize=32`) already maps
  one scale per column per K-block. The existing `[E,k/32,n]` fused-scale layout
  (`TryBuildGemvFp4Scales`) needed no change.

### Result — no speedup (gpt-oss-20b FP4, H200, `-b 1 -l 512 -g 256`, cuda-graph off)

| Variant | decode tps |
|---|---|
| FP4 GEMV non-interleaved (`kInterleave=1`, baseline) | 188.3 |
| FP4 GEMV **interleaved (`kInterleave=4`)** | **188.8** |
| INT4 GEMV (reference target) | ~250 |
| FP4 dequant fallback (GEMV off) | ~13 |

`+0.3 %` — within run-to-run noise. The 4× reduction in K-loop trips did **not**
translate into any decode-throughput gain, and the gap to INT4 (~250 tps) was
unmoved.

### Why it didn't win

The hypothesis (loop/index integer-ALU overhead from `kStepK=8` is the binding
constraint) was **wrong, or at best not the whole story**. With the interleaved
layout each thread now holds `kStepK=32` weights + activations per step in
registers (4× the non-interleaved footprint) and the warp reduction is split
across only `32/kInterleave = 8` lanes per column-group. The reduced K-trip
count is offset by **higher per-thread register pressure** (the kernel was
already occupancy-bound at ~42 %, so any register growth pushes occupancy the
wrong way) and by the **per-element e2m1 decode cost**, which is independent of
the K-tiling and remains the same total work. Net: a wash.

### Accuracy regression (the reason it cannot ship even at break-even)

The GEMV accumulates in-register in the **activation dtype** (`tile_acc` is
`TypeA`, not fp32). Interleaving lengthens each bf16 accumulation chain (32
products/accumulator/step vs 8) and reduces the number of reducing lanes, so the
bf16 rounding error roughly **doubles**:

| `test_qmoe_fp4_cuda.py` bf16 SwiGLU case | dequant fallback | interleaved GEMV | atol |
|---|---|---|---|
| tokens=1, hidden=inter=512 | 0.0625 | **0.1563** | 0.15 |
| tokens=2, hidden=inter=512 | 0.0938 | **0.1875** | 0.15 |

Both bf16 GEMV-shaped cases exceed the 0.15 bf16 tolerance (2/19 fail); the fp16
cases (tighter 0.12 atol) still pass. This is intrinsic to the interleaved
layout under in-register bf16 accumulation — the only fix is fp32 accumulation,
which was **already tried and reverted** (register pressure → occupancy
35 % → wash, see the wide-load experiment above). So interleaving and bf16
accuracy are fundamentally in tension here.

### Conclusion — reverted

No throughput gain on the target decode shape **and** a bf16 parity regression,
with the natural mitigation (fp32 accum) independently known to erase any gain.
The interleaved layout is therefore **not** the lever it was predicted to be: the
FP4 GEMV bottleneck is **intra-SM occupancy / per-element e2m1 decode + bf16
accumulation precision**, not K-loop integer-ALU overhead. All five files were
reverted (`git revert`). Future work that wants the interleaved layout must pair
it with an accumulation strategy that is both higher-precision (for bf16 parity)
and register-frugal (to preserve occupancy) — the two have so far been mutually
exclusive in this kernel.

---

## 2026-06-28 — FP4 vs INT4 e2e re-baseline on current binary (CUDA graph ON) + prefill is the bigger gap

Fresh side-by-side benchmark of the production-shaped FP4 (MXFP4) QMoE model vs
the INT4 QMoE baseline on the **current** `cu130_fp4_bench` build, with each
model's `genai_config.json` used **as-is** (`enable_cuda_graph=1`). This updates
the older decode-only numbers above (which were all measured with CUDA graph
*off* on a previous binary), and adds a prefill measurement that reframes where
the remaining gap actually is.

### Setup

- GPU: 1× H200 (SM90). Build: `build/cu130_fp4_bench/Release`
  (`onnxruntime_USE_FP4_QMOE=ON`), which already contains the branch-free e2m1
  converter and the `{CtaN,Threads}` FP4-GEMV autotuner from the experiments above.
- Models:
  - FP4: `/tianlei/models/gpt-oss-20b/cuda_int4_rtn_mixed_fp4_qmoe`
    (MXFP4 QMoE experts, block 32).
  - INT4 baseline: `cuda_int4_int4_qmoe_rtn_mixed_lmh4_qknorm_qmoe32`
    (same rtn-mixed recipe, INT4 QMoE block 32).
- Driver: `~/dev/scripts/h200_18/bench_gpt_oss_fp4_vs_int4.sh`
  (`benchmark_e2e.py`, batch 1, prompt 512, gen 128, reps 5, warmup 2).
- Decode kernel attribution: `profile_fp4_vs_int4_decode.sh` +
  `ORT_FORCE_DETERMINISTIC_MOE=1` (CUDA graph off, prompt 8, gen 128).

### End-to-end throughput (batch 1, prompt 512, gen 128, **CUDA graph ON**)

| Config | Prefill (tps) | Decode (tps) | Decode (ms/tok) | vs INT4 decode |
|--------|---------------|--------------|-----------------|----------------|
| INT4 baseline | 25300 | **427.2** | 2.34 | 1.00× |
| FP4, `ORT_ENABLE_FP4_GEMV=1` (fused GEMV) | 3768 | **270.0** | 3.70 | 0.63× (1.58× slower) |
| FP4, GEMV off (top-k dequant fallback) | 3742 | 15.0 | 66.77 | 0.035× (28.5× slower) |

Takeaways that differ from the older (graph-off) record:

1. **The fused GEMV path is now ~18× the fallback end-to-end** (270 vs 15 tps),
   not the "~1 %" of the very first Phase-5 measurement. The Phase-0 fallback
   regressed badly on this model/binary (15 tps); `ORT_ENABLE_FP4_GEMV=1` is
   effectively mandatory for usable FP4 decode. **It is still opt-in (env-gated)
   and should be made the SM<120 default for this model.**
2. **CUDA graph helps INT4 more than FP4** (INT4 ~250→427 = 1.7×; FP4 ~188→270 =
   1.4×). FP4 decode spends a larger fraction of each token *inside* the GEMV
   kernels, so removing launch overhead buys proportionally less — i.e. the
   residual decode gap is GPU-kernel-bound, consistent with the ncu diagnoses above.
3. **Prefill is now the larger relative gap (6.7×), and it is untouched.** All
   prior FP4 work optimized *decode*; prefill was never addressed.

### Decode kernel attribution (deterministic, per call)

| MoE kernel (per layer / token) | FP4 (GEMV on) | INT4 | FP4/INT4 |
|--------------------------------|---------------|------|----------|
| fc1 SwiGLU `moe_gemv_interleaved_swiglu_kernel` | 49.6 µs | 15.3 µs | 3.2× |
| fc2 `moe_gemv_kernel` | 28.8 µs | 11.5 µs | 2.5× |
| **MoE total / layer** | **78.4 µs** | **26.8 µs** | **2.9×** |

The FP4 fc1 is down from the 63.5 µs baseline recorded above (the converter +
autotune fixes are in this binary), but is still ~2.9× the INT4 MoE cost. This
2.9× per-layer MoE deficit is the entire source of the 1.58× decode gap (the rest
of the token — attention, norm, RoPE, router — is shared and identical).

### Why FP4 prefill is 6.7× slower (the new finding)

Prefill (`num_rows = 512 > 256`) does **not** engage the fused GEMV (decode-only
gate), so it falls through to the dequant path. The deterministic trace shows the
prefill-only kernels that never appear in INT4:

- `QMoEDequantizeFp4WeightsKernel<half>` — ~2.0 ms/call, 48 calls per prefill
  token-block (24 layers × {fc1,fc2}).
- dense A16 `cutlass GemmUniversal GroupProblemShape` — the un-fused grouped GEMM
  consuming the materialized BF16 weights.

INT4 prefill instead uses the **fused** `MoeFCGemm … DqMmaMultistage` grouped GEMM
(dequant fused in the GEMM mainloop, top-k experts only), which is why INT4
prefill hits 25300 tps vs FP4's 3768.

### How to close the gap

Two distinct levers, in priority order:

1. **Prefill (6.7×, biggest and untouched lever).** Route FP4 prefill through a
   **fused-dequant grouped GEMM** like INT4's `DqMmaMultistage` (dequantize e2m1
   in the GEMM mainloop, top-k experts only) instead of the all-expert
   dequant-to-BF16 + dense A16 GEMM. Note the existing `QMoEBuildActiveExpertMask`
   top-k skip in the dequant fallback does **not** help prefill: at prompt 512 ×
   top-k 4 essentially **all 32 experts are active**, so the mask elides nothing
   and the full dense-weight materialization + HBM round-trip happens anyway. The
   structural fix is fusing dequant into the grouped-GEMM mainloop (as INT4 does).
   This is the highest-value remaining FP4 work and is orthogonal to all the
   (exhausted) decode-GEMV micro-opts.

2. **Decode (1.58×, largely diminishing returns).** Per the reverted experiments
   above, the GEMV is **intra-SM occupancy + per-element e2m1 decode + bf16-accum
   precision** bound — interleaved layout, wide loads and split-K were all tried
   and reverted (no gain and/or bf16 parity regression; the natural fix, fp32
   accum, costs occupancy). The only remaining viable decode lever is a
   **register-frugal higher-precision accumulation** (periodic fp32 flush of a
   bf16 running sum) that would *unlock* the wide-load / interleaved K-loop
   reduction without the occupancy penalty — i.e. break the
   accuracy-vs-occupancy tension that blocked every prior attempt. This is a
   harder kernel change for a capped ~1.58× ceiling, so it should come **after**
   the prefill fix.

3. **Ship the fast path by default. — LANDED 2026-06-28.** `ORT_ENABLE_FP4_GEMV`
   was opt-in; without it FP4 decode was 15 tps (28.5× slower). The fused GEMV is
   now the **default** on the SM<120 fallback regime (`moe_quantization.cc`:
   `enable_fp4_gemv_` defaults true unless `ORT_ENABLE_FP4_GEMV=0`). Prefill and
   any unsupported shape still fall through to the dequant path at dispatch time,
   so this is a pure decode win with no prefill regression.

   Validation: `test_qmoe_fp4_cuda.py` **19/19** with the new default *and* with
   the explicit opt-out `ORT_ENABLE_FP4_GEMV=0` (dequant fallback). End-to-end on
   gpt-oss-20b FP4 with **no env var set** (batch 1, prompt 512, gen 64, cg=1):
   decode **263 tps** — up from 15 tps previously, a ~17.6× out-of-the-box
   improvement, closing the decode gap to INT4 to the residual 1.58× kernel gap.
   The bench/profile scripts' `*-off` configs were updated to set
   `ORT_ENABLE_FP4_GEMV=0` explicitly (an unset env now means GEMV-on).

### Artifacts

- `~/dev/scripts/h200_18/bench_gpt_oss_fp4_vs_int4.sh` (new; INT4 / FP4-off /
  FP4-on, config used as-is, optional `--cuda-graph` override).
- `~/onnxruntime/profile_fp4_vs_int4_decode.sh`,
  `~/onnxruntime/profile_fp4_gemv_onoff.sh` (updated to the `cu130_fp4_bench`
  binary + current FP4/INT4 model paths; honor `GPU=` for device pinning).
- CSVs under `~/bench_results/gpt_oss_fp4_vs_int4_*`; nsys reports under
  `/tmp/qmoe_fp4_profile/` and `/tmp/qmoe_fp4_det/`.

> Bench/profile concurrency caveat: the FP4 bench (`--cuda-graph`) and the profile
> scripts both edit the **same** model `genai_config.json` (cuda-graph toggle).
> Never run them concurrently against the same model directory — a race made an
> early FP4-on run read `enable_cuda_graph=0`. Run sequentially, or point them at
> separate model copies.

---

## 2026-06-28 — Closing the residual 1.58× decode gap: vLLM (MXFP4) comparison + plan

After the default-on win, FP4 decode is **263 tps** vs INT4 **354–427 tps** and
vLLM MXFP4 **~350 tps**. Why is INT4/vLLM faster, and what is the lever?

### What vLLM does (gpt-oss MXFP4, H200/SM90)

vLLM picks the **Triton `matmul_ogs`** backend on SM90 (FlashInfer-TRTLLM is
SM100+; Marlin is lower priority). That kernel is a **tensor-core grouped GEMM**
(M = tokens × top_k) that dequantizes e2m1 + e8m0 **in the mainloop** with a
Hopper value/scale swizzle, fused SwiGLU epilogue, one kernel for prefill and
decode. So vLLM avoids both ORT problems at once: no dense dequant round-trip
(prefill) and tensor-core MMA at low M (decode).

### Why ORT INT4 already beats vLLM but FP4 does not

ORT INT4 decode is **not** tensor-core — it is the CUDA-core `moe_gemv` with the
`ColumnMajorInterleaved` layout (`kStepK = 128/4 = 32`, 128-bit weight loads) and
fp16 accumulation, and it hits 354–427 tps. FP4 uses a **separate** `moe_gemv_fp4`
with non-interleaved `ColumnMajor` (`kStepK = 8`, 32-bit loads) → 4× more K-trips
and worse weight coalescing, hence ~2.9× slower per MoE layer (1.58× e2e). The
structural reasons FP4 has not matched INT4's interleaving are recorded above
(interleaved + bf16 regression; wide-load + fp32-accum occupancy wash).

### Two candidate levers

1. **Match INT4's interleaved GEMV for FP4 (in-family).** INT4 already supports a
   clean per-shape **fp32-accum** instantiation (`TypeTag<float>` in `moe_gemv.cu`;
   `bf16 always fp32`). The reverted FP4 interleave failed because it kept bf16
   accum; pairing interleave with INT4's fp32-accum *and* a smaller `CtaN`/
   register budget to hold occupancy is the untried combination. Ceiling ≈ INT4
   GEMV (~400 tps) — closes decode but **not** prefill.
2. **Tensor-core grouped GEMM with in-mainloop e2m1 dequant (vLLM-equivalent).**
   Route FP4 through the CUTLASS `fpA_intB` grouped GEMM that INT4 prefill uses,
   adding an e2m1+e8m0 converter. Closes **both** decode and prefill, matches
   vLLM, but is a large kernel effort. Highest payoff, highest risk.

Recommended order: try (1) first (bounded, reuses INT4 fp32-accum), keep (2) as
the structural follow-up that also kills the 6.7× prefill gap.
