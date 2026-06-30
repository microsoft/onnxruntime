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

---

## 2026-06-28 — FP4 GEMV converter: 2-nibbles-per-32-bit-store packing (re-run 2026-06-30: performance-neutral)

Restart of "FlashInfer-style decode" experiments. Small contained converter
change while warming back up to the kernel; deliberately low-risk.

> **Update 2026-06-30:** Re-run on a fully idle H200 (all 8 GPUs at 0 % SM).
> Original 2026-06-28 numbers were contention noise. Conclusion below is now
> **resolved**: the change is performance-neutral, confirming the store count
> is not the GEMV bottleneck.

### Change

- Commit: `0fb02a5fb7` — *EXP: FP4 GEMV converter packs 2 nibbles into one 32-bit
  store (bit-identical)*. File: `onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemv/details.h`.
- `Fp4I2FConverter::convert<N>` now decodes both nibbles of each byte via a
  `decode_bits()` helper, ORs them into one `uint32_t`, and emits a single aligned
  store instead of two scalar `half`/`bf16` writes. `decode()` is preserved.
  Bit-identical to two scalar decodes.

### Reproduce

```bash
# Build (C++/CUDA only)
cd /home/tianlei/onnxruntime/build/cu130_fp4_bench/Release && ninja onnxruntime_providers_cuda
cd /home/tianlei/onnxruntime
cp build/cu130_fp4_bench/Release/libonnxruntime_providers_cuda.so \
   .venv_cu130/lib/python*/site-packages/onnxruntime/capi/
cp build/cu130_fp4_bench/Release/libonnxruntime_providers_cuda.so \
   /home/tianlei/ort_home_cu130_fp4_bench/lib/

# Correctness (20 FP4 tests)
cd onnxruntime/test/python/transformers
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/tianlei/ort_home_cu130_fp4_bench/lib:/home/tianlei/onnxruntime/build/cu130_fp4_bench/Release:/home/tianlei/cuda13.0/lib64:$LD_LIBRARY_PATH
/home/tianlei/onnxruntime/.venv_cu130/bin/python -m pytest test_qmoe_fp4_cuda.py -k fp4 -q

# Decode microbench (small fast shape; gpt-oss 32E build is slow)
export ORT_FORCE_DETERMINISTIC_MOE=1
for r in 1 2 3; do /home/tianlei/onnxruntime/.venv_cu130/bin/python \
  bench_fp4_gemv_autotune.py --hidden 512 --inter 512 --experts 8 --top_k 4 \
  --tokens 1 --iters 200 --reps 4 2>&1 | tail -1; done
```

### Result — re-run 2026-06-30 on idle GPU (resolved: neutral)

The 2026-06-28 run was abandoned because the farm was contended (406/510/422 µs,
557 s test wall-clock). Re-ran on 2026-06-30 with all 8 H200s idle (0 % SM), doing
a proper A/B: rebuilt the **baseline** (pre-change scalar converter, parent of
`0fb02a5fb7`) and the **experiment** (2-nibble store) from the same tree and
benchmarked both back-to-back on GPU 0.

- Correctness: **20/20** `test_qmoe_fp4_cuda.py` pass in 44 s (vs 557 s contended).
- Decode bench (512/512/E8, tok=1). H200 clocks are unlocked on this farm
  (idle 345 MHz → boost 1980 MHz; `nvidia-smi -lgc` denied — no permission), so
  per-launch `BEST` is bimodal between a fully-boosted state (~56–58 µs) and a
  partially-boosted state (~100–119 µs). The boosted **best-case minimum** across
  many launches is the only clock-stable metric:

  | Build | Best-case min (µs) | Typical band (µs) |
  |-------|--------------------|-------------------|
  | Baseline (scalar store) | **55.87** | 100–119 |
  | Experiment (2-nibble store) | **57.58** | 98–117 |

  The 55.87 vs 57.58 µs gap is ~3 %, inside clock/measurement jitter, and a
  bit-identical store-packing change cannot produce a real 2× swing — the
  bimodality is purely the GPU boost state. **Verdict: performance-neutral.**

### Conclusion

- The 2-nibbles-per-32-bit-store packing is **bit-identical and performance-neutral**
  on SM90. It neither helps nor hurts, which **confirms the GEMV is not store-bound**
  — consistent with the 2026-06-15 SWAR/`decode2` negative result and its ncu finding
  that the kernel is **integer-ALU / loop-index bound**. Store count is not the lever.
- Because the change is correct, simpler than the reverted `decode2`, and at worst
  neutral, it is kept as a tidy refactor rather than reverted.

### Next step

- Stop pursuing converter/store-side micro-optimizations for the decode GEMV; they
  are dead ends while the kernel is ALU/loop-index bound.
- Pivot to the two structural levers recorded above, in order:
  1. **Interleaved layout + fp32 accumulation** (lever 1) — bounded, reuses the
     INT4 path; attack the integer-ALU/loop-index bound directly.
  2. **Tensor-core grouped GEMM** (lever 2) — the structural fix that also closes
     the 6.7× prefill gap.
- If any further decode-converter idea is tried, profile with ncu (not wall-clock)
  and lock GPU clocks first; unlocked-clock wall-clock A/B on this farm cannot
  resolve deltas below ~10 %.

## 2026-06-30 — FP4 fc1 split-K + **fp32-accumulation** SwiGLU GEMV, opt-in (FINISHED 2026-07-02 — confirmed slower on both H200 (~2.6 % fc1 / ~1.2 % e2e) and A100 (~4.5 %); second split-K negative)

Revisit of the 2026-06-16 `kSplitK2` negative result, motivated by a new INT4
data point: in the **current** tree INT4's two-pass split-K SwiGLU GEMV (`SplitK=2`,
`AccT=float`) is *faster* than its non-split-K path with fp32 accumulation
(`moe_gemv.cu` gates it behind `ORT_MOE_GEMV_SPLITK2_SWIGLU`). The split-K kernels
were refactored since 2026-06-16: `moe_gemv_splitk_partials_kernel` now defaults
`AccT=float`, so **each split accumulates its (shorter) K-chain in fp32 per-thread**
and the cross-split reduce sums fp32 partials — not just the bf16-accum + fp32-reduce
the original `kSplitK2` used. The question: does borrowing INT4's *current* split-K +
fp32-accum recipe finally help FP4?

### What was tried (kept, opt-in — not reverted)

All behind a new env gate `ORT_FP4_GEMV_SPLITK=1`; default path is byte-for-byte
unchanged (off by default).

- `moe_gemv_fp4.h`: `Fp4MoeGemvUseSplitK()` accessor, `kFp4MoeGemvSplitK = 2`, and a
  `void* splitk_partials` parameter on `launch_moe_gemv_fp4_symmetric_interleaved_swiglu`.
- `moe_gemv_fp4.cu`: when `splitk_partials != nullptr && Fp4MoeGemvUseSplitK()`, the
  fc1 launcher dispatches the **existing generic** two-pass kernel
  `dispatch_moe_gemv_splitk_twopass_swiglu_group_size<Fp4KernelDetails<T>, CtaN,
  Threads, 2, T, float, float>` (`AccT=float`, `PartialT=float`). The shared
  `dequantize<Details,…>` routes to `Fp4I2FConverter` automatically (same `Details`
  as the single-pass FP4 kernel), so no new device code is needed. Auto-degrades to
  single-pass when `num_iters < 2`.
- `moe_quantization.cc`: allocates the fp32 partials scratch
  (`kFp4MoeGemvSplitK · expanded · 2·inter_size` floats) only when the env is set, and
  plumbs it to the fc1 launch. fc2 (non-SwiGLU) is unchanged.

Only fc1 (the dominant ~49.6 µs SwiGLU GEMV) is wired; fc2 keeps single-pass.

### Engagement caveat (important for benchmarking)

FP4 `ColumnMajor` has `kStepK = 128/16 = 8`, so `CtaK = StepK·Threads = 1024` and
`num_iters = ceil(k / 1024)`. Split-K only engages when `num_iters ≥ 2`, i.e.
**k ≥ 2048**. The small microbench shape (`--hidden 512`) degrades to single-pass and
cannot measure this — the A/B **must** use realistic dims (gpt-oss-20b
`hidden=inter=2880` → `num_iters=3`).

### Result — correctness PASS; performance ~2.6 % slower (idle GPU, resolved)

- Correctness: **20/20** `test_qmoe_fp4_cuda.py` pass with `ORT_FP4_GEMV_SPLITK=1`
  (fp32 per-split + fp32 cross-split reduce holds the bf16/fp16 tolerance).
- Clean A/B on a **fully idle** H200 (all 8 GPUs at 0 % SM), clocks locked to
  1980 MHz (`sudo nvidia-smi -i 0 -lgc 1980`; reset `-rgc` after), no-dump-node
  build (`git-commit-id=e8df8f4dd6`, `dump-node` compiled out — the earlier
  `dump-node=1` build added per-node I/O dumps that inflated and de-stabilized the
  measurement). `bench_fp4_gemv_autotune.py --hidden 2880 --inter 2880 --experts 8
  --top_k 4 --tokens 1 --warmup 30 --iters 300 --reps 8`, best-of-reps:

  | Config | Whole MoE-layer BEST (µs) |
  |--------|---------------------------|
  | Baseline (single-pass) | **119.88** |
  | Split-K + fp32-accum (`SPLITK=1`) | **121.30** (+1.2 %) |

- **Per-kernel attribution** (nsys `cuda_gpu_kern_sum`, median over 254 fc1 calls)
  isolates the change to the fc1 GEMV itself — the cleanest signal:

  | fc1 component | Baseline | Split-K |
  |---------------|----------|---------|
  | `moe_gemv_interleaved_swiglu_kernel` (single-pass) | 53,439 ns | — |
  | `moe_gemv_splitk_partials_kernel` (two-pass, fp32) | — | 52,927 ns |
  | `moe_gemv_splitk_reduce_swiglu_kernel` | — | +1,920 ns |
  | **fc1 total** | **53,439 ns** | **54,847 ns (+2.6 %)** |

  The split-K **partials** kernel is marginally *faster* than the baseline
  single-pass (52,927 vs 53,439 ns — the shorter, fp32-accumulated per-split
  K-chains help slightly and tighten the variance), **but** the mandatory reduce
  pass (1,920 ns) more than erases that gain. Net fc1 is +2.6 % slower per call,
  matching the +1.2 % whole-layer e2e delta (diluted by the rest of the op).

### Why it's slower — confirmed grid-occupancy mechanism

The mechanism is exactly as hypothesized from the 2026-06-16 `kSplitK2` result:
FP4's non-interleaved `ColumnMajor` (`kInterleave=1`, `CtaN=8`) already launches
`expanded × n/8` base CTAs — **4× more y-blocks than INT4's interleaved
`n/(CtaN·4)` grid** — so the SMs are already saturated. Splitting K therefore only
adds the second reduce pass + partials round-trip without filling idle SMs. INT4
benefits from the same recipe because its interleaved grid is 4× coarser and *does*
leave SMs to fill. The fp32-per-split refinement that helped INT4 does **not**
transfer to FP4 because the two paths differ in **grid occupancy, not accumulation
precision** — the partials kernel even ran slightly faster here, so it is purely the
reduce-pass overhead on an already-full grid.

### Status / verdict

- **Resolved negative — the second split-K negative for FP4** (after 2026-06-16
  `kSplitK2`). Split-K + fp32-accum is correct but ~2.6 % slower on the fc1 GEMV
  and ~1.2 % slower e2e on idle, locked-clock H200/SM90.
- The code is **kept dormant**, opt-in behind `ORT_FP4_GEMV_SPLITK` (default off,
  default path byte-for-byte unchanged, correctness-validated 20/20) so the A/B is
  reproducible and so the same plumbing can be re-tested cheaply on a *different*
  architecture (e.g. A100/SM80, which has fewer SMs and so an even-more-saturated
  FP4 grid — expected to confirm the same negative, but worth a data point) without
  rebuilding. No revert needed; it adds no cost when off.
- The FP4 decode lever is therefore confirmed to be **not** grid parallelism but the
  **interleaved layout (lever 1)** / **tensor-core grouped GEMM (lever 2)** recorded
  above.

### Benchmark-harness note (why setup, not measurement, was the slow part)

The earlier "contention-blocked" run also suffered from a pathological *setup*
cost unrelated to the kernel: the MXFP4 reference quantizer
(`test_qmoe_fp4_cuda.py::quantize_weight_to_mxfp4`) computed the ue8m0 block scales
in a **pure-Python double loop calling `.item()` per element** — ~25 M serialized
GPU↔CPU syncs at the gpt-oss-20b 2880/32-expert shape (a reference quantizer
written for tiny 128–256 unit-test shapes, reused unchanged at full size), which
alone pushed one build past 7 minutes. It was **vectorized** to pure tensor ops
(round-half-to-even `log2`, clamp `[1,254]`, default `1.0`/code `127` for all-zero
blocks), verified **bit-identical** to the loop (0 mismatches over a 2880×90 tensor
including zero blocks). The *measurement* loop was always fine; only the per-process
setup was slow. After the fix, a full A/B (both arms, best-of-8) runs in well under
a minute.

### Result — re-run 2026-07-02 on idle A100 (resolved: split-K confirmed ~4.5 % SLOWER)

The clean A/B was re-run on a **fully idle A100-SXM4-80GB** (sm_80, 108 SMs, all 8
GPUs at 0 % SM), GPU 0 persistence-mode on and graphics clock locked to its 1410 MHz
max (`sudo nvidia-smi -i 0 -lgc 1410`; reset with `-rgc` after). Same realistic shape
(`hidden=inter=2880`, `E=32`, `top_k=4`, 1 token, fp16 → fc1 `n=5760 k=2880`,
`num_iters=3`, so split-K genuinely engages). Best-case min µs/run over 8 internal
reps × 3 process launches per arm:

| Config | launch 1 | launch 2 | launch 3 | Best-case min (µs) |
|--------|---------:|---------:|---------:|-------------------:|
| Baseline (single-pass) | 171.87 | 172.40 | 171.63 | **171.63** |
| Split-K + fp32-accum (`SPLITK=1`) | 179.27 | 179.30 | 179.66 | **179.27** |

**Verdict: split-K is ~4.5 % slower** (179.27 / 171.63 = 1.045), with **zero overlap**
between the two arms across all six launches — a clean, reproducible negative. This
confirms (and is directionally stronger than) the contended H200 preliminary read of
~1.5–3 % slower, and matches the 2026-06-16 `kSplitK2` finding.

The mechanism argument holds and is in fact **reinforced on A100**: FP4's
non-interleaved `ColumnMajor` (`kInterleave=1`, `CtaN=8`) launches `expanded × n/8`
= `4 × 720 = 2880` base CTAs for fc1 — far above A100's 108 SMs — so the SMs are
already saturated and split-K only adds the cross-split reduce pass + fp32-partials
round-trip without filling idle SMs. With **fewer** SMs than the H200 (108 vs 132),
the device is even more saturated, so the relative overhead of the extra pass is
larger here (~4.5 % vs the H200's predicted 1.5–3 %). The fp32-per-split refinement
that helped INT4 does not transfer to FP4, because the two paths differ in **grid
occupancy**, not accumulation precision (INT4's interleaved grid is 4× coarser and
*does* leave SMs to fill; FP4's does not).

### Status — FINISHED

- **Done.** This is the **second split-K negative** for FP4. The FP4 decode lever is
  **not** grid parallelism but the **interleaved layout (lever 1)** / **tensor-core
  grouped GEMM (lever 2)** recorded above.
- The split-K code stays **committed but dormant** behind `ORT_FP4_GEMV_SPLITK=1`
  (default off, single-pass path byte-for-byte unchanged, correctness 5/5 on sm_80 +
  20/20 on session-capable GPUs). Keeping it opt-in preserves the reproducible
  experiment without any cost to the shipping path; a follow-up may revert it once the
  negative is considered permanently settled.

#### Environment / harness notes (A100 re-run)

The re-run was on a newer-onnx box than the original H200 capture; two
environment-compatibility fixes to the test tree were needed to make the harness
runnable (behavior-preserving; correctness re-validated):

- `quantize_weight_to_mxfp4` (`test_qmoe_fp4_cuda.py`) used a per-element Python
  `.item()` double loop for the ue8m0 block-scale codes (~24.9 M serialized GPU↔CPU
  syncs at this shape) → `build_session` took **> 7 min and never finished**. Replaced
  with a vectorized torch computation (`torch.round(log2(...))` + `clamp(1,254)`),
  dropping per-session setup to ~8 s. Correctness unchanged.
- `create_fp4_moe_onnx_graph` stamps the onnx-package default opset (27 for
  onnx ≥ 1.22), which ORT rejects (`ai.onnx` officially supported through opset 26).
  The MoE op is a `com.microsoft` contrib op, so `bench_fp4_gemv_autotune.py`'s
  `build_session` now clamps the `ai.onnx` opset to 26 before session creation.

---

## 2026-06-30 — FP4 GEMV interleaved (“Lever A”) + **dtype-conditional accumulation**, opt-in (fp16 decode win ~4 %; bf16 accuracy-safe)

This revisits the **2026-06-17 interleaved-layout negative** above, which closed
with the open question:

> Future work that wants the interleaved layout must pair it with an accumulation
> strategy that is both higher-precision (for bf16 parity) **and** register-frugal
> (to preserve occupancy) — the two have so far been mutually exclusive in this
> kernel.

“Lever A” is the attempt to satisfy that constraint. The key realization is that
the precision-vs-occupancy tension **does not have to be resolved by one
accumulator** — it can be split **per activation dtype**, because fp16 and bf16
have different mantissa budgets. The result: **fp16 gets a real ~4 % decode win**
from the interleaved layout, and **bf16 stays accurate** at a small (~1.6 %) cost.
The path is kept **committed but opt-in** behind `ORT_FP4_GEMV_INTERLEAVED=1`
(default off; shipping `ColumnMajor` path byte-for-byte unchanged).

### What was implemented (kept, opt-in — not reverted)

The three "levers" the prior single attempts kept separate, now combined:

1. **Interleaved layout (lever 1).** `Fp4KernelDetailsInterleaved<T>` =
   `KernelDetails<…, ColumnMajorInterleaved, false, TileSizeK=64>` ⇒
   `kInterleave=4`, `kStepK=32`, `kThreadsPerInterleavedTile=2` (4× fewer K-trips
   than the `kStepK=8` `ColumnMajor` baseline). The linear `Fp4I2FConverter` is
   reused (`UseInterleavedConverter=false`); the preprocessor's layout-only
   steps 1–3 produce exactly the nibble order the linear converter expects.
2. **dtype-conditional accumulation (lever 2 — the new idea).**
   `Fp4LeverAAccT<T> = std::conditional_t<std::is_same_v<T, half>, half, float>`:
   - **fp16 → fp16 accum.** fp16's 10-bit mantissa tolerates 16-bit accumulation
     over the longer `kStepK=32` chains; this keeps registers low (~79 reg).
   - **bf16 → fp32 accum.** bf16's 7-bit mantissa does **not** — 16-bit accum
     fails the bf16 tolerance — so bf16 accumulates in fp32 (~96 reg).
3. **Smaller `CtaN` (lever 3).** `kInterleavedCtaN=4` (vs default 8), pinned with
   `kInterleavedThreads=128` (config ignored so weights and kernel always agree).

A diagnostic override `ORT_FP4_GEMV_INTERLEAVED_HALFACC=1` forces **16-bit accum
for both dtypes** (used to isolate the layout-vs-accum effect; it regresses bf16).

Files (all behind the gate; native WFP4AFP8 paths unchanged):

- **`moe_gemv_fp4.{h,cu}`**: `Fp4MoeGemvUseInterleaved()` (env gate),
  `Fp4MoeGemvInterleavedHalfAccum()` (diagnostic), `Fp4KernelDetailsInterleaved<T>`,
  `Fp4LeverAAccT<T>`, interleaved branch in both `launch_moe_gemv_fp4_symmetric`
  (fc2) and `launch_moe_gemv_fp4_symmetric_interleaved_swiglu` (fc1, before split-K),
  and the interleaved shape gate (`n % (CtaN*4)==0`, `k % 64==0`).
- **`fpA_intB_gemm_preprocessors.{h,_impl.cu}`**: `apply_bias_interleave=true`
  param; FP4 passes `false` to skip integer-only step 4 (the `+8` bias +
  pair-interleave would corrupt e2m1 float codes); layout-only steps 1–3 apply
  to e2m1 unchanged.
- **`moe_quantization.{h,cc}`**: `PrePackRepackFP4Weights(..., gemv_interleaved)`;
  the interleaved branch routes raw `[E,k,n/2]` e2m1 per-expert through the CUTLASS
  fpA_intB SM80 `W4_A16` preprocessor (`apply_bias_interleave=false`,
  `packing_sm=80`). Scales unchanged (`kStepK=32` == MXFP4 block size).

### Result — correctness PASS (4/4), fp16 decode win, bf16 accuracy-safe

Hardware: **A100-SXM4-80GB (sm_80)**, graphics clock locked to 1410 MHz, idle GPU,
CUDA 13.0. Correctness shape `hidden=inter=512`, swiglu, top_k=4; microbench shape
`hidden=inter=2880`, `E=32`, `top_k=4`, `tokens=1` (`bench_fp4_gemv_autotune.py`).

**Correctness (vs torch reference, all 4 cases PASS):**

| `test_fp4_decode_swiglu_gemv` case | baseline (fp32 accum) | Lever A | atol |
|---|---|---|---|
| FP16 tokens=1 | 0.0078 | 0.0156 | 0.12 |
| FP16 tokens=2 | 0.0078 | 0.0234 | 0.12 |
| BF16 tokens=1 | 0.0625 | **0.0625** | 0.15 |
| BF16 tokens=2 | 0.0625 | **0.0625** | 0.15 |

fp16 uses the cheaper 16-bit accum (slightly looser but well within the 0.12
atol); bf16 keeps fp32 accum and is bit-for-bit the baseline quality — the
2026-06-17 bf16 regression (0.1563 / 0.1875) is **resolved**.

**Decode kernel (ncu, fc1 fused-SwiGLU GEMV, `moe_gemv_interleaved_swiglu_kernel`):**

| Config | reg/thread | occupancy | fc1 GEMV duration |
|---|---|---|---|
| baseline `ColumnMajor` (fp32 accum) | 56 | 48.9 % | ~101 µs |
| **Lever A fp16** (fp16 accum) | **79** | **31.7 %** | **~96.8 µs** |
| Lever A bf16 (fp32 accum) | 96 | 27.9 % | ~107.8 µs |

**End-to-end (`BEST` of 8 reps × 400 iters, µs/run):**

| dtype | baseline | Lever A | Δ |
|---|---|---|---|
| **fp16** | 172.1 | **165.3** | **−4.0 % (faster)** |
| bf16 | 173.8 | 176.6 | +1.6 % (slower) |

### Why fp16 wins and bf16 does not

The 2026-06-17 diagnosis was correct that this GEMV is **occupancy/register-bound,
not K-loop-bound** — the 4× K-trip reduction alone is a wash. The lever that
actually moves occupancy is the **accumulator width**, not the layout, `CtaN`, or
K-tiling:

- A `CtaN` probe (4 vs 2) gave **identical** 96 reg / 28 % occ — `CtaN` is *not*
  the register lever.
- fp32 accum costs **+17 registers** (79 → 96), which drops occupancy 31.7 % →
  27.9 % and **exactly cancels** the interleaved layout's K-trip savings (Lever A
  bf16 is ~neutral-to-slightly-slower vs baseline).
- fp16 accum stays at 79 reg / 31.7 % occ, so fp16 **keeps** the interleaved
  layout's K-trip savings → a genuine ~4 % e2e win.

So the precision-vs-occupancy tension is real and per-dtype: fp16 can afford the
cheap accumulator and pockets the layout win; bf16 cannot, so it pays the fp32
register cost and the layout win is erased. There is no single accumulator that is
both high-precision and register-frugal here — the dtype split is the resolution.

### Reproduce

```bash
# Build (C++/CUDA only, sm_80, FP4 QMoE on)
cd ~/git/onnxruntime && ./build.sh --config Release --build_dir build/cu130_fp4_bench \
  --build_wheel --parallel --nvcc_threads 2 --use_cuda \
  --cuda_home <cuda> --cudnn_home <cudnn> --compile_no_warning_as_error --skip_tests \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=80 \
  --cmake_extra_defines onnxruntime_USE_FP4_QMOE=ON \
  --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF

# Correctness (A100/sm_80; monkeypatch the SM<90 skip to exercise the decode path)
ORT_ENABLE_FP4_GEMV=1 ORT_FP4_GEMV_INTERLEAVED=1 python test_fp4_decode_swiglu_gemv ...

# Decode microbench / ncu (lock clocks first: sudo nvidia-smi -i 0 -lgc 1410)
cd onnxruntime/test/python/transformers
ORT_ENABLE_FP4_GEMV=1 ORT_FP4_GEMV_INTERLEAVED=1 python bench_fp4_gemv_autotune.py \
  --dtype fp16 --reps 8 --iters 400      # baseline: drop ORT_FP4_GEMV_INTERLEAVED
```

> **Reproducibility gotcha (cost a wrong measurement).** The ORT `--build_wheel`
> step **strips** `libonnxruntime_providers_cuda.so`, which can drop newly-added
> template-instantiated `__global__` kernels (here the fp16 16-bit-accum interleaved
> kernels). The installed wheel `.so` then silently falls back to a surviving
> instantiation, making two configs that should differ produce **bit-identical**
> ncu numbers. Verify with `nm -C <installed.so> | grep moe_gemv_interleaved_swiglu_kernel
> | grep ColumnMajorInterleaved | grep ', 4, 128,'`; if the expected `__half, __half`
> / `__nv_bfloat16, float` instantiations are missing, copy the **unstripped**
> `build/.../Release/libonnxruntime_providers_cuda.so` over the venv copy before
> profiling.

### Status / verdict

- **Kept, opt-in.** First **positive** FP4 decode result in this series: a real
  ~4 % fp16 e2e decode win, accuracy-safe for bf16. The interleaved layout *is* a
  lever — but only for fp16, and only once the accumulator is chosen per dtype.
- **bf16 gets no win** (the fp32 register cost erases it), so for bf16 the
  interleaved path is not worth enabling; the gate being opt-in lets fp16-only
  deployments take the win without affecting bf16 or the shipping default.
- **Open follow-up** (still the 2026-06-17 dream for bf16): a **register-frugal
  higher-precision accumulator** — e.g. fp32 only in the final cross-step reduce,
  or fewer live fp32 accumulators — to give bf16 the same occupancy as fp16-accum
  while keeping fp32 precision. That, plus the tensor-core grouped-GEMM lever,
  remain the paths to close the bf16 decode gap.

---

## 2026-06-30 — "Lever B": native CUTLASS WFP4A16 grouped GEMM for **prefill** (the tensor-core lever) + per-expert-tokens dispatch heuristic

Every decode-GEMV experiment above closed by pointing at the two structural
levers, and the bigger of the two — the **6.7× prefill gap** (2026-06-28 finding)
— was untouched. "Lever B" is that lever: route FP4 (MXFP4 / WFP4A16) QMoE
**prefill** through the **native CUTLASS SM90 grouped GEMM** with in-mainloop
e2m1 + e8m0 dequant (the vLLM-equivalent path), instead of the all-expert
dequant-to-BF16 + dense A16 GEMM. FP16 weights, SM90-only, double opt-in
(`ORT_ENABLE_FP4_CUTLASS_GEMM=1` + `ORT_ENABLE_FP4_CUTLASS_UNSAFE=1`).

### Part 1 — Option B: enable native CUTLASS for non-256-aligned shapes (gpt-oss 2880)

The native path's static shape gate required `hidden % 256 == 0 && inter % 256 == 0`
(`kMxfp4CutlassAlignment = 256`) because the SM90 scale-pack kernel tiles K into
groups of 8 e8m0 block-scales (`k_blocks = k/32` must be `% 8 == 0`, i.e. `k % 256`).
gpt-oss-20b's 2880 → `k_blocks = 90`, `90 % 8 = 2` → the shape fell back. Relaxing
the gate naively (256 → 32) crashed at the scale-pack `ORT_ENFORCE` (strictly worse
than falling back).

**Option B** enables 2880 by zero-padding **only the scale-factor (SF) buffer
tail**, leaving activations and weights untouched (CUTLASS TMA predicates OOB-K to
zero). Committed as `dd240d2b6a` (3 files, +58/-14):

- `moe_quantization.cc` `StaticFp4CutlassShapeSupported`: `kMxfp4CutlassAlignment`
  256 → 64 (+comment).
- `qmoe_kernels.cu` `QMoEPackFp4ScalesForTmaWsKernel`/launcher: added a
  `k_blocks_padded = ceil(k_blocks/8)*8` parameter; tail entries (`k_block >= k_blocks`)
  write 0; **removed** the `k_blocks % 8` `ORT_ENFORCE`.
- `moe_quantization.cc` `PrePackFp4ScalesForTmaWs`: allocate
  `experts * rows * k_blocks_padded` (via `SafeInt`), fully initialized by the kernel.
- `moe_kernels.cu` `computeTmaWarpSpecializedInputPointers` (WFP4A16):
  `scale_offset = expert * (gemm_n * k_blocks_padded)` (no-op for 256-aligned shapes).
- `moe_kernels.cu` runMoe alignment `ORT_ENFORCE` relaxed
  `64*8/sizeof_bits<fp4>` (128 elements) → `32*8/sizeof_bits` (64 elements) so
  `2880 % 64 == 0` passes (the WFP4A16 path reuses the `mxfp8_mxfp4.weight_block_scale`
  field; 2880 % 128 = 64 crashed).

Validation: `test_qmoe_fp4_cuda.py` 20/20; 2880 E4/M128 SiLU + E8/M256 SwiGLU
correct (native quant error == fallback, e.g. 0.125/0.25); compute-sanitizer
memcheck clean on 2880 native (no IMA from the relaxed alignment + SF tail-pad).

### Part 2 — Crossover sweep: native FP4 wins at low M, loses at high M

With 2880 enabled, a prefill latency sweep (QMoE node, io_binding, native-vs-dequant-
fallback) across the **three production model shapes** revealed native FP4 is a
large win at small per-call token counts but a **regression** at large counts:

| Model (hidden / inter / E / top_k) | 128 tok | 256 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|---|
| gpt-oss (2880 / 2880 / 32 / 4) | 4.97× | 2.74× | 1.66× | 0.96× | 0.58× | 0.37× |
| qwen3 (2048 / 512 / 256 / 8) | 6.43× | 6.42× | 4.74× | 2.79× | 1.63× | 0.96× |
| gemma4 (2816 / 704 / 128 / 8) | 5.99× | 4.27× | 2.65× | 1.53× | 0.89× | 0.54× |

(speedup = dequant-fallback latency / native-CUTLASS latency; > 1 means native faster.)

**Key insight — the curves collapse when re-indexed by per-expert tokens**
(`tokens × top_k / E`), and the crossover is **model-independent at ≈ 128
tokens/expert**:

| Per-expert tokens | ~16 | 32 | 64 | **128** | 256 |
|---|---|---|---|---|---|
| Speedup (all 3 models) | 4.3–5× | ~2.7× | 1.5–1.66× | **0.89–0.96× (crossover)** | 0.54–0.58× |

Native FP4 (fp32 accum, tensor-core in-mainloop dequant) wins decisively while each
expert's GEMM is small (decode + light prefill), and loses to the dense A16
grouped GEMM once experts are large enough to be compute-bound — exactly the regime
the dense fallback was already tuned for.

### Part 3 — dual-runner per-expert-tokens dispatch heuristic (captures the win, drops the regression)

To get the low-M win everywhere **without** the large-M regression, the op now holds
**two** runners and routes per call by average tokens/expert:

- **Second runner.** Alongside the native FP4 runner
  (`CutlassMoeFCRunner<half/bf16, __nv_fp4_e2m1, half>`), the constructor now also
  builds a dense A16 fallback runner `m_fp4_dense_fallback_runner_`
  (`CutlassMoeFCRunner<half/bf16, half, half>`) inside the `ENABLE_FP4 / USE_FP4_QMOE`
  block.
- **Routing.** `avg_tokens_per_expert = (num_rows * k_) / num_experts`;
  `route_native_fp4` is true when native is available and
  `avg_tokens_per_expert <= ORT_FP4_NATIVE_MAX_TOKENS_PER_EXPERT` (default **128** =
  the measured crossover; **0 disables** the heuristic = always-native, prior
  behavior). `active_runner` is the dense fallback when native is available but the
  threshold is exceeded, else the native runner. `active_runner` is threaded through
  profiling, `getTactics`/`profileTactics`/`setTactic`, `getWorkspaceSize`, and
  `runMoe`. `quant_params`/packed-FP4-weight selection are set only on the native
  route; the dequant block runs on the dense route.
- **Critical scale-layout fix.** When native is enabled, `packed_fp4_*_block_scales_`
  is **TMA-swizzled**, but the dequant kernel needs raw `[E, n, k_blocks]` e8m0
  scales. The dense route therefore prefers the raw copy kept by the native prepack
  (`gemv_fp4_*_block_raw_`), falling back to `packed_fp4_*_block_scales_` (which holds
  raw on the SM<90 / dequant-only build) then the input tensor. Null-safe, so SM<90
  and wfp4afp8 paths are unaffected. The raw FP4 weight initializer stays available
  because the native weight prepack sets `is_packed = false`.

Files: `moe_quantization.{h,cc}` (uncommitted, layered on top of `dd240d2b6a`).

### Validation (H200 / SM90)

- `test_qmoe_fp4_cuda.py` **20/20** (these are m=1, so all route native — confirms no
  regression on the native path).
- gpt-oss-20b prefill bench, default threshold 128 (per-expert tokens in parens):

  | Tokens | per-expert | route | speedup vs fallback | prior (always-native) |
  |---|---|---|---|---|
  | 512 | 64 | native | **1.66×** | 1.66× |
  | 2048 | 256 | dense | **1.00×** | 0.58× |
  | 4096 | 512 | dense | **1.01×** | 0.37× |

  The low-M native win is preserved and the large-M regression is **eliminated**
  (parity with the dense fallback instead of 0.58×/0.37×).
- compute-sanitizer memcheck on the large-M dense route (2880, E32, 2048 tokens):
  **0 errors** — the dual-runner dequant path (reading `gemv_fp4_*_block_raw_`) is
  memory-clean.

### Status / verdict

- **Option B (`dd240d2b6a`) ships native CUTLASS prefill for gpt-oss-20b (2880)** —
  correct + memcheck-clean — closing the 6.7× prefill gap in the low-per-expert-M
  regime for the first time. This is the **tensor-core grouped-GEMM lever (lever 2)**
  that every decode experiment above deferred to.
- **The dual-runner heuristic** (uncommitted) makes native FP4 safe to enable across
  the whole token range: it captures the multi-× win below ≈128 tokens/expert and
  falls back to dense A16 parity above it, so there is no large-batch regression.
  Tunable via `ORT_FP4_NATIVE_MAX_TOKENS_PER_EXPERT` (default 128; 0 = always native).
- **Open follow-up:** at exactly 128 tok/expert the speedup is ~0.9× (a slight loss);
  the default could be lowered to ~96 for a strictly ≥ 1.0× guarantee. The crossover
  is model-independent in this sweep, but should be re-measured if the native kernel's
  tactic selection changes.

### Artifacts

- Commit `dd240d2b6a` (Option B SF-tail-padding), branch `tlwu/20260625/qmoe_fp4`.
- Prefill bench: `~/leverB_prefill_bench.py` (QMoE-node latency via io_binding;
  `PROBE_HID`/`PROBE_INT`/`PROBE_E`/`PROBE_TOPK`/`PROBE_TOK` env vars; reports
  native-vs-fallback speedup + max_diff).
- Build: `build/cu130_fp4_bench/Release`; bench env
  `CUDA_VISIBLE_DEVICES=0 ORT_ENABLE_FP4_CUTLASS_GEMM=1 ORT_ENABLE_FP4_CUTLASS_UNSAFE=1`.

---

## 2026-06-30 — Why the native CUTLASS prefill is *still* ~50× off vLLM: the FP4 SM80 lockout (the real prefill lever)

The "Lever B" entry above compared native FP4 only against ORT's **own** dequant
fallback (a 1.66× win at 64 tok/expert) and declared the prefill gap "closed."
Re-profiling against the **hardware roofline** and against **INT4 at identical
shapes** shows that framing was misleading: *both* FP4 prefill paths run at ~2 %
of FP16 peak, so the real gap to vLLM was never closed — only the gap to ORT's
slower path was. This section pins the gap and identifies the structural fix.

### Where the time goes (clean steady-state profile)

`nsys` with a `cudaProfilerApi` capture range around the timed iterations (warmup +
per-shape tactic profiling excluded), gpt-oss-20b QMoE node, 512 tokens, native
CUTLASS (`ORT_ENABLE_FP4_CUTLASS_GEMM=1`):

| Kernel | Share of GPU time |
|---|---|
| CUTLASS grouped GEMM (FC1 + FC2) | **99.1 %** |
| doActivation / expandInputRows / finalizeMoeRouting / softmax-topk / strides / prefix-sums | < 1 % combined (~30 µs) |

The two grouped GEMMs are bimodal — one instance at ~3.29 ms (FC1, N=5760) and
one at ~1.69 ms (FC2, N=2880) per forward. **The entire MoE cost is the grouped
GEMM**; routing/permute/activation/scatter are negligible.

### Efficiency: ~2 % of peak, neither compute- nor bandwidth-bound

- FC1: 68 GFLOP / 3.29 ms = **20.7 TFLOPS = 2.1 %** of FP16 peak (989 TFLOPS).
- FC2: 34 GFLOP / 1.69 ms = **20.1 TFLOPS = 2.0 %** of peak.
- Weight-bandwidth utilisation ≈ 0.8 % (40 GB/s of 4.8 TB/s).

Both GEMMs run at a near-constant ~20 TFLOPS regardless of N/K, and the weight
read is < 1 % of HBM bandwidth, so the kernel is **latency/occupancy-bound at
M = 64 tokens/expert** (32 experts, all active at prompt 512 × top-k 4). The
per-shape tactic profiler already selects the fastest tile, so this is **not** a
tactic-selection miss — it is the SM90 TMA-WS *mixed-input* kernel itself being
the wrong tool for the small-per-expert-M regime.

### The decisive comparison: INT4 (SM80) vs FP4 (SM90) at identical shapes

Same gpt-oss-20b shape (H=2880, I=2880, E=32, top-k=4), QMoE-node latency via
io_binding, idle H200, 30 timed iters. INT4 blockwise (block 32) vs native FP4:

| tokens | tok/expert | FP4 ms / %peak | INT4 ms / %peak | INT4 speedup |
|---|---|---|---|---|
| **512** | **64** | 5.10 / **2.0 %** | 0.66 / **15.7 %** | **7.7×** |
| 1024 | 128 | 9.53 / 2.2 % | 0.99 / 20.9 % | 9.6× |
| 2048 | 256 | 10.24 / 4.0 % | 1.62 / 25.4 % | 6.3× |
| 4096 | 512 | 9.25 / 8.9 % | 2.86 / 28.8 % | 3.2× |
| 8192 | 1024 | 10.64 / 15.5 % | 5.52 / 29.9 % | 1.9× |

INT4 is 2–10× faster at the **same bit-width, same memory traffic, same shapes** —
the only difference is the **kernel**. The gap is largest exactly in the prefill
regime (64–128 tok/expert). This empirically reproduces the earlier 6.7× e2e
INT4-vs-FP4 prefill gap and localises it entirely to the grouped GEMM. (INT4
itself caps at ~30 % of peak, so it is still ~1.6× off vLLM's Triton `matmul_ogs`;
matching vLLM fully would also require an SM90-path fix, but the SM80 port closes
the bulk of the FP4 prefill gap.)

### Root cause (code-proven): FP4 is explicitly excluded from the SM80 path

`MoeGemmRunner::getConfigs()` returns **both** the SM90 TMA-WS configs and the
SM80 Ampere configs, and the profiler picks the fastest. INT4 (`uint4b_t`) gets
to choose the SM80 `DqMmaMultistage` fused-dequant grouped GEMM at low M. FP4 does
not, because two guards exclude it:

- `moe_tma_warp_specialized_traits.h` → `isValidAmpereMOESpecialisation<T, WeightType>()`
  returns `!is_same<T, e2m1> && !is_same<WeightType, e2m1>` → **false** when the
  weight is `__nv_fp4_e2m1`, so `getAmpereConfigs()` returns `{}` for FP4.
- `moe_gemm_template_dispatch.h` → the SM80 `dispatch()` body is gated `… && !isFp4`,
  so `genericMoeGemmKernelLauncher` is never instantiated for FP4.

Net effect: FP4 can **only** run the SM90 TMA-WS mixed-input kernel
(`FINEGRAINED_SCALE_ONLY`) → the ~2 % path measured above. INT4's win is purely
the SM80 kernel choice.

### The fix: port INT4's SM80 fused-dequant grouped GEMM to FP4

1. **e2m1 → half/bf16 converter.** `interleaved_numeric_conversion.h` has
   `FastInterleavedAndBiasedNumericArrayConverter` for `uint8_t` and `uint4b_t`
   only — no `float_e2m1_t`. Add `<half_t, float_e2m1_t, N>` and `<bfloat16_t,
   float_e2m1_t, N>`. e2m1 has 16 values `{0,.5,1,1.5,2,3,4,6}±`; the converter is
   *simpler* than INT4's (no zero-point bias) and can use a `lop3`/`prmt` LUT
   expansion of the nibble into the half/bf16 exponent+mantissa fields.
2. **MX block scale.** Pre-convert the e8m0 (power-of-2, group-32) block scales to
   fp16 so the existing `FINEGRAINED_SCALE_ONLY` mainloop applies them unchanged.
3. **Un-gate + instantiate.** Make `isValidAmpereMOESpecialisation` true for e2m1
   weight, drop the `!isFp4` guard in the SM80 `dispatch()`, instantiate
   `genericMoeGemmKernelLauncher<half/bf16, float_e2m1_t, …>` for SM80, and emit the
   interleaved 4-bit weight repack (`pack_weights_for_cuda_mixed_gemm` already
   supports 4-bit @ sm80).

Expected payoff: at 64 tok/expert the profiler would pick the SM80 path (as INT4
does), turning the 5.10 ms / 2.0 % FP4 prefill into roughly the INT4 0.66 ms /
15.7 % envelope — a ~7–8× MoE-node speedup, and the bulk of the vLLM prefill gap.

### Artifacts

- Profiling: `~/fp4_native_profile.py` (cudaProfilerApi capture-range, steady-state
  kernel breakdown), `~/fp4_token_sweep.py` (FP4 native+fallback TFLOPS vs
  tokens/expert), `~/int4_token_sweep.py` (INT4 SM80 TFLOPS vs tokens/expert). All
  run from `CWD=/tmp` against the venv binary; `CUDA_VISIBLE_DEVICES=3`,
  `LD_LIBRARY_PATH=~/ort_home_cu130_fp4_bench/lib:~/cuda13.0/lib64`.
- Code sites: `moe_tma_warp_specialized_traits.h` (`isValidAmpereMOESpecialisation`),
  `moe_gemm_template_dispatch.h` (SM80 `dispatch()` `!isFp4` guard, `getAmpereConfigs`),
  `cutlass_extensions/interleaved_numeric_conversion.h` (converter specializations).

## Phase 3 complete: SM80 FP4 grouped GEMM lands and is correct (opt-in)

The SM80 FP4 fused-dequant grouped GEMM is implemented end-to-end and gated behind
`ORT_FP4_SM80_GEMM=1` (fp16 activation only this phase; bf16 still compiles). With the
flag **off** behavior is byte-identical to before (the SM90 TMA / dense-fallback paths),
so the feature is safe to ship dark.

### What made it run and be correct

1. **dlopen fix.** `moe_gemm_template_dispatch.h` threw with a bare
   `Arch::kMinComputeCapability` const-ref inside `MakeString`, ODR-using the undefined
   CUTLASS static `cutlass::arch::Sm80::kMinComputeCapability` and breaking `.so` load.
   Wrapping it in an `int{…}` prvalue removed the undefined symbol.
2. **`can_implement` accepts e2m1.** `moe_cutlass_kernel.h` rejected `float_e2m1_t`
   weights when `weight_scales != nullptr` (`kInvalid` for every tactic). Added
   `float_e2m1_t` to the groupwise-scale accept condition.
3. **Profiler-based tactic selection.** The SM80-FP4 prefill routes through the existing
   per-tactic profiler (with `wtype = kINT4` and `group_size = 32` to size the groupwise
   4-bit workspace). The profiler's per-tactic try/catch discards tiles that fail
   `can_implement`, so the fastest valid SM80 tile is chosen automatically.
4. **The numerics fix — interleave without bias.** The e2m1 dequant converter
   (`FastInterleavedAndBiasedNumericArrayConverter<half_t, float_e2m1_t, 8>`) *inverts*
   step 4's `[e0,e2,e4,e6,e1,e3,e5,e7]` nibble pair-interleave. The FP4 prepack passed
   `apply_bias_interleave=false`, which skipped step 4 **entirely** → the grouped GEMM
   read un-interleaved nibbles (`max_diff≈14`). Fix: a new interleave-only kernel
   (`interleave_int4s_inplace_kernel`) that applies the same permutation with the raw
   4-bit code (no `+8` bias), wired through `preprocess_weights_for_mixed_gemm_cuda`
   via a new `interleave_without_bias` parameter and enabled in `PrePackRepackFP4Weights`
   when `enable_fp4_sm80_gemm_`. The fused MXFP4 GEMV decode kernel (disabled under the
   SM80 flag) keeps the plain steps-1-3 layout, so the `gemv_fp4_fc*_weights_` buffer is
   safely repurposed for the grouped GEMM layout.

### Correctness

`test_qmoe_fp4_cuda.py` — 13 fp16-related tests pass with `ORT_FP4_SM80_GEMM=1`
(swiglu/silu, larger dims, more experts, token-count and top-4 variants); the headline
`test_fp4_fp16_swiglu` reports `max_diff=0.0078` (atol 0.12). All 20 tests pass with the
flag off (default path unchanged).

### Measured prefill speedup (gpt-oss-20b shape, H200, FC1 N=5760 K=2880 / FC2 N=2880 K=2880)

| tokens/expert | FP4 SM90 TMA (native) | **FP4 SM80 (this work)** | dense A16 fallback | speedup vs native |
|--------------:|----------------------:|-------------------------:|-------------------:|------------------:|
| 64            | 5.10 ms               | **1.08 ms**              | 8.12 ms            | 4.7×              |
| 128           | 9.53 ms               | **1.52 ms**              | —                  | 6.3×              |
| 256           | 10.24 ms              | **2.52 ms**              | 8.61 ms            | 4.1×              |
| 512           | 9.25 ms               | **4.44 ms**              | —                  | 2.1×              |

At the gpt-oss-20b prefill regime (~64 tok/expert) the SM80 path is **4.7× faster than the
SM90 TMA mixed kernel and 7.5× faster than the dense A16 dequant fallback**, landing close
to INT4's SM80 envelope (0.66 ms) — closing the bulk of the FP4-vs-INT4 prefill gap.

---

## 2026-06-30 — A100 / SM80 default-on FP4 grouped GEMM repair (manual port from `91ef688`)

Follow-up to the opt-in SM80 FP4 grouped-GEMM implementation above. The current
`tlwu/20260625/qmoe_fp4` branch had enabled `ORT_FP4_SM80_GEMM` by default, but on
an actual A100 build the default-on path failed before any useful timing could be
collected. This experiment records the baseline failure, the minimal manual port from
`91ef68800761bd5773032863d6426a58e062c83e`, and the repaired A100 numbers.

### Setup

- GPU: A100-SXM4-80GB, SM80, `CUDA_VISIBLE_DEVICES=0`.
- Branch/head under test: `tlwu/20260625/qmoe_fp4`, head
  `17694aa4e3768e6a24b54c55a207bb8901df449f` before local edits.
- Build: `build/cu130_fp4_bench/Release`, `CMAKE_CUDA_ARCHITECTURES=80`,
  `onnxruntime_USE_FP4_QMOE=ON`, CUDA 13.0 + cuDNN 9.23.
- Python path for probes/tests:

  ```bash
  PYTHONPATH=/home/tianlei/git/onnxruntime/build/cu130_fp4_bench/Release:\
/home/tianlei/git/onnxruntime/onnxruntime/test/python/transformers
  ```

- Important gate semantics on this branch: **do not** set
  `ORT_ENABLE_FP4_CUTLASS_GEMM=1` when testing the SM80 path. That env var requests
  the native/TMA path and suppresses the SM80 fallback. Use `ORT_FP4_SM80_GEMM=1`
  for default-on SM80, and `ORT_FP4_SM80_GEMM=0` for the dense fallback baseline.

### Baseline before the port

The fallback path was healthy, but the default-on SM80 path failed on production
shapes and on the small FP4 tests. The 512-token gpt-oss-shaped probe emitted:

```text
No valid GEMM config found for (N;K)=(5760;2880), dtype=1 wtype=9 gemm_type=0
No valid GEMM config found for (N;K)=(2880;2880), dtype=1 wtype=9 gemm_type=1
wfp4a16 (FP4 weights with FP16/BF16 activations) requires SM120+
```

Fallback validation with `ORT_FP4_SM80_GEMM=0` passed the SM80-relevant FP4 test
suite and established the pre-port fallback baseline:

| Tokens | Fallback latency |
|-------:|-----------------:|
| 512    | 18.21084 ms |
| 1024   | 18.60141 ms |
| 2048   | 19.45548 ms |
| 4096   | 21.40929 ms |

### Manual port from `91ef688`

Only the pieces needed to repair the observed A100 failure were ported; newer fixes
on `tlwu/20260625/qmoe_fp4` such as `interleave_without_bias` were preserved.

1. **SM80 WFP4A16 groupwise quant workspace + params**
   (`moe_gemm/moe_kernels.cu`): for `mSM < 90`, FP4 weights with FP16/BF16
   activations now allocate two groupwise scale workspaces and use
   `QuantParams::GroupWise(mGroupSize > 0 ? mGroupSize : 32, quant_1, quant_2)`.
   The SM90/TMA FP4 quant-param layout remains gated to the non-SM80 path.
2. **SM80 fp16 WFP4A16 dispatch**
   (`moe_gemm/moe_gemm_template_dispatch.h`): the SM80 branch now dispatches
   `use_wfp4a16 && T == half` to
   `dispatchMoeGemmToCutlass<..., cutlass::arch::Sm80, ...>` instead of throwing
   `requires SM120+`. BF16 still throws on this Ampere grouped-GEMM path.

The first change alone was insufficient: after rebuilding and staging the provider,
the same 512-token probe still failed with the `requires SM120+` throw. The second
dispatch change directly addressed that remaining failure.

### Rebuild / staging note

After each provider-only rebuild, the fresh provider was copied into both Python
load locations:

```bash
cd ~/git/onnxruntime && source .venv/bin/activate
cmake --build build/cu130_fp4_bench/Release --target onnxruntime_providers_cuda \
  --config Release -- -j96 \
  > /tmp/ort_fp4_sm80_dispatch_build.log 2>&1
cp build/cu130_fp4_bench/Release/libonnxruntime_providers_cuda.so \
  build/cu130_fp4_bench/Release/onnxruntime/capi/libonnxruntime_providers_cuda.so
cp build/cu130_fp4_bench/Release/libonnxruntime_providers_cuda.so \
  build/cu130_fp4_bench/Release/build/lib/onnxruntime/capi/libonnxruntime_providers_cuda.so
```

The final rebuild relinked `libonnxruntime_providers_cuda.so` and staged a provider
of size `109582528` bytes at all three locations.

### Correctness / engagement validation

The repaired default-on path now runs the production-shaped probe:

```bash
cd /tmp && source ~/git/onnxruntime/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0 ORT_FP4_SM80_GEMM=1 \
PYTHONPATH=/home/tianlei/git/onnxruntime/build/cu130_fp4_bench/Release:/home/tianlei/git/onnxruntime/onnxruntime/test/python/transformers \
python /home/tianlei/git/onnxruntime/onnxruntime/test/python/transformers/bench_fp4_gemv_autotune.py \
  --tokens 512 --dtype fp16 --warmup 2 --iters 5 --reps 1
```

Result: **1713.32 us/run** at 512 tokens, with no `No valid GEMM config` warnings
and no SM120 throw.

The SM80-relevant FP4 CUDA tests also pass with the default-on path. The test harness
was run from `/tmp`, monkeypatching the old SM90 skip (`_skip_if_no_fp4`) and clamping
the default ONNX opset to 26. The native SM90-only prepack-scale test was excluded
from the SM80 subset.

```text
Ran 14 tests in 2.105s
OK
```

The full class run had the same 14 functional passes plus one expected SM90-only
skip (`test_fp4_native_cutlass_row_varying_scales`).

### A100 benchmark sweep (clock locked to 1410 MHz)

Bench command shape: gpt-oss QMoE node (`hidden=2880`, `inter=2880`, `E=32`,
`top_k=4`, fp16), `--warmup 5 --iters 20 --reps 3`. GPU clocks were locked with
`sudo -n nvidia-smi -i 0 -lgc 1410` and reset with `-rgc` after the sweep.

| Tokens | SM80 FP4 default-on (`ORT_FP4_SM80_GEMM=1`) | Fallback (`ORT_FP4_SM80_GEMM=0`) | Speedup |
|-------:|--------------------------------------------:|----------------------------------:|--------:|
| 512    | **1.69445 ms** | 18.20956 ms | **10.75×** |
| 1024   | **2.50430 ms** | 18.61366 ms | **7.43×** |
| 2048   | **4.03634 ms** | 19.45979 ms | **4.82×** |
| 4096   | **7.35478 ms** | 21.35033 ms | **2.90×** |

Compared with the pre-port baseline, the fallback numbers are unchanged within
noise, while the default-on path changes from hard failure to a large speedup.

### Status / verdict

- The manual port repairs the A100 default-on SM80 FP4 grouped-GEMM path with only
  two focused source changes.
- SM80 fp16 WFP4A16 grouped GEMM is correct on the SM80-relevant FP4 tests and gives
  a **2.9×–10.8×** QMoE-node speedup over the dense fallback across 512–4096 tokens.
- BF16 Ampere grouped-GEMM dispatch remains intentionally unsupported in this port;
  BF16 test coverage still passes through the supported fallback paths.
- The result makes `ORT_FP4_SM80_GEMM` default-on viable on A100 for fp16 production
  shapes, while preserving `ORT_FP4_SM80_GEMM=0` as a fallback / comparison knob.

## 2026-06-30 — A100 / SM80 BF16 FP4 grouped GEMM enablement

Follow-up to the A100 repair above. The FP16 path proved that the SM80 fused-dequant
grouped GEMM and groupwise MXFP4 scale plumbing are correct on A100; the remaining
BF16 blocker was policy/gating rather than a missing CUTLASS specialization.

### Change

- `moe_quantization.cc`: `enable_fp4_sm80_gemm_` is no longer gated on `is_fp16_`.
  In the FP4 fallback regime (`sm < 120`) and when native FP4 CUTLASS is not
  explicitly requested, both FP16 and BF16 activation QMoE nodes now build the
  SM80 FP4 runner and prepack the SM80 interleaved e2m1 weights plus activation-dtype
  group scales.
- `moe_gemm_template_dispatch.h`: the SM80 `use_wfp4a16` branch now dispatches both
  `T == half` and `T == __nv_bfloat16` to the Ampere grouped GEMM instead of throwing
  for BF16.
- Nearby comments were updated from fp16-only / fp16-scale wording to FP16/BF16 /
  activation-dtype wording.

### Validation

Provider build and staging:

```bash
cd /home/tianlei/git/onnxruntime
cmake --build build/cu130_fp4_bench/Release --target onnxruntime_providers_cuda --parallel 16
cp build/cu130_fp4_bench/Release/libonnxruntime_providers_cuda.so \
  build/cu130_fp4_bench/Release/onnxruntime/capi/libonnxruntime_providers_cuda.so
cp build/cu130_fp4_bench/Release/libonnxruntime_providers_cuda.so \
  build/cu130_fp4_bench/Release/build/lib/onnxruntime/capi/libonnxruntime_providers_cuda.so
```

Result: provider build passed, including `moe_gemm_kernels_bf16_fp4.cu.o`; staged
provider size was `109590720` bytes at all three locations.

512-token gpt-oss-shaped BF16 probe (`hidden=inter=2880`, `E=32`, `top_k=4`,
`--warmup 2 --iters 5 --reps 1`):

| Setting | BF16 latency |
|---------|-------------:|
| `ORT_FP4_SM80_GEMM=1` | **1.72201 ms** |
| `ORT_FP4_SM80_GEMM=0` | 19.44485 ms |

Speedup: **11.29x** over the dense fallback. This also confirms the env gate is
selecting the new BF16 SM80 grouped-GEMM path.

SM80-relevant FP4 CUDA tests were run with `_skip_if_no_fp4` monkeypatched off
and `ORT_FP4_SM80_GEMM=1`. The native SM90-only scale-prepack test still skipped
on A100:

```text
Ran 15 tests in 2.576s
OK (skipped=1)
```

BF16-specific test headlines:

```text
test_fp4_bf16_silu_basic ... max_diff=0.031250 ok
test_fp4_bf16_swiglu ... max_diff=0.062500 ok
test_fp4_decode_swiglu_gemv_2 ... max_diff=0.062500 ok
test_fp4_decode_swiglu_gemv_3 ... max_diff=0.062500 ok
```

### Status / verdict

BF16 WFP4A16 is now enabled on the same A100 SM80 grouped-GEMM path as FP16. The
runtime probe and unit tests validate both correctness and fast-path engagement;
`ORT_FP4_SM80_GEMM=0` remains the fallback/comparison knob.
