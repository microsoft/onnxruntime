// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp4.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstring>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif

#include "core/platform/env_var_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime::contrib::cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080

namespace {

// bit_cast / e4m3_to_float / from_float / to_float / Vec2Traits come from cuda_type_helper.cuh.

template <typename T>
__global__ void DequantizeNvFp4Kernel(T* __restrict__ out,
                                      const uint8_t* __restrict__ b_packed,
                                      const uint8_t* __restrict__ weight_scale,
                                      const float* __restrict__ weight_scale_2,
                                      int n,
                                      int k,
                                      int k_blocks,
                                      int block_size) {
  const int half_k = k >> 1;
  const long long total = static_cast<long long>(n) * half_k;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int row = static_cast<int>(idx / half_k);
  const int pair = static_cast<int>(idx - static_cast<long long>(row) * half_k);
  const int k0 = pair << 1;

  const uint8_t packed = b_packed[idx];
  const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(packed), __NV_E2M1);
  const __half2 h2 = __half2(hr);
  const float2 v = __half22float2(h2);

  const float g = *weight_scale_2;
  const int blk0 = k0 / block_size;
  const int blk1 = (k0 + 1) / block_size;
  const float s0 = e4m3_to_float(weight_scale[row * k_blocks + blk0]) * g;
  const float s1 = e4m3_to_float(weight_scale[row * k_blocks + blk1]) * g;

  const long long out_base = static_cast<long long>(row) * k + k0;
  out[out_base] = from_float<T>(v.x * s0);
  out[out_base + 1] = from_float<T>(v.y * s1);
}

template <typename T>
__global__ void AddBiasKernel(T* __restrict__ y, const T* __restrict__ bias, int m, int n) {
  const long long total = static_cast<long long>(m) * n;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % n);
  y[idx] = from_float<T>(to_float<T>(y[idx]) + to_float<T>(bias[col]));
}

// Number of warps (= output columns) per thread block in the GEMV.
constexpr int kGemvWarpsPerBlock = 8;

// Largest M tile the GEMV will fold into a single block. M is at most kGemvMaxM (8) at the
// call site; beyond 4 rows the register pressure from the per-row accumulators and the A
// fragments starts to cost more occupancy than the extra weight reuse buys.
constexpr int kGemvMaxRowsPerBlock = 4;

// SM75 allows only 1024 resident threads per multiprocessor, i.e. 4 blocks of 256. Asking for
// more there is unsatisfiable and only makes ptxas over-restrict registers and spill.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
constexpr int kGemvMaxBlocksPerSm = 4;
#else
constexpr int kGemvMaxBlocksPerSm = 8;
#endif

// Occupancy target for the GEMV. Holding RowsPerBlock accumulators costs registers, and left
// unconstrained nvcc trades occupancy for scheduling freedom: the 4-row tile lands on 66
// registers, which drops the block from 4 to 3 per SM and costs ~7% on the large shapes. Pin a
// register budget per tile instead; all three instantiations compile spill-free at these targets.
template <int RowsPerBlock>
struct GemvMinBlocksPerSm {
  static constexpr int target = (RowsPerBlock == 1) ? 6 : ((RowsPerBlock == 2) ? 5 : 4);
  static constexpr int value = target < kGemvMaxBlocksPerSm ? target : kGemvMaxBlocksPerSm;
};

// -----------------------------------------------------------------------------
// Fused NVFP4 weight-only GEMV fast path for the decode phase (small M).
//
// Each warp computes RowsPerBlock output elements Y[row0 .. row0 + RowsPerBlock, col].
// The 32 lanes cooperatively reduce over K reading the packed NVFP4 weight directly
// (two E2M1 values per byte) with 16-byte coalesced loads, so the weight is streamed
// exactly once per block and no [N, K] dequantized buffer is materialized. Each lane
// consumes 32 contiguous K elements = 16 packed bytes, which span exactly two
// 16-element blocks; the two per-block E4M3 scales are folded in per half. The global
// fp32 scale is applied once after the warp reduction. Runs on any architecture with
// NVFP4 conversion intrinsics (CUDA >= 12.8), including SM90 and SM120.
//
// RowsPerBlock > 1 amortizes the weight load and the E2M1 decode across several rows
// of A, which matters for speculative decoding / MTP verify where M = N_spec + 1 > 1.
// It trades grid parallelism for that reuse (M no longer contributes to gridDim.y), so
// the launcher only enables it when gridDim.x alone already fills the device; see
// Fp4GemvRowsPerBlock() below. The per-row fp32 accumulation order is independent of
// RowsPerBlock, so results are bit-identical across tilings.
//
// Note that on SM80+ the tensor-core kernel further below takes precedence whenever
// K % 128 == 0, so row tiling is what actually runs on pre-SM80 devices or when K is not a
// multiple of 128.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Fast NVFP4 (E2M1) -> half / bfloat16 conversion.
//
// __nv_cvt_fp4x2_to_halfraw2() is emulated in software on pre-Blackwell parts and
// costs ~10 ALU ops per pair, which dominates the decode GEMV below (measured 3.5x
// slowdown on H200). E2M1 has only eight magnitudes {0, 0.5, 1, 1.5, 2, 3, 4, 6},
// so the 16-bit float bit pattern is built directly with a `prmt.b32` byte-select
// from packed magnitude constants plus a shifted sign bit. One prmt performs *four*
// magnitude lookups at once, so a whole 32-bit weight word (eight codes) decodes in
// ~14 instructions instead of ~48 for the per-byte bit-twiddling it replaces. This
// matters because the GEMV is instruction-issue bound, not bandwidth bound (it runs
// at 1.57 TB/s = 33% of H200 HBM peak).
//
// The magnitude bytes below are the exact half/bf16 encodings of the eight FP4
// values, so the decoded value is bit-identical to the previous path (which produced
// value * 2^-(bias - 1) and multiplied by 2^(bias - 1) afterwards) and to the
// __nv_cvt_fp4x2_to_halfraw2() intrinsic. The scalar fallback keeps the host-side
// build working and is used when inline PTX is unavailable.
//
// Shared with the QMoE FP4 GEMV; see Fp4I2FConverter in
// contrib_ops/cuda/llm/fpA_intB_gemv/details.h for the same decode.
template <typename T>
struct Fp4Cvt;

template <>
struct Fp4Cvt<half> {
  using Traits = Vec2Traits<half>;
  using T2 = typename Traits::Type2;

  // Decodes four consecutive E2M1 codes. `mag_sel` holds the four 3-bit magnitudes as the four
  // low nibbles (bit 3 cleared so prmt stays in byte-select mode rather than sign-replicate
  // mode); `sgn_sel` holds the four sign bits as 0/1 nibbles.
  static __device__ __forceinline__ void DecodeQuad(uint32_t mag_sel, uint32_t sgn_sel,
                                                    T2& lo2, T2& hi2) {
#if defined(__CUDA_ARCH__)
    // Sign bytes: nibble 0 -> 0x00, nibble 1 -> 0x80 (bit 7 of the half high byte).
    uint32_t sb;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sb) : "r"(0x00008000u), "r"(0u), "r"(sgn_sel));
    // half high byte per magnitude (low byte is always 0):
    //   codes 0..3 -> {0x00, 0x38, 0x3C, 0x3E}, codes 4..7 -> {0x40, 0x42, 0x44, 0x46}.
    uint32_t hb;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hb) : "r"(0x3E3C3800u), "r"(0x46444240u), "r"(mag_sel));
    hb |= sb;
    // Expand {b0,b1,b2,b3} to {0,b0,0,b1} and {0,b2,0,b3}, pulling the zero low bytes from the
    // second (all-zero) prmt operand.
    uint32_t lo, hi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(hb), "r"(0u), "n"(0x1404));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(hb), "r"(0u), "n"(0x3424));
    lo2 = bit_cast<T2>(lo);
    hi2 = bit_cast<T2>(hi);
#else
    ScalarDecodeQuad(mag_sel, sgn_sel, lo2, hi2);
#endif
  }

  static __device__ __forceinline__ void ScalarDecodeQuad(uint32_t mag_sel, uint32_t sgn_sel,
                                                          T2& lo2, T2& hi2) {
    constexpr uint16_t kMag[8] = {0x0000u, 0x3800u, 0x3C00u, 0x3E00u,
                                  0x4000u, 0x4200u, 0x4400u, 0x4600u};
    uint16_t e[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      e[i] = static_cast<uint16_t>(kMag[(mag_sel >> (4 * i)) & 0x7u] |
                                   (((sgn_sel >> (4 * i)) & 0x1u) << 15));
    }
    lo2 = bit_cast<T2>(static_cast<uint32_t>(e[0]) | (static_cast<uint32_t>(e[1]) << 16));
    hi2 = bit_cast<T2>(static_cast<uint32_t>(e[2]) | (static_cast<uint32_t>(e[3]) << 16));
  }

  static __device__ __forceinline__ T2 Mul(T2 a, T2 b) { return Traits::mul2(a, b); }
  static __device__ __forceinline__ float2 ToFloat2(T2 v) { return Traits::to_float2(v); }
};

template <>
struct Fp4Cvt<nv_bfloat16> {
  using Traits = Vec2Traits<nv_bfloat16>;
  using T2 = typename Traits::Type2;

  static __device__ __forceinline__ void DecodeQuad(uint32_t mag_sel, uint32_t sgn_sel,
                                                    T2& lo2, T2& hi2) {
#if defined(__CUDA_ARCH__)
    uint32_t sb;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(sb) : "r"(0x00008000u), "r"(0u), "r"(sgn_sel));
    // bf16 high byte {0x00,0x3F,0x3F,0x3F, 0x40,0x40,0x40,0x40} and
    //      low  byte {0x00,0x00,0x80,0xC0, 0x00,0x40,0x80,0xC0}.
    uint32_t hb, lb;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hb) : "r"(0x3F3F3F00u), "r"(0x40404040u), "r"(mag_sel));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lb) : "r"(0xC0800000u), "r"(0xC0804000u), "r"(mag_sel));
    hb |= sb;
    // bf16 needs both bytes: interleave low/high bytes of elements 0,1 and 2,3.
    uint32_t lo, hi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(lb), "r"(hb), "n"(0x5140));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(lb), "r"(hb), "n"(0x7362));
    lo2 = bit_cast<T2>(lo);
    hi2 = bit_cast<T2>(hi);
#else
    ScalarDecodeQuad(mag_sel, sgn_sel, lo2, hi2);
#endif
  }

  static __device__ __forceinline__ void ScalarDecodeQuad(uint32_t mag_sel, uint32_t sgn_sel,
                                                          T2& lo2, T2& hi2) {
    constexpr uint16_t kMag[8] = {0x0000u, 0x3F00u, 0x3F80u, 0x3FC0u,
                                  0x4000u, 0x4040u, 0x4080u, 0x40C0u};
    uint16_t e[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      e[i] = static_cast<uint16_t>(kMag[(mag_sel >> (4 * i)) & 0x7u] |
                                   (((sgn_sel >> (4 * i)) & 0x1u) << 15));
    }
    lo2 = bit_cast<T2>(static_cast<uint32_t>(e[0]) | (static_cast<uint32_t>(e[1]) << 16));
    hi2 = bit_cast<T2>(static_cast<uint32_t>(e[2]) | (static_cast<uint32_t>(e[3]) << 16));
  }

  static __device__ __forceinline__ T2 Mul(T2 a, T2 b) { return Traits::mul2(a, b); }
  static __device__ __forceinline__ float2 ToFloat2(T2 v) { return Traits::to_float2(v); }
};

template <typename T, int RowsPerBlock>
__global__ __launch_bounds__(32 * kGemvWarpsPerBlock,
                             GemvMinBlocksPerSm<RowsPerBlock>::value) void MatMulBlockQuantizedFp4WeightGemvKernel(T* __restrict__ y,
                                                                                                                   const T* __restrict__ a,
                                                                                                                   const uint8_t* __restrict__ b_packed,
                                                                                                                   const uint8_t* __restrict__ weight_scale,
                                                                                                                   const float* __restrict__ weight_scale_2,
                                                                                                                   const T* __restrict__ bias,
                                                                                                                   int m,
                                                                                                                   int n,
                                                                                                                   int k,
                                                                                                                   int k_blocks) {
  using Cvt = Fp4Cvt<T>;
  using T2 = typename Cvt::T2;

  const int lane = threadIdx.x;                           // 0..31
  const int col = blockIdx.x * blockDim.y + threadIdx.y;  // n
  const int row0 = blockIdx.y * RowsPerBlock;             // m
  if (row0 >= m || col >= n) {
    return;
  }

  // Clamp the row base pointers instead of masking inside the K loop: a ragged tail tile
  // recomputes the last row and discards it at the store, which keeps the inner loop
  // branch-free. Only the last block of gridDim.y can be ragged, and only when
  // RowsPerBlock does not divide M.
  const T* a_rows[RowsPerBlock];
#pragma unroll
  for (int r = 0; r < RowsPerBlock; ++r) {
    a_rows[r] = a + static_cast<size_t>(row0 + r < m ? row0 + r : m - 1) * k;
  }
  const uint8_t* b_row = b_packed + static_cast<size_t>(col) * (k >> 1);
  const uint8_t* ws_row = weight_scale + static_cast<size_t>(col) * k_blocks;

  constexpr int kBlockSize = 16;
  constexpr int kElemsPerLane = 32;       // two 16-element blocks
  const int stride = 32 * kElemsPerLane;  // 1024 elements per warp iteration

  float acc[RowsPerBlock];
#pragma unroll
  for (int r = 0; r < RowsPerBlock; ++r) {
    acc[r] = 0.0f;
  }

  for (int base = 0; base < k; base += stride) {
    const int koff = base + lane * kElemsPerLane;
    if (koff < k) {
      // 16-byte vectorized loads. Both are guaranteed to be naturally aligned, so no guarded
      // fallback is needed:
      //   * b_packed / a come from ORT device allocations, whose base address is at least
      //     256-byte aligned (see CUDAAllocator / cudaMalloc guarantees).
      //   * The launcher enforces k % 32 == 0, so the row offsets are multiples of 16 bytes:
      //     b_row = b_packed + col * (k / 2) with (k / 2) % 16 == 0, and
      //     a_row = a + row * k with k * sizeof(T) % 16 == 0 for 2-byte T.
      //   * Within a row, byte offsets are (koff / 2) for B and koff * sizeof(T) for A, and
      //     koff is a multiple of kElemsPerLane == 32.
      const uint4 packed = *reinterpret_cast<const uint4*>(b_row + (koff >> 1));
      const uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};

      // Decode the 32 weights owned by this lane once; every row of A reuses them.
      // Each 32-bit weight word holds 8 codes = 4 T2 pairs; words 0/1 cover the first
      // 16-element scale block and words 2/3 the second.
      T2 b2[4][4];
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const uint32_t mag = words[w] & 0x77777777u;
        const uint32_t sgn = (words[w] >> 3) & 0x11111111u;
        Cvt::DecodeQuad(mag, sgn, b2[w][0], b2[w][1]);
        Cvt::DecodeQuad(mag >> 16, sgn >> 16, b2[w][2], b2[w][3]);
      }

      const int kb0 = koff / kBlockSize;
      const int kb1 = kb0 + 1;
      const float s0 = e4m3_to_float(ws_row[kb0]);
      const float s1 = e4m3_to_float(ws_row[kb1]);

#pragma unroll
      for (int r = 0; r < RowsPerBlock; ++r) {
        const uint4* ap = reinterpret_cast<const uint4*>(a_rows[r] + koff);

        // fp32 accumulation per 16-element scale block, matching the reference path.
        // One uint4 of A is live at a time so the 4-row tile stays inside its register budget.
        float p[2] = {0.0f, 0.0f};
#pragma unroll
        for (int w = 0; w < 4; ++w) {
          uint4 av4 = ap[w];
          const T2* av = reinterpret_cast<const T2*>(&av4);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const float2 pv = Cvt::ToFloat2(Cvt::Mul(av[j], b2[w][j]));
            p[w >> 1] += pv.x + pv.y;
          }
        }
        acc[r] += p[0] * s0 + p[1] * s1;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RowsPerBlock; ++r) {
    float v = acc[r];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if (lane == 0 && row0 + r < m) {
      float result = v * (*weight_scale_2);
      if (bias != nullptr) {
        result += to_float<T>(bias[col]);
      }
      y[static_cast<size_t>(row0 + r) * n + col] = from_float<T>(result);
    }
  }
}

// -----------------------------------------------------------------------------
// Tensor-core sub-path for the decode GEMV (SM80+).
//
// The scalar path above assigns one output column to a warp and reduces over K with
// __shfl_down. That makes it re-read the whole A tile once per output column: with
// RowsPerBlock = 4 a warp pulls 16 KB of activation for 1 KB of weight, a 16:1
// amplification that dominates the lm_head shape (N = 248320: 538 us on H200).
//
// This path replaces the shuffle reduction with mma.m16n8k16, which computes 16 output
// columns x up to 8 rows per warp and cuts the activation re-reads by 16x. The trick that
// makes the fragments free is to put the *weight* in the mma A slot and the *activation*
// in the mma B slot: B_weight is [N, K] row-major, which is exactly the A-row-major
// fragment, and A_act is [M, K] row-major, which is exactly the B-col-major fragment. No
// transpose, no ldmatrix, no shared-memory staging. The mma "M" extent becomes our column
// count (16) and the mma "N" extent becomes our row count (M <= 8).
//
// Per-lane fragment ownership (PTX m16n8k16 layout, g = lane >> 2, t = lane & 3). A lane does NOT
// compute a whole dot product on its own: the tensor core exchanges partial products across the
// warp, so the A rows, the B column and the D rows a lane holds are three independent slices.
//
//   ra[0] = (mma row g,     mma k 2t,   2t+1)  <- weight column col_lo
//   ra[1] = (mma row g + 8, mma k 2t,   2t+1)  <- weight column col_hi = col_lo + 8
//   ra[2] = (mma row g,     mma k 2t+8, 2t+9)  <- weight column col_lo
//   ra[3] = (mma row g + 8, mma k 2t+8, 2t+9)  <- weight column col_hi
//   rb[0] = (mma k 2t,   2t+1, mma col g)      <- activation row g
//   rb[1] = (mma k 2t+8, 2t+9, mma col g)      <- activation row g
//   acc[0], acc[1] = (mma row g,     mma col 2t, 2t+1) -> y[2t][col_lo], y[2t+1][col_lo]
//   acc[2], acc[3] = (mma row g + 8, mma col 2t, 2t+1) -> y[2t][col_hi], y[2t+1][col_hi]
//
// So `g` selects both the output-column pair (through the A fragment) and the activation row
// (through the B fragment), while the accumulator the lane ends up owning covers output rows 2t
// and 2t + 1. Loads keyed off `g` and stores keyed off `t` are therefore both correct and are
// deliberately different; see the GemvTensorCoreLaneOwnership* tests for a one-hot probe of this
// mapping.
//
// The K axis is then permuted so that the four k-slots a lane needs per mma step are
// contiguous in memory. This is legal because K is a reduction axis and the same
// permutation is applied to both operands. A window is 128 K elements = 64 packed bytes;
// lane (g, t) owns elements [32t, 32t + 32), i.e. one uint4 of weight and one uint4 of
// activation per 32 elements, and exactly two 16-element NVFP4 scale blocks.
//
// Because the mma sums across all four t lanes -- which hold *different* scale blocks --
// the accumulator cannot be flushed per block the way the scalar path does. The per-block
// E4M3 scale is instead folded into the decoded weight before the mma. That is exact in
// both half and bfloat16: E2M1 magnitudes carry 2 significand bits and E4M3 scales carry
// 4, so the product needs at most 6, well inside half's 11 and bfloat16's 8. The range is
// safe too (max 6 * 448 = 2688 < 65504, min 0.5 * 2^-9 = 2^-10 >> half's 2^-14 minimum).
//
// KSplit warps per block take a strided share of the K windows and reduce through shared
// memory. Without it, 16 columns per warp yields 16x fewer warps than the scalar path and
// the small MLP shapes lose more to idle SMs than they gain per warp.
//
// Note the fp32 accumulation order differs from the scalar path, so this path is not
// bit-identical to it. Measured max relative error against an fp64 reference is identical
// to the scalar path on every shape tested (2.4e-04 .. 3.5e-04, i.e. NVFP4 quantization
// noise, not accumulation noise).
template <typename T>
struct Fp4GemvMma;

template <>
struct Fp4GemvMma<half> {
  using T2 = half2;

  // E4M3 -> half is exact, and half broadcasts into both lanes of the half2 weight pair.
  static __device__ __forceinline__ T2 BroadcastScale(uint8_t e4m3) {
    const half h = static_cast<half>(
        __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(e4m3), __NV_E4M3));
    return __half2half2(h);
  }

  static __device__ __forceinline__ uint32_t Pack(T2 v) {
    uint32_t r;
    memcpy(&r, &v, sizeof(r));
    return r;
  }

  static __device__ __forceinline__ void Mma(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    ORT_UNUSED_PARAMETER(d);
    ORT_UNUSED_PARAMETER(a);
    ORT_UNUSED_PARAMETER(b);
#endif
  }
};

template <>
struct Fp4GemvMma<nv_bfloat16> {
  using T2 = nv_bfloat162;

  // E4M3 has 4 significand bits and an exponent range of [-9, 8], so the half hop and the
  // bfloat16 result are both exact.
  static __device__ __forceinline__ T2 BroadcastScale(uint8_t e4m3) {
    const half h = static_cast<half>(
        __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(e4m3), __NV_E4M3));
    return __bfloat162bfloat162(__float2bfloat16(__half2float(h)));
  }

  static __device__ __forceinline__ uint32_t Pack(T2 v) {
    uint32_t r;
    memcpy(&r, &v, sizeof(r));
    return r;
  }

  static __device__ __forceinline__ void Mma(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    ORT_UNUSED_PARAMETER(d);
    ORT_UNUSED_PARAMETER(a);
    ORT_UNUSED_PARAMETER(b);
#endif
  }
};

// Decodes one 32-bit packed weight word (eight E2M1 codes) into four T2 pairs, so that
// out[i] holds elements {2i, 2i + 1}.
template <typename T>
__device__ __forceinline__ void Fp4DecodeWord(uint32_t word, typename Fp4Cvt<T>::T2 (&out)[4]) {
  const uint32_t mag = word & 0x77777777u;
  const uint32_t sgn = (word >> 3) & 0x11111111u;
  Fp4Cvt<T>::DecodeQuad(mag, sgn, out[0], out[1]);
  Fp4Cvt<T>::DecodeQuad(mag >> 16, sgn >> 16, out[2], out[3]);
}

template <typename T, int KSplit, int ColTiles>
__global__ __launch_bounds__(32 * KSplit * ColTiles, 1) void MatMulBlockQuantizedFp4WeightMmaGemvKernel(
    T* __restrict__ y,
    const T* __restrict__ a,
    const uint8_t* __restrict__ b_packed,
    const uint8_t* __restrict__ weight_scale,
    const float* __restrict__ weight_scale_2,
    const T* __restrict__ bias,
    int m,
    int n,
    int k,
    int k_blocks) {
  using Cvt = Fp4Cvt<T>;
  using MmaOp = Fp4GemvMma<T>;
  using T2 = typename Cvt::T2;

  const int lane = threadIdx.x;
  // See the fragment-ownership table above: `g` indexes the A rows (output columns col_lo/col_hi)
  // and the B column (activation row); `t` indexes the 32-element K slice and, in the accumulator,
  // the output rows 2t / 2t + 1 this lane stores.
  const int g = lane >> 2;
  const int t = lane & 3;
  const int warp = static_cast<int>(threadIdx.y);
  const int warp_k = (KSplit == 1) ? 0 : (warp % KSplit);
  const int warp_c = (KSplit == 1) ? warp : (warp / KSplit);

  const int col_lo = (static_cast<int>(blockIdx.x) * ColTiles + warp_c) * 16 + g;
  const int col_hi = col_lo + 8;
  const bool lo_ok = col_lo < n;
  const bool hi_ok = col_hi < n;

  // Out-of-range columns fold onto column 0 so every load stays in bounds; their results
  // are masked off at the store.
  const int half_k = k >> 1;
  const uint8_t* b_lo = b_packed + static_cast<size_t>(lo_ok ? col_lo : 0) * half_k;
  const uint8_t* b_hi = b_packed + static_cast<size_t>(hi_ok ? col_hi : 0) * half_k;
  const uint8_t* ws_lo = weight_scale + static_cast<size_t>(lo_ok ? col_lo : 0) * k_blocks;
  const uint8_t* ws_hi = weight_scale + static_cast<size_t>(hi_ok ? col_hi : 0) * k_blocks;

  const bool a_ok = g < m;
  const T* a_row = a + static_cast<size_t>(a_ok ? g : 0) * k;

  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const int windows = k >> 7;

  for (int wi = warp_k; wi < windows; wi += KSplit) {
    const int kbase = (wi << 7) + (t << 5);
    const uint4 wl4 = *reinterpret_cast<const uint4*>(b_lo + (kbase >> 1));
    const uint4 wh4 = *reinterpret_cast<const uint4*>(b_hi + (kbase >> 1));
    const int kb = (wi << 3) + (t << 1);
    const T2 sl[2] = {MmaOp::BroadcastScale(ws_lo[kb]), MmaOp::BroadcastScale(ws_lo[kb + 1])};
    const T2 sh[2] = {MmaOp::BroadcastScale(ws_hi[kb]), MmaOp::BroadcastScale(ws_hi[kb + 1])};
    const uint4* ap = reinterpret_cast<const uint4*>(a_row + kbase);
    const uint32_t wl[4] = {wl4.x, wl4.y, wl4.z, wl4.w};
    const uint32_t wh[4] = {wh4.x, wh4.y, wh4.z, wh4.w};

#pragma unroll
    for (int w = 0; w < 4; ++w) {
      T2 bl[4], bh[4];
      Fp4DecodeWord<T>(wl[w], bl);
      Fp4DecodeWord<T>(wh[w], bh);
      // Words 0,1 fall in the first 16-element scale block of this lane's slice, words 2,3
      // in the second.
      const T2 s0 = sl[w >> 1];
      const T2 s1 = sh[w >> 1];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        bl[i] = Cvt::Mul(bl[i], s0);
        bh[i] = Cvt::Mul(bh[i], s1);
      }
      const uint4 av4 = a_ok ? ap[w] : make_uint4(0u, 0u, 0u, 0u);
      const uint32_t* av = reinterpret_cast<const uint32_t*>(&av4);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        // A regs are (row g, k 2t..2t+1), (row g+8, ...), (row g, k 2t+8..), (row g+8, ...).
        const uint32_t ra[4] = {MmaOp::Pack(bl[2 * j]), MmaOp::Pack(bh[2 * j]),
                                MmaOp::Pack(bl[2 * j + 1]), MmaOp::Pack(bh[2 * j + 1])};
        const uint32_t rb[2] = {av[2 * j], av[2 * j + 1]};
        MmaOp::Mma(acc, ra, rb);
      }
    }
  }

  if constexpr (KSplit > 1) {
    __shared__ float red[ColTiles * KSplit * 32 * 4];
    float* mine = red + ((warp_c * KSplit + warp_k) * 32 + lane) * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      mine[i] = acc[i];
    }
    __syncthreads();
    if (warp_k != 0) {
      return;
    }
#pragma unroll
    for (int ws = 1; ws < KSplit; ++ws) {
      const float* other = red + ((warp_c * KSplit + ws) * 32 + lane) * 4;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[i] += other[i];
      }
    }
  }

  // D regs are (row g, cols 2t, 2t+1) and (row g+8, cols 2t, 2t+1): mma rows are our output
  // columns and mma columns are our M rows.
  const float g2 = *weight_scale_2;
  const int r0 = t << 1;
  const float bias_lo = (bias != nullptr && lo_ok) ? to_float<T>(bias[col_lo]) : 0.0f;
  const float bias_hi = (bias != nullptr && hi_ok) ? to_float<T>(bias[col_hi]) : 0.0f;
  if (lo_ok) {
    if (r0 < m) {
      y[static_cast<size_t>(r0) * n + col_lo] = from_float<T>(acc[0] * g2 + bias_lo);
    }
    if (r0 + 1 < m) {
      y[static_cast<size_t>(r0 + 1) * n + col_lo] = from_float<T>(acc[1] * g2 + bias_lo);
    }
  }
  if (hi_ok) {
    if (r0 < m) {
      y[static_cast<size_t>(r0) * n + col_hi] = from_float<T>(acc[2] * g2 + bias_hi);
    }
    if (r0 + 1 < m) {
      y[static_cast<size_t>(r0 + 1) * n + col_hi] = from_float<T>(acc[3] * g2 + bias_hi);
    }
  }
}

bool Fp4GemvRowTilingEnabled() {
  static bool const enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<bool>("ORT_FP4_GEMV_ROW_TILING", true);
  return enabled;
}

// Picks the M tile for the GEMV.
//
// Folding M into the block removes gridDim.y parallelism: the grid shrinks from
// ceil(N / warps) * M blocks to ceil(N / warps) * ceil(M / RowsPerBlock). That is a win when
// the kernel is weight-bound and the device is already saturated by the N dimension alone
// (the packed weight and its E2M1 decode are amortized over RowsPerBlock rows), and a loss
// when it leaves SMs idle. Measured on H200 (132 SMs, M = 4, half):
//
//   N = 248320 (lm_head)      31040 col blocks   615.8 -> 537.7 us  (1.15x)
//   N =   2048 (shared down)    256 col blocks     4.30 ->  3.33 us  (1.29x)
//   N =    512 (shared gate)     64 col blocks     3.47 ->  4.27 us  (0.81x, fewer blocks than SMs)
//
// So gate on the column grid covering at least one full wave of SMs on its own.
int Fp4GemvRowsPerBlock(int m, int n, int sm_count) {
  if (m <= 1 || !Fp4GemvRowTilingEnabled()) {
    return 1;
  }
  const int col_blocks = (n + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
  if (col_blocks < sm_count) {
    return 1;
  }
  // Only 1, 2 and 4 are instantiated: a 3-row tile spills and is no better than two 2-row
  // blocks, and M > 4 splits across gridDim.y (M is at most 8 here).
  return (m >= kGemvMaxRowsPerBlock) ? kGemvMaxRowsPerBlock : 2;
}

bool Fp4GemvMmaEnabled() {
  static bool const enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<bool>("ORT_FP4_GEMV_MMA", true);
  return enabled;
}

struct Fp4MmaConfig {
  int k_split;
  int col_tiles;
};

// Picks the (KSplit, ColTiles) shape for the tensor-core GEMV.
//
// A warp owns 16 output columns, so the column grid is 16x smaller than the scalar path's.
// Wide column blocks (ColTiles = 4) give the best weight locality, but only pay off once N
// alone still covers a full wave of SMs; below that the block count collapses and the
// device idles, so columns are traded back for K-split warps. Measured on H200 (132 SMs,
// M = 4, half), scalar path -> best tensor-core config:
//
//   N = 248320, K = 2048 (lm_head)      538.5 -> 107.7 us (5.00x)  KSplit 2,  ColTiles 4
//   N =    512, K = 2048 (gate/up)        3.90 ->  3.34 us (1.11x) KSplit 16, ColTiles 1
//   N =   2048, K =  512 (down)           4.16 ->  2.97 us (1.40x) KSplit 4,  ColTiles 1
Fp4MmaConfig PickFp4MmaConfig(int n, int k, int sm_count) {
  const int windows = k >> 7;  // >= 1; the launcher only takes this path when k % 128 == 0
  const int col_tiles = (n + 15) / 16;

  if ((col_tiles + 3) / 4 >= sm_count) {
    return {std::min(2, windows), 4};
  }
  // Column-starved: give every block as many K-split warps as there are windows, up to the
  // 16-warp (512-thread) block ceiling.
  int k_split = 1;
  while (k_split < 16 && k_split * 2 <= windows) {
    k_split <<= 1;
  }
  return {k_split, 1};
}

}  // namespace

#endif  // CUDA_VERSION >= 12080

Status LaunchDequantizeNvFp4(void* b_dequant,
                             const void* b_packed,
                             const void* weight_scale,
                             const float* weight_scale_2,
                             int n,
                             int k,
                             int block_size,
                             bool is_bf16,
                             cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  const int half_k = k >> 1;
  const long long total = static_cast<long long>(n) * half_k;
  if (total == 0) {
    return Status::OK();
  }
  const int k_blocks = (k + block_size - 1) / block_size;
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const uint8_t* bp = reinterpret_cast<const uint8_t*>(b_packed);
  const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);

  if (is_bf16) {
    DequantizeNvFp4Kernel<nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(b_dequant), bp, ws, weight_scale_2, n, k, k_blocks, block_size);
  } else {
    DequantizeNvFp4Kernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(b_dequant), bp, ws, weight_scale_2, n, k, k_blocks, block_size);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(b_dequant);
  ORT_UNUSED_PARAMETER(b_packed);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(weight_scale_2);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp4Weight requires CUDA 12.8 or newer for NVFP4 support.");
#endif
}

Status LaunchAddBiasNvFp4(void* y,
                          const void* bias,
                          int m,
                          int n,
                          bool is_bf16,
                          cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  const long long total = static_cast<long long>(m) * n;
  if (total == 0) {
    return Status::OK();
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    AddBiasKernel<nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(y), reinterpret_cast<const nv_bfloat16*>(bias), m, n);
  } else {
    AddBiasKernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(y), reinterpret_cast<const half*>(bias), m, n);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp4Weight requires CUDA 12.8 or newer for NVFP4 support.");
#endif
}

Status LaunchMatMulBlockQuantizedFp4WeightGemv(void* y,
                                               const void* a,
                                               const void* b_packed,
                                               const void* weight_scale,
                                               const float* weight_scale_2,
                                               const void* bias,
                                               int m,
                                               int n,
                                               int k,
                                               int block_size,
                                               bool is_bf16,
                                               const cudaDeviceProp& device_prop,
                                               cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  if (m <= 0 || n <= 0 || k <= 0) {
    return Status::OK();
  }
  // This kernel is hard-coded for block_size == 16 and assumes K is a multiple of 32 so that each
  // warp lane always owns a full 32-element slice (and one E4M3 scale per 16-element block). Guard
  // against misuse if this helper is ever reused outside the callers that already check these.
  ORT_RETURN_IF_NOT(block_size == 16, "MatMulBlockQuantizedFp4Weight GEMV requires block_size == 16, got ", block_size, ".");
  ORT_RETURN_IF_NOT(k % 32 == 0, "MatMulBlockQuantizedFp4Weight GEMV requires K divisible by 32, got ", k, ".");
  const int k_blocks = (k + block_size - 1) / block_size;

  // Tensor-core sub-path: needs mma.m16n8k16 (SM80+), a whole number of 128-element K windows,
  // and M within the mma's 8-row N extent. Everything else falls back to the scalar GEMV.
  //
  // The M <= 8 bound is structural: one warp owns a 16-column x 8-row output tile, where lane
  // (g = lane >> 2, t = lane & 3) reads activation row g, covers weight columns 16 * tile + g and
  // + 8, and stores output rows 2t and 2t + 1 of those two columns.
  if (device_prop.major >= 8 && k % 128 == 0 && m <= 8 && Fp4GemvMmaEnabled()) {
    const Fp4MmaConfig cfg = PickFp4MmaConfig(n, k, device_prop.multiProcessorCount);
    const int cols_per_block = 16 * cfg.col_tiles;
    const dim3 mma_threads{32, static_cast<unsigned int>(cfg.k_split * cfg.col_tiles)};
    const dim3 mma_blocks{static_cast<unsigned int>((n + cols_per_block - 1) / cols_per_block)};
    const uint8_t* mbp = reinterpret_cast<const uint8_t*>(b_packed);
    const uint8_t* mws = reinterpret_cast<const uint8_t*>(weight_scale);

#define ORT_LAUNCH_FP4_MMA_GEMV(T, KS, CT)                                                       \
  MatMulBlockQuantizedFp4WeightMmaGemvKernel<T, KS, CT>                                          \
      <<<mma_blocks, mma_threads, 0, stream>>>(reinterpret_cast<T*>(y),                          \
                                               reinterpret_cast<const T*>(a), mbp, mws,          \
                                               weight_scale_2, reinterpret_cast<const T*>(bias), \
                                               m, n, k, k_blocks)

#define ORT_DISPATCH_FP4_MMA_GEMV(T)         \
  do {                                       \
    if (cfg.col_tiles == 4) {                \
      if (cfg.k_split >= 2) {                \
        ORT_LAUNCH_FP4_MMA_GEMV(T, 2, 4);    \
      } else {                               \
        ORT_LAUNCH_FP4_MMA_GEMV(T, 1, 4);    \
      }                                      \
    } else {                                 \
      switch (cfg.k_split) {                 \
        case 16:                             \
          ORT_LAUNCH_FP4_MMA_GEMV(T, 16, 1); \
          break;                             \
        case 8:                              \
          ORT_LAUNCH_FP4_MMA_GEMV(T, 8, 1);  \
          break;                             \
        case 4:                              \
          ORT_LAUNCH_FP4_MMA_GEMV(T, 4, 1);  \
          break;                             \
        case 2:                              \
          ORT_LAUNCH_FP4_MMA_GEMV(T, 2, 1);  \
          break;                             \
        default:                             \
          ORT_LAUNCH_FP4_MMA_GEMV(T, 1, 1);  \
          break;                             \
      }                                      \
    }                                        \
  } while (0)

    if (is_bf16) {
      ORT_DISPATCH_FP4_MMA_GEMV(nv_bfloat16);
    } else {
      ORT_DISPATCH_FP4_MMA_GEMV(half);
    }
#undef ORT_DISPATCH_FP4_MMA_GEMV
#undef ORT_LAUNCH_FP4_MMA_GEMV
    return CUDA_CALL(cudaGetLastError());
  }

  const int rows_per_block = Fp4GemvRowsPerBlock(m, n, device_prop.multiProcessorCount);
  const dim3 threads{32, kGemvWarpsPerBlock};
  const dim3 blocks{static_cast<unsigned int>((n + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock),
                    static_cast<unsigned int>((m + rows_per_block - 1) / rows_per_block)};
  const uint8_t* bp = reinterpret_cast<const uint8_t*>(b_packed);
  const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);

#define ORT_DISPATCH_FP4_GEMV(T)                                                            \
  do {                                                                                      \
    switch (rows_per_block) {                                                               \
      case 4:                                                                               \
        MatMulBlockQuantizedFp4WeightGemvKernel<T, 4><<<blocks, threads, 0, stream>>>(      \
            reinterpret_cast<T*>(y), reinterpret_cast<const T*>(a), bp, ws, weight_scale_2, \
            reinterpret_cast<const T*>(bias), m, n, k, k_blocks);                           \
        break;                                                                              \
      case 2:                                                                               \
        MatMulBlockQuantizedFp4WeightGemvKernel<T, 2><<<blocks, threads, 0, stream>>>(      \
            reinterpret_cast<T*>(y), reinterpret_cast<const T*>(a), bp, ws, weight_scale_2, \
            reinterpret_cast<const T*>(bias), m, n, k, k_blocks);                           \
        break;                                                                              \
      default:                                                                              \
        MatMulBlockQuantizedFp4WeightGemvKernel<T, 1><<<blocks, threads, 0, stream>>>(      \
            reinterpret_cast<T*>(y), reinterpret_cast<const T*>(a), bp, ws, weight_scale_2, \
            reinterpret_cast<const T*>(bias), m, n, k, k_blocks);                           \
        break;                                                                              \
    }                                                                                       \
  } while (0)

  if (is_bf16) {
    ORT_DISPATCH_FP4_GEMV(nv_bfloat16);
  } else {
    ORT_DISPATCH_FP4_GEMV(half);
  }
#undef ORT_DISPATCH_FP4_GEMV
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(a);
  ORT_UNUSED_PARAMETER(b_packed);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(weight_scale_2);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(device_prop);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantizedFp4Weight requires CUDA 12.8 or newer for NVFP4 support.");
#endif
}

}  // namespace onnxruntime::contrib::cuda
