// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <type_traits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/quantization/matmul_4bits_batched.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_common.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_m1.cuh"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"

// Include env_var_utils.h after cuda_common.h: the latter transitively pulls in provider_api.h,
// which defines SHARED_PROVIDER. That guard suppresses env_var_utils.h's own logging.h include and
// avoids redefining CREATE_MESSAGE/LOGS_CATEGORY in this provider-bridge translation unit.
#include "core/platform/env_var_utils.h"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// The specialized router GEMV kernel only handles M=1, or batch size 1.
constexpr int kRouterM = 1;

// MoE router shape (N = number of experts, K = hidden size). The specialization is exact-shape
// gated to the GPT-OSS-20B router projection to keep the dispatch change conservative.
constexpr int kGptOssRouterN = 32;
constexpr int kGptOssRouterK = 2880;

static bool IsRouterGemvSpecializationDisabled() {
  // Use ORT's cross-platform env var helper instead of std::getenv, which is unsafe on Windows.
  return ParseEnvironmentVariableWithDefault<bool>("ORT_DISABLE_QMOE_ROUTER_GEMV_SPECIALIZATION", false);
}

// The router GEMV kernel handles any symmetric (no zero point) M=1 shape with an int4 group size of
// 32 or 64 (whichever quantizes best) and N divisible by kColsPerThreadBlock. We gate on the exact
// GPT-OSS-20B router shape to avoid changing the dispatch for general MatMulNBits cases. K must be a
// multiple of the group size (always true for a router) and N a multiple of kColsPerThreadBlock
// (checked in TryMatMul4Bits). Note kPerIter (256) is divisible by both 32 and 64, so the scale
// stride is exact.
static bool IsSupportedRouterGemvShape(const uint8_t* zero_points, int m, int n, int k, int block_size) {
  if (zero_points != nullptr || m != kRouterM || (block_size != 32 && block_size != 64)) {
    return false;
  }
  if (k % block_size != 0) {
    return false;
  }
  return n == kGptOssRouterN && k == kGptOssRouterK;  // gpt-oss-20b
}

// Reduces kUnroll groups of kPerIter elements per step, advancing the packed-weight pointer, scale
// index and k position in lockstep. Factored out of MatMulFloatInt4RouterKernel so the three unroll
// factors (16, 4, 1) share one implementation instead of a macro.
template <class T, int BlockSize, int kUnroll>
__device__ __forceinline__ void RouterUnrollReduction(
    const uint8_t*& b_data_quant,
    const T* scales_data,
    const T* a_data,
    int k,
    int& k_id,
    int& scale_id,
    T (&sums)[8]) {
  constexpr int kPerIter = kWarpSize * kElementsPerThreadPerIteration;
  constexpr int kUnrollStep = kUnroll * kPerIter;
  const int k_unroll_bound = k - k % kUnrollStep;
  for (; k_id < k_unroll_bound; k_id += kUnrollStep) {
#pragma unroll
    for (int i = 0; i < kUnroll; i++) {
      uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + kPerIter / 2 * i));
      T scale = scales_data[scale_id + kPerIter / BlockSize * i];
      AccumulateEightElements4b(value, scale, 8, a_data + k_id + i * kPerIter, sums);
    }
    b_data_quant += kPerIter / 2 * kUnroll;
    scale_id += kPerIter / BlockSize * kUnroll;
  }
}

// GEMV specialization for MoE routers: output(1, N) = a(1, K) x dequant(B(N, K)) [+ bias(N)].
// B is 4-bit block-quantized (symmetric, no zero point) with group size BlockSize (32 or 64). One warp
// computes one expert column; the thread block holds kColsPerThreadBlock warps. N is passed via the
// grid and K at runtime, so a single instantiation per (T, BlockSize) serves every router shape.
// Requirements (satisfied by the dispatch in TryMatMul4Bits): N % kColsPerThreadBlock == 0,
// K % BlockSize == 0, and kPerIter % BlockSize == 0 so the per-iteration scale stride is exact.
template <class T, int BlockSize>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloatInt4RouterKernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const T* bias_data,
    int n,
    int k) {
  constexpr int kPerIter = kWarpSize * kElementsPerThreadPerIteration;
  static_assert(kPerIter % BlockSize == 0, "kPerIter must be a multiple of BlockSize for exact scale stride");

  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = blockIdx.x * kColsPerThreadBlock + warp_id;

  // Each column occupies k/2 bytes of packed 4-bit weights and k/BlockSize scales.
  a_data += lane_id << 3;
  b_data_quant += static_cast<size_t>(n_id) * (k / 2) + lane_id * 4;
  scales_data += static_cast<size_t>(n_id) * (k / BlockSize);

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int scale_id = lane_id * 8 / BlockSize;

  RouterUnrollReduction<T, BlockSize, 16>(b_data_quant, scales_data, a_data, k, k_id, scale_id, sums);
  RouterUnrollReduction<T, BlockSize, 4>(b_data_quant, scales_data, a_data, k, k_id, scale_id, sums);
  RouterUnrollReduction<T, BlockSize, 1>(b_data_quant, scales_data, a_data, k, k_id, scale_id, sums);

  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = scales_data[scale_id];
    AccumulateEightElements4b(value, scale, 8, a_data + k_id, sums);
  }

  float sum = static_cast<float>(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  for (int i = kWarpSize / 2; i > 0; i = i / 2) {
    sum += WARP_SHFL_DOWN(sum, i);
  }

  if (lane_id == 0) {
    if (bias_data != nullptr) {
      sum += static_cast<float>(bias_data[n_id]);
    }
    output[n_id] = sum;
  }
}

// ===== Small-M batched GEMV (short prefill / small batch) =====
// The single-row M1 kernel launches one block per output row (grid.y = m), so each row
// independently re-reads and re-dequantizes all of B; weight traffic and dequant work scale with M.
// For 2 <= M <= cap we instead dequantize each packed weight word once and accumulate it against
// CtaM activation rows held in registers, cutting weight traffic to ceil(M/CtaM)x. This is the same
// design used by TensorRT-LLM weightOnlyBatchedGemv / AWQ / llama.cpp MMVQ for small batch.
//
// Upper bound on M is kSmallMMax for all dtypes (measured on A100 vs the dequantize+cuBLAS fallback).
// half/bf16 run the register-tiled batched kernel, which stays ahead of the tensor-core GEMM through
// M<=16. float has no tensor-core GEMM fallback, so it uses the shared-memory small-M kernel over the
// same range.
constexpr int kSmallMMax = 16;
template <class T>
__host__ __device__ constexpr int SmallMCap() {
  return kSmallMMax;
}

// Holds the 8 dequantized weights produced from one packed 4-bit word, in the layout each
// AccumulateRow expects for its dtype. Splitting dequantization from the per-row multiply lets one
// dequantized weight feed all CtaM rows.
template <class T>
struct DequantizedEight;

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
template <>
struct DequantizedEight<half> {
  half2 v[4];  // order [04, 15, 26, 37], matching Convert8xInt4To8xHalfs + the prmt below
};
__device__ __forceinline__ void DequantizeEight(uint32_t values_quant, half scale, uint8_t zp, DequantizedEight<half>& d) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  half2 elements[4];
  Convert8xInt4To8xHalfs(values_quant, elements);
  d.v[0] = elements[0] * scale_half2 + zp_adjust2;
  d.v[1] = elements[1] * scale_half2 + zp_adjust2;
  d.v[2] = elements[2] * scale_half2 + zp_adjust2;
  d.v[3] = elements[3] * scale_half2 + zp_adjust2;
}
__device__ __forceinline__ void AccumulateRow(const DequantizedEight<half>& d, const half* a, half* sums) {
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));
  constexpr uint32_t kLowHalf2 = 0x5410;
  constexpr uint32_t kHighHalf2 = 0x7632;
  uint4 vp;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vp.x) : "r"(vec_a.x), "r"(vec_a.z), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vp.y) : "r"(vec_a.x), "r"(vec_a.z), "r"(kHighHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vp.z) : "r"(vec_a.y), "r"(vec_a.w), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vp.w) : "r"(vec_a.y), "r"(vec_a.w), "r"(kHighHalf2));
  half2* s = reinterpret_cast<half2*>(sums);
  s[0] = s[0] + d.v[0] * (*(reinterpret_cast<half2*>(&vp.x)));
  s[1] = s[1] + d.v[1] * (*(reinterpret_cast<half2*>(&vp.y)));
  s[2] = s[2] + d.v[2] * (*(reinterpret_cast<half2*>(&vp.z)));
  s[3] = s[3] + d.v[3] * (*(reinterpret_cast<half2*>(&vp.w)));
}
#else
template <>
struct DequantizedEight<half> {
  half v[8];
};
__device__ __forceinline__ void DequantizeEight(uint32_t values_quant, half scale, uint8_t zp, DequantizedEight<half>& d) {
  half zp_adjust = -scale * __short2half_rn(zp);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    d.v[i] = __uint2half_rn((values_quant >> (4 * i)) & 0xF) * scale + zp_adjust;
  }
}
__device__ __forceinline__ void AccumulateRow(const DequantizedEight<half>& d, const half* a, half* sums) {
#pragma unroll
  for (int i = 0; i < 8; i++) {
    sums[i] += d.v[i] * a[i];
  }
}
#endif

template <>
struct DequantizedEight<float> {
  float v[8];
};
__device__ __forceinline__ void DequantizeEight(uint32_t values_quant, float scale, uint8_t zp, DequantizedEight<float>& d) {
  float zp_adjust = -scale * zp;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    d.v[i] = float((values_quant >> (4 * i)) & 0xF) * scale + zp_adjust;
  }
}
__device__ __forceinline__ void AccumulateRow(const DequantizedEight<float>& d, const float* a, float* sums) {
  float4 a0 = *(reinterpret_cast<const float4*>(a));
  float4 a1 = *(reinterpret_cast<const float4*>(a + 4));
  sums[0] += d.v[0] * a0.x;
  sums[1] += d.v[1] * a0.y;
  sums[2] += d.v[2] * a0.z;
  sums[3] += d.v[3] * a0.w;
  sums[4] += d.v[4] * a1.x;
  sums[5] += d.v[5] * a1.y;
  sums[6] += d.v[6] * a1.z;
  sums[7] += d.v[7] * a1.w;
}

template <>
struct DequantizedEight<nv_bfloat16> {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat162 v[4];
#else
  nv_bfloat16 v[8];
#endif
};
__device__ __forceinline__ void DequantizeEight(uint32_t values_quant, nv_bfloat16 scale, uint8_t zp, DequantizedEight<nv_bfloat16>& d) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat162 scale_bf162 = __bfloat162bfloat162(scale);
  nv_bfloat16 zp_adjust = -scale * __uint2bfloat16_rn(zp);
  __nv_bfloat162 zp_adjust2 = __bfloat162bfloat162(zp_adjust);
  __nv_bfloat162 elements[4];
  Convert8xInt4To8xBF16s(values_quant, elements);
  d.v[0] = __hfma2(elements[0], scale_bf162, zp_adjust2);
  d.v[1] = __hfma2(elements[1], scale_bf162, zp_adjust2);
  d.v[2] = __hfma2(elements[2], scale_bf162, zp_adjust2);
  d.v[3] = __hfma2(elements[3], scale_bf162, zp_adjust2);
#endif
}
__device__ __forceinline__ void AccumulateRow(const DequantizedEight<nv_bfloat16>& d, const nv_bfloat16* a, nv_bfloat16* sums) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));
  constexpr uint32_t kLowHalf2 = 0x5410;
  constexpr uint32_t kHighHalf2 = 0x7632;
  uint4 vp;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vp.x) : "r"(vec_a.x), "r"(vec_a.z), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vp.y) : "r"(vec_a.x), "r"(vec_a.z), "r"(kHighHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vp.z) : "r"(vec_a.y), "r"(vec_a.w), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vp.w) : "r"(vec_a.y), "r"(vec_a.w), "r"(kHighHalf2));
  __nv_bfloat162* s = reinterpret_cast<__nv_bfloat162*>(sums);
  s[0] = __hfma2(d.v[0], *reinterpret_cast<__nv_bfloat162*>(&vp.x), s[0]);
  s[1] = __hfma2(d.v[1], *reinterpret_cast<__nv_bfloat162*>(&vp.y), s[1]);
  s[2] = __hfma2(d.v[2], *reinterpret_cast<__nv_bfloat162*>(&vp.z), s[2]);
  s[3] = __hfma2(d.v[3], *reinterpret_cast<__nv_bfloat162*>(&vp.w), s[3]);
#endif
}

// ---- Small-M batched GEMV (half/bf16): CtaM x CtaN register tiling --------------------------
// 2-wide accumulator type (half2 / bf162).
template <class T>
struct Acc2;
template <>
struct Acc2<half> {
  using type = half2;
};
template <>
struct Acc2<nv_bfloat16> {
  using type = __nv_bfloat162;
};

// Four 2-wide weight lanes in NATURAL element order [01,23,45,67], so a naturally-loaded activation
// (uint4 reinterpreted as four half2) can be multiply-accumulated with no per-activation permute.
template <class T>
struct WPack {
  typename Acc2<T>::type v[4];
};

// DequantizeEight emits [04,15,26,37] (the order of Convert8xInt4To8xHalfs); repack to natural order
// once per column. Doing the prmt on the (CtaN) weights instead of the (CtaM) activations cuts the
// permute count by CtaM/CtaN, which dominates at small M. The signatures are always defined so
// MatMulFloat4BatchedKernel<half> still compiles for archs below sm_53 (e.g. sm_52); only the
// half2-intrinsic bodies are gated, mirroring the nv_bfloat16 helpers below.
__device__ __forceinline__ WPack<half> PackNatural(const DequantizedEight<half>& d) {
  WPack<half> w;
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
  uint32_t d0 = *reinterpret_cast<const uint32_t*>(&d.v[0]);
  uint32_t d1 = *reinterpret_cast<const uint32_t*>(&d.v[1]);
  uint32_t d2 = *reinterpret_cast<const uint32_t*>(&d.v[2]);
  uint32_t d3 = *reinterpret_cast<const uint32_t*>(&d.v[3]);
  constexpr uint32_t kLo = 0x5410;  // (x0,x1) of two half2 -> elements 0,1
  constexpr uint32_t kHi = 0x7632;  // (y0,y1) -> elements 4,5
  uint32_t t;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t) : "r"(d0), "r"(d1), "r"(kLo));
  w.v[0] = *reinterpret_cast<half2*>(&t);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t) : "r"(d2), "r"(d3), "r"(kLo));
  w.v[1] = *reinterpret_cast<half2*>(&t);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t) : "r"(d0), "r"(d1), "r"(kHi));
  w.v[2] = *reinterpret_cast<half2*>(&t);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t) : "r"(d2), "r"(d3), "r"(kHi));
  w.v[3] = *reinterpret_cast<half2*>(&t);
#endif
  return w;
}
__device__ __forceinline__ void DotAccum(const WPack<half>& w, const half2* a4, half2& acc) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
  acc = __hfma2(w.v[0], a4[0], acc);
  acc = __hfma2(w.v[1], a4[1], acc);
  acc = __hfma2(w.v[2], a4[2], acc);
  acc = __hfma2(w.v[3], a4[3], acc);
#endif
}
__device__ __forceinline__ float HorizontalAdd(half2 acc) {
#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
  return static_cast<float>(acc.x) + static_cast<float>(acc.y);
#else
  return 0.f;
#endif
}

__device__ __forceinline__ WPack<nv_bfloat16> PackNatural(const DequantizedEight<nv_bfloat16>& d) {
  WPack<nv_bfloat16> w;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  uint32_t d0 = *reinterpret_cast<const uint32_t*>(&d.v[0]);
  uint32_t d1 = *reinterpret_cast<const uint32_t*>(&d.v[1]);
  uint32_t d2 = *reinterpret_cast<const uint32_t*>(&d.v[2]);
  uint32_t d3 = *reinterpret_cast<const uint32_t*>(&d.v[3]);
  constexpr uint32_t kLo = 0x5410;
  constexpr uint32_t kHi = 0x7632;
  uint32_t t;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t) : "r"(d0), "r"(d1), "r"(kLo));
  w.v[0] = *reinterpret_cast<__nv_bfloat162*>(&t);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t) : "r"(d2), "r"(d3), "r"(kLo));
  w.v[1] = *reinterpret_cast<__nv_bfloat162*>(&t);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t) : "r"(d0), "r"(d1), "r"(kHi));
  w.v[2] = *reinterpret_cast<__nv_bfloat162*>(&t);
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(t) : "r"(d2), "r"(d3), "r"(kHi));
  w.v[3] = *reinterpret_cast<__nv_bfloat162*>(&t);
#endif
  return w;
}
__device__ __forceinline__ void DotAccum(const WPack<nv_bfloat16>& w, const __nv_bfloat162* a4, __nv_bfloat162& acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  acc = __hfma2(w.v[0], a4[0], acc);
  acc = __hfma2(w.v[1], a4[1], acc);
  acc = __hfma2(w.v[2], a4[2], acc);
  acc = __hfma2(w.v[3], a4[3], acc);
#endif
}
__device__ __forceinline__ float HorizontalAdd(__nv_bfloat162 acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return static_cast<float>(acc.x) + static_cast<float>(acc.y);
#else
  return 0.f;
#endif
}

// Each warp computes CtaN output columns x CtaM rows. Lanes split K (8 elems/lane/iter); per-row
// activations are loaded once (uint4) and reused across CtaN columns, and the int4->half order permute
// is applied once per column weight (not per row). A single 2-wide accumulator per (row,column) keeps
// CtaM=m up to 16 in registers with the weight streamed exactly once. The launch bound pins >=3 blocks
// per SM so the CtaN=4 tiling (which minimizes activation L2 traffic) keeps enough occupancy to hide
// memory latency. Standard MatMulNBits [N, blocks, blob] layout, no prepacking; scales/zp from global.
template <class T, int block_size, bool has_zero_point, int CtaM, int CtaN>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock, 3) MatMulFloat4BatchedKernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  using AccT = typename Acc2<T>::type;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int col_base = (blockIdx.x * kColsPerThreadBlock + warp_id) * CtaN;
  const int m_base = blockIdx.y * CtaM;
  const int valid = m - m_base;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;  // 256
  const int zp_blocks = (blocks_per_K + 1) / 2;

  const T* a_base = a_data + static_cast<size_t>(m_base) * k + (lane_id << 3);
  const uint8_t* b_ptr[CtaN];
#pragma unroll
  for (int c = 0; c < CtaN; c++) {
    b_ptr[c] = b_data_quant + static_cast<size_t>(col_base + c) * blocks_per_K * (block_size / 2) + lane_id * 4;
  }

  AccT acc[CtaM][CtaN];
#pragma unroll
  for (int r = 0; r < CtaM; r++) {
#pragma unroll
    for (int c = 0; c < CtaN; c++) {
      acc[r][c] = AccT{};
    }
  }

  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  constexpr int kWork = CtaM * CtaN;
  constexpr int kMainUnroll = (kWork >= 20) ? 1 : (kWork >= 12) ? 2
                                                                : 4;

#define BATCHED_BODY(i)                                                                       \
  do {                                                                                        \
    WPack<T> w[CtaN];                                                                         \
    const int bk = t_meta_k + k_per_iter / block_size * (i);                                  \
    _Pragma("unroll") for (int c = 0; c < CtaN; c++) {                                        \
      uint32_t value = *(reinterpret_cast<const uint32_t*>(b_ptr[c] + k_per_iter / 2 * (i))); \
      T scale = scales_data[static_cast<size_t>(col_base + c) * blocks_per_K + bk];           \
      uint8_t zp = 8;                                                                         \
      if constexpr (has_zero_point) {                                                         \
        uint8_t zpb = zero_points[static_cast<size_t>(col_base + c) * zp_blocks + (bk >> 1)]; \
        zp = (bk & 1) ? (zpb >> 4) : (zpb & 0x0f);                                            \
      }                                                                                       \
      DequantizedEight<T> d;                                                                  \
      DequantizeEight(value, scale, zp, d);                                                   \
      w[c] = PackNatural(d);                                                                  \
    }                                                                                         \
    _Pragma("unroll") for (int r = 0; r < CtaM; r++) {                                        \
      if (r < valid) {                                                                        \
        AccT a4[4];                                                                           \
        *reinterpret_cast<uint4*>(a4) = *reinterpret_cast<const uint4*>(                      \
            a_base + static_cast<size_t>(r) * k + k_id + (i) * k_per_iter);                   \
        _Pragma("unroll") for (int c = 0; c < CtaN; c++) {                                    \
          DotAccum(w[c], a4, acc[r][c]);                                                      \
        }                                                                                     \
      }                                                                                       \
    }                                                                                         \
  } while (false)

#define BATCHED_UNROLL(unroll_size)                         \
  do {                                                      \
    constexpr int kUnroll = unroll_size;                    \
    constexpr int kUnrollStep = kUnroll * k_per_iter;       \
    const int k_unroll_bound = k - k % kUnrollStep;         \
    for (; k_id < k_unroll_bound; k_id += kUnrollStep) {    \
      _Pragma("unroll") for (int i = 0; i < kUnroll; i++) { \
        BATCHED_BODY(i);                                    \
      }                                                     \
      _Pragma("unroll") for (int c = 0; c < CtaN; c++) {    \
        b_ptr[c] += k_per_iter / 2 * kUnroll;               \
      }                                                     \
      t_meta_k += k_per_iter / block_size * kUnroll;        \
    }                                                       \
  } while (false)

  BATCHED_UNROLL(kMainUnroll);
  BATCHED_UNROLL(1);
#undef BATCHED_UNROLL

  if (k_id + lane_id * 8 < k) {
    WPack<T> w[CtaN];
    const int bk = t_meta_k;
#pragma unroll
    for (int c = 0; c < CtaN; c++) {
      uint32_t value = *(reinterpret_cast<const uint32_t*>(b_ptr[c]));
      T scale = scales_data[static_cast<size_t>(col_base + c) * blocks_per_K + bk];
      uint8_t zp = 8;
      if constexpr (has_zero_point) {
        uint8_t zpb = zero_points[static_cast<size_t>(col_base + c) * zp_blocks + (bk >> 1)];
        zp = (bk & 1) ? (zpb >> 4) : (zpb & 0x0f);
      }
      DequantizedEight<T> d;
      DequantizeEight(value, scale, zp, d);
      w[c] = PackNatural(d);
    }
#pragma unroll
    for (int r = 0; r < CtaM; r++) {
      if (r < valid) {
        AccT a4[4];
        *reinterpret_cast<uint4*>(a4) = *reinterpret_cast<const uint4*>(a_base + static_cast<size_t>(r) * k + k_id);
#pragma unroll
        for (int c = 0; c < CtaN; c++) {
          DotAccum(w[c], a4, acc[r][c]);
        }
      }
    }
  }
#undef BATCHED_BODY

#pragma unroll
  for (int r = 0; r < CtaM; r++) {
    if (r >= valid) continue;
#pragma unroll
    for (int c = 0; c < CtaN; c++) {
      float sum = HorizontalAdd(acc[r][c]);
      for (int i = kWarpSize / 2; i > 0; i = i / 2) {
        sum += WARP_SHFL_DOWN(sum, i);
      }
      if (lane_id == 0) {
        output[static_cast<size_t>(m_base + r) * n + (col_base + c)] = static_cast<T>(sum);
      }
    }
  }
}

// Batched GEMV: block computes CtaM rows x kColsPerThreadBlock columns. Grid is
// (ceil(N/kColsPerThreadBlock), ceil(M/CtaM)). It uses shared-memory scale/zp staging and
// packed-weight indexing while looping over CtaM rows per dequantized word.
template <class T, int block_size, bool has_zero_point, int CtaM>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloatInt4KernelSmallM(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int n_block_id = blockIdx.x;
  const int m_base = blockIdx.y * CtaM;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = n_block_id * kColsPerThreadBlock + warp_id;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

  extern __shared__ char shared_buffer[];
  T* b_scale_vec = (T*)shared_buffer;
  int offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
  for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  uint8_t* b_zp_vec;
  (void)b_zp_vec;
  if constexpr (has_zero_point) {
    b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + kColsPerThreadBlock * blocks_per_K);
    const int b_zp_k = (blocks_per_K + 1) / 2;
    int zp_offset = n_block_id * kColsPerThreadBlock * b_zp_k;
    for (int i = warp_id * kWarpSize + lane_id; i < kColsPerThreadBlock * b_zp_k; i += kColsPerThreadBlock * kWarpSize) {
      b_zp_vec[2 * i] = (zero_points[zp_offset + i] & 0x0f);
      b_zp_vec[2 * i + 1] = (zero_points[zp_offset + i] >> 4);
    }
    b_zp_vec += warp_id * b_zp_k * 2;
  }
  __syncthreads();

  const int valid = m - m_base;
  const T* a_row[CtaM];
#pragma unroll
  for (int r = 0; r < CtaM; r++) {
    a_row[r] = a_data + static_cast<size_t>(m_base + r) * k + (lane_id << 3);
  }
  b_scale_vec += warp_id * blocks_per_K;

  T sums[CtaM][8];
#pragma unroll
  for (int r = 0; r < CtaM; r++) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      sums[r][j] = static_cast<T>(0);
    }
  }

  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

#define SmallMUnRoll(unroll_size)                                                                 \
  do {                                                                                            \
    constexpr int kUnroll = unroll_size;                                                          \
    constexpr int kUnrollStep = kUnroll * k_per_iter;                                             \
    const int k_unroll_bound = k - k % kUnrollStep;                                               \
    for (; k_id < k_unroll_bound; k_id += kUnrollStep) {                                          \
      _Pragma("unroll") for (int i = 0; i < kUnroll; i++) {                                       \
        uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_per_iter / 2 * i)); \
        T scale = b_scale_vec[t_meta_k + k_per_iter / block_size * i];                            \
        uint8_t zp = 8;                                                                           \
        if constexpr (has_zero_point) {                                                           \
          zp = b_zp_vec[t_meta_k + k_per_iter / block_size * i];                                  \
        }                                                                                         \
        DequantizedEight<T> d;                                                                    \
        DequantizeEight(value, scale, zp, d);                                                     \
        _Pragma("unroll") for (int r = 0; r < CtaM; r++) {                                        \
          if (r < valid) AccumulateRow(d, a_row[r] + k_id + i * k_per_iter, sums[r]);             \
        }                                                                                         \
      }                                                                                           \
      b_data_quant += k_per_iter / 2 * kUnroll;                                                   \
      t_meta_k += k_per_iter / block_size * kUnroll;                                              \
    }                                                                                             \
  } while (false)

  SmallMUnRoll(16);
  SmallMUnRoll(4);
  SmallMUnRoll(1);
#undef SmallMUnRoll

  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[t_meta_k];
    uint8_t zp = 8;
    if constexpr (has_zero_point) {
      zp = b_zp_vec[t_meta_k];
    }
    DequantizedEight<T> d;
    DequantizeEight(value, scale, zp, d);
#pragma unroll
    for (int r = 0; r < CtaM; r++) {
      if (r < valid) AccumulateRow(d, a_row[r] + k_id, sums[r]);
    }
  }

#pragma unroll
  for (int r = 0; r < CtaM; r++) {
    if (r >= valid) continue;
    float sum = static_cast<float>(sums[r][0] + sums[r][1] + sums[r][2] + sums[r][3] +
                                   sums[r][4] + sums[r][5] + sums[r][6] + sums[r][7]);
    for (int i = kWarpSize / 2; i > 0; i = i / 2) {
      sum += WARP_SHFL_DOWN(sum, i);
    }
    if (lane_id == 0) {
      output[static_cast<size_t>(m_base + r) * n + n_id] = sum;
    }
  }
}

// Launches the batched small-M kernel for 2 <= m <= SmallMCap<T>(). Returns false if m is out of range
// so the caller can fall back to the single-row (m==1) or dequant+GEMM (large m) paths.
template <class T>
bool TryMatMulSmallM4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_size,
    cudaStream_t stream) {
  if (m < 2 || m > SmallMCap<T>()) {
    return false;
  }
  const int cta_m = (m <= 2) ? 2 : 4;
  dim3 threads(GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, (m + cta_m - 1) / cta_m);

#define SmallMDispatch(BS, CM)                                                                   \
  if (nullptr != zero_points) {                                                                  \
    MatMulFloatInt4KernelSmallM<T, BS, true, CM><<<blocks, threads, shared_mem_size, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, (k + BS - 1) / BS);     \
  } else {                                                                                       \
    MatMulFloatInt4KernelSmallM<T, BS, false, CM><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, (k + BS - 1) / BS);     \
  }
#define SmallMDispatchBlock(CM)   \
  if (16 == block_size) {         \
    SmallMDispatch(16, CM)        \
  } else if (32 == block_size) {  \
    SmallMDispatch(32, CM)        \
  } else if (64 == block_size) {  \
    SmallMDispatch(64, CM)        \
  } else if (128 == block_size) { \
    SmallMDispatch(128, CM)       \
  } else {                        \
    return false;                 \
  }

  if (cta_m == 2) {
    SmallMDispatchBlock(2)
  } else {
    SmallMDispatchBlock(4)
  }

#undef SmallMDispatchBlock
#undef SmallMDispatch
  return true;
}

template <class T>
bool TryMatMul4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    const T* bias_data,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    int sm_count,
    cudaStream_t stream) {
  if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > SmallMCap<T>()) {
    return false;
  }

  if (IsSupportedRouterGemvShape(zero_points, m, n, k, block_size) &&
      !IsRouterGemvSpecializationDisabled()) {
    const dim3 blocks(n / kColsPerThreadBlock, 1);
    const dim3 threads(GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
    if (block_size == 32) {
      MatMulFloatInt4RouterKernel<T, 32><<<blocks, threads, 0, stream>>>(
          output, a_data, b_data_quant, scales_data, bias_data, n, k);
    } else {
      MatMulFloatInt4RouterKernel<T, 64><<<blocks, threads, 0, stream>>>(
          output, a_data, b_data_quant, scales_data, bias_data, n, k);
    }
    return true;
  }

  if (bias_data != nullptr) {
    return false;
  }

  // The register-tiled batched path (half/bf16, 2 <= m <= cap) launches with no shared memory, so try
  // it before the shared-memory budget gate that only constrains the shared-memory kernels below.
  if constexpr (!std::is_same<T, float>::value) {
    if (m >= 2) {
      if (TryMatMulBatched4Bits<T>(output, a_data, b_data_quant, scales_data, zero_points,
                                   m, n, k, block_size, 0, stream)) {
        return true;
      }
    }
  }

  // Float, and any half/bf16 shape the batched path rejected, falls back to the shared-memory small-M
  // kernel. m == 1 uses the single-row kernel below; larger m used the dequantize + cuBLAS path
  // (returned false above).
  if (m >= 2) {
    int blocks_per_K = (k + block_size - 1) / block_size;
    size_t shared_mem_size = sizeof(T) * blocks_per_K * kColsPerThreadBlock +
                             static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
    if (shared_mem_size > shared_mem_per_block) {
      return false;
    }
    return TryMatMulSmallM4Bits<T>(output, a_data, b_data_quant, scales_data, zero_points,
                                   m, n, k, block_size, shared_mem_size, stream);
  }

  return TryMatMul4BitsM1<T>(output, a_data, b_data_quant, scales_data, zero_points,
                             n, k, block_size, shared_mem_per_block, sm_count, stream);
}

template bool TryMatMul4Bits<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    const float* bias_data,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    int sm_count,
    cudaStream_t stream);

template bool TryMatMul4Bits<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    const half* bias_data,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    int sm_count,
    cudaStream_t stream);

template bool TryMatMul4Bits<nv_bfloat16>(
    nv_bfloat16* output,
    const nv_bfloat16* a_data,
    const uint8_t* b_data_quant,
    const nv_bfloat16* scales_data,
    const uint8_t* zero_points,
    const nv_bfloat16* bias_data,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    int sm_count,
    cudaStream_t stream);

template <typename T>
__global__ void MatMulNBitsBiasAddKernel(T* output, const T* bias_data, int n, int64_t total) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total;
       idx += stride) {
    const int col = static_cast<int>(idx % n);
    // Accumulate in float to stay accurate for half/bfloat16.
    output[idx] = static_cast<T>(static_cast<float>(output[idx]) + static_cast<float>(bias_data[col]));
  }
}

template <class T>
void LaunchMatMulNBitsBiasAdd(T* output, const T* bias_data, int m, int n, cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(m) * static_cast<int64_t>(n);
  if (total == 0) {
    return;
  }
  constexpr int kThreadsPerBlock = 256;
  // Cap the grid at the CUDA gridDim.x limit (2^31 - 1); the kernel uses a grid-stride loop, so a
  // capped grid still covers every element without truncating the launch size.
  constexpr int64_t kMaxGridBlocks = 2147483647;
  const int64_t blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const int64_t num_blocks = blocks < kMaxGridBlocks ? blocks : kMaxGridBlocks;
  MatMulNBitsBiasAddKernel<T><<<static_cast<unsigned int>(num_blocks), kThreadsPerBlock, 0, stream>>>(
      output, bias_data, n, total);
}

template void LaunchMatMulNBitsBiasAdd<float>(
    float* output, const float* bias_data, int m, int n, cudaStream_t stream);

template void LaunchMatMulNBitsBiasAdd<half>(
    half* output, const half* bias_data, int m, int n, cudaStream_t stream);

template void LaunchMatMulNBitsBiasAdd<nv_bfloat16>(
    nv_bfloat16* output, const nv_bfloat16* bias_data, int m, int n, cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
