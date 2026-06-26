// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <type_traits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
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

template <typename T>
__device__ __forceinline__ T WarpUniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  p.asInt = WARP_SHFL((unsigned)p.asInt, 0);
  return p.value;
}

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
// Convert 8 4bits integer stored in one uint32_t to 8 halfs.
// 8 4bits with order 0,1,2,3,4,5,6,7,8 will be converted to 8 halfs with order 0,4,1,5,2,6,3,7
__device__ __forceinline__ void Convert8xInt4To8xHalfs(uint32_t value, half2* half_2x4) {
  uint32_t* h = reinterpret_cast<uint32_t*>(half_2x4);

  // From https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
  // First, we extract the i4s and construct an intermediate fp16 number.
  constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
  constexpr uint32_t kBottomMask = 0x000f000f;
  constexpr uint32_t kTopMask = 0x00f000f0;
  constexpr uint32_t kI4sToF16sMagicNum = 0x64006400;

  // Note that the entire sequence only requires 1 shift instruction. This is thanks to the register packing
  // format and the fact that we force our integers to be unsigned, and account for this in the fp16 subtractions.
  // In addition, I exploit the fact that sub and fma have the same throughput in order to convert elt_23 and
  // elt_67 to fp16 without having to shift them to the bottom bits before hand.

  // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide RAW dependency if we issue
  // immediately before required.
  const uint32_t top_i4s = value >> 8;
  // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(value), "n"(kBottomMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(value), "n"(kTopMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(kBottomMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(kTopMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));

  // I use inline PTX below because I am not sure if the compiler will emit float2half instructions if I use the
  // half2 ctor. In this case, I chose performance reliability over code readability.

  // This is the half2 {1024, 1024} represented as an integer.
  constexpr uint32_t kFp16TopMagicNum = 0x64006400;
  // This is the half2 {1 / 16, 1 / 16} represented as an integer.
  constexpr uint32_t kOneSixteenth = 0x2c002c00;
  // This is the half2 {-64, -64} represented as an integer.
  constexpr uint32_t kNeg64 = 0xd400d400;

  // Finally, we construct the output numbers.
  // Convert elt_01
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(kFp16TopMagicNum));
  // Convert elt_23
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(kOneSixteenth), "r"(kNeg64));
  // Convert elt_45
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(kFp16TopMagicNum));
  // Convert elt_67
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(kOneSixteenth), "r"(kNeg64));
}

__device__ __forceinline__ void AccumulateEightElements4b(uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  constexpr uint32_t kLowHalf2 = 0x5410;
  constexpr uint32_t kHighHalf2 = 0x7632;

  uint4 vec_permuted;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.x) : "r"(vec_a.x), "r"(vec_a.z), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.y) : "r"(vec_a.x), "r"(vec_a.z), "r"(kHighHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.z) : "r"(vec_a.y), "r"(vec_a.w), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.w) : "r"(vec_a.y), "r"(vec_a.w), "r"(kHighHalf2));

  half2 elements[4];  // [04, 15, 26, 37]

  Convert8xInt4To8xHalfs(values_quant, elements);

  half2 v0 = elements[0] * scale_half2 + zp_adjust2;
  half2 v1 = elements[1] * scale_half2 + zp_adjust2;
  half2 v2 = elements[2] * scale_half2 + zp_adjust2;
  half2 v3 = elements[3] * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_permuted.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_permuted.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_permuted.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_permuted.w))));
}
#else
__device__ __forceinline__ void AccumulateEightElements4b(uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  half2 element01 = __halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF));
  half2 element23 = __halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF));
  half2 element45 = __halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF));
  half2 element67 = __halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF));

  half2 v0 = element01 * scale_half2 + zp_adjust2;
  half2 v1 = element23 * scale_half2 + zp_adjust2;
  half2 v2 = element45 * scale_half2 + zp_adjust2;
  half2 v3 = element67 * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_a.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_a.w))));
}
#endif

__device__ __forceinline__ void AccumulateEightElements4b(uint32_t values_quant, float scale, uint8_t zp, const float* a, float* sums) {
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xF) * scale + zp_adjust;
  float v1 = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  float v2 = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  float v3 = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  float v4 = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  float v5 = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  float v6 = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  float v7 = float((values_quant >> 28) & 0xF) * scale + zp_adjust;

  sums[0] += v0 * a_vec_0.x;
  sums[1] += v1 * a_vec_0.y;
  sums[2] += v2 * a_vec_0.z;
  sums[3] += v3 * a_vec_0.w;
  sums[4] += v4 * a_vec_1.x;
  sums[5] += v5 * a_vec_1.y;
  sums[6] += v6 * a_vec_1.z;
  sums[7] += v7 * a_vec_1.w;
}

// Convert 8 4-bit integers stored in one uint32_t to 8 bfloat16s.
// The output order is [0,4], [1,5], [2,6], [3,7]
__device__ __forceinline__ void Convert8xInt4To8xBF16s(uint32_t value, __nv_bfloat162* bf16_2x4) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int i0 = (value >> 0) & 0xF;
  const int i1 = (value >> 4) & 0xF;
  const int i2 = (value >> 8) & 0xF;
  const int i3 = (value >> 12) & 0xF;
  const int i4 = (value >> 16) & 0xF;
  const int i5 = (value >> 20) & 0xF;
  const int i6 = (value >> 24) & 0xF;
  const int i7 = (value >> 28) & 0xF;

  bf16_2x4[0] = __floats2bfloat162_rn(static_cast<float>(i0), static_cast<float>(i4));
  bf16_2x4[1] = __floats2bfloat162_rn(static_cast<float>(i1), static_cast<float>(i5));
  bf16_2x4[2] = __floats2bfloat162_rn(static_cast<float>(i2), static_cast<float>(i6));
  bf16_2x4[3] = __floats2bfloat162_rn(static_cast<float>(i3), static_cast<float>(i7));
#endif
}

__device__ __forceinline__ void AccumulateEightElements4b(uint32_t values_quant, nv_bfloat16 scale, uint8_t zp, const nv_bfloat16* a, nv_bfloat16* sums) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  __nv_bfloat162 scale_bf162 = __bfloat162bfloat162(scale);
  nv_bfloat16 zp_adjust = -scale * __uint2bfloat16_rn(zp);
  __nv_bfloat162 zp_adjust2 = __bfloat162bfloat162(zp_adjust);

  const uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  constexpr uint32_t kLowHalf2 = 0x5410;
  constexpr uint32_t kHighHalf2 = 0x7632;

  uint4 vec_permuted;
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.x) : "r"(vec_a.x), "r"(vec_a.z), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.y) : "r"(vec_a.x), "r"(vec_a.z), "r"(kHighHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.z) : "r"(vec_a.y), "r"(vec_a.w), "r"(kLowHalf2));
  asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(vec_permuted.w) : "r"(vec_a.y), "r"(vec_a.w), "r"(kHighHalf2));

  __nv_bfloat162 elements[4];  // [04, 15, 26, 37]
  Convert8xInt4To8xBF16s(values_quant, elements);

  __nv_bfloat162 v0 = __hfma2(elements[0], scale_bf162, zp_adjust2);
  __nv_bfloat162 v1 = __hfma2(elements[1], scale_bf162, zp_adjust2);
  __nv_bfloat162 v2 = __hfma2(elements[2], scale_bf162, zp_adjust2);
  __nv_bfloat162 v3 = __hfma2(elements[3], scale_bf162, zp_adjust2);

  __nv_bfloat162* sums_bf162 = reinterpret_cast<__nv_bfloat162*>(sums);
  sums_bf162[0] = __hfma2(v0, *reinterpret_cast<const __nv_bfloat162*>(&vec_permuted.x), sums_bf162[0]);
  sums_bf162[1] = __hfma2(v1, *reinterpret_cast<const __nv_bfloat162*>(&vec_permuted.y), sums_bf162[1]);
  sums_bf162[2] = __hfma2(v2, *reinterpret_cast<const __nv_bfloat162*>(&vec_permuted.z), sums_bf162[2]);
  sums_bf162[3] = __hfma2(v3, *reinterpret_cast<const __nv_bfloat162*>(&vec_permuted.w), sums_bf162[3]);
#endif
}

constexpr int kColsPerThreadBlock = 8;
constexpr int kElementsPerThreadPerIteration = 8;
constexpr int kWarpSize = GPU_WARP_SIZE;

// kernel for 4bits quantized gemv, i.e., computing A(1,K) x B(K, N)
// B(K, N) is quantized blockwise with 4bits and stored as [N, (K + block_size - 1)/block_size, blob]
// The thread block size is (kWarpSize, kColsPerThreadBlock) and grid size is (N/kColsPerThreadBlock, 1)
// Each thread block computes [1, K] x [kColsPerThreadBlock, (K + block_size - 1)/block_size, blob],
//     i.e., computing kColsPerThreadBlock per block and a warp reduce (1, K) x (K)
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloatInt4Kernel(
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
  const int m_id = blockIdx.y;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = n_block_id * kColsPerThreadBlock + warp_id;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

  extern __shared__ char shared_buffer[];
  // load scale to shared buffer
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

  a_data += m_id * k + (lane_id << 3);

  b_scale_vec += warp_id * blocks_per_K;

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

#define UnRollReduction(unroll_size)                                                              \
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
        AccumulateEightElements4b(value, scale, zp, a_data + k_id + i * k_per_iter, sums);        \
      }                                                                                           \
      b_data_quant += k_per_iter / 2 * kUnroll;                                                   \
      t_meta_k += k_per_iter / block_size * kUnroll;                                              \
    }                                                                                             \
  } while (false)

  UnRollReduction(16);
  UnRollReduction(4);
  UnRollReduction(1);
#undef UnRollReduction

  // handle reminder
  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[t_meta_k];
    uint8_t zp = 8;
    if constexpr (has_zero_point) {
      zp = b_zp_vec[t_meta_k];
    }
    AccumulateEightElements4b(value, scale, zp, a_data + k_id, sums);
  }

  float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  // warp reduction
  for (int i = kWarpSize / 2; i > 0; i = i / 2) {
    sum += WARP_SHFL_DOWN(sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}  // namespace cuda

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

// ===== Small-M batched GEMV (speculative-decode verify / short prefill) =====
// The single-row MatMulFloatInt4Kernel launches one block per output row (grid.y = m), so each row
// independently re-reads and re-dequantizes all of B; weight traffic and dequant work scale with M.
// For 2 <= M <= cap we instead dequantize each packed weight word once and accumulate it against
// CtaM activation rows held in registers, cutting weight traffic to ceil(M/CtaM)x. This is the same
// design used by TensorRT-LLM weightOnlyBatchedGemv / AWQ / llama.cpp MMVQ for small batch.
//
// Upper bound on M is dtype-dependent (measured on A100 vs the dequantize+cuBLAS fallback): for
// fp16/bf16 the batched GEMV becomes compute-bound and loses to tensor-core GEMM by M~16, so it is
// only used through M<=8. fp32 has no tensor-core GEMM and a costly fp32 dequant, so batched wins
// through M<=16.
constexpr int kSmallMMax = 16;
template <class T>
__host__ __device__ constexpr int SmallMCap() {
  return std::is_same<T, float>::value ? kSmallMMax : 8;
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

// Batched GEMV: block computes CtaM rows x kColsPerThreadBlock columns. Grid is
// (ceil(N/kColsPerThreadBlock), ceil(M/CtaM)). Mirrors MatMulFloatInt4Kernel's shared-memory scale/zp
// staging and packed-weight indexing; the only change is looping CtaM rows per dequantized word.
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

#define SmallMDispatch(BS, CM)                                                                  \
  if (nullptr != zero_points) {                                                                 \
    MatMulFloatInt4KernelSmallM<T, BS, true, CM><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, (k + BS - 1) / BS);    \
  } else {                                                                                      \
    MatMulFloatInt4KernelSmallM<T, BS, false, CM><<<blocks, threads, shared_mem_size, stream>>>(\
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, (k + BS - 1) / BS);    \
  }
#define SmallMDispatchBlock(CM)            \
  if (16 == block_size) {                  \
    SmallMDispatch(16, CM)                 \
  } else if (32 == block_size) {           \
    SmallMDispatch(32, CM)                 \
  } else if (64 == block_size) {           \
    SmallMDispatch(64, CM)                 \
  } else if (128 == block_size) {          \
    SmallMDispatch(128, CM)                \
  } else {                                 \
    return false;                          \
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

// ===== accuracy_level=4 path: int8-activation dp4a batched GEMV (W4A8) =====
// Quantizes the fp16/bf16/float activation to int8 per row (symmetric), then runs a batched GEMV that
// reuses each 4-bit weight (unpacked to int8) across CtaM rows via dp4a. dp4a has ~4x the throughput of
// the fp16 FMA path, so the GEMV stays memory-bound (flat vs M) through the small-M verify range where
// the fp16 CtaM kernel becomes compute-bound. Reads the same [N, blocks, blob] weight layout (no prepack).
__device__ __forceinline__ float ToFloatAct(half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloatAct(nv_bfloat16 v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __bfloat162float(v);
#else
  return float(v);
#endif
}
__device__ __forceinline__ float ToFloatAct(float v) { return v; }

// One block per activation row: int8 symmetric per-row quantization. aq[m][k], ascale[m] = maxabs/127.
template <typename T>
__global__ void QuantizeRowwiseInt8Kernel(const T* a, int8_t* aq, float* ascale, int m, int k) {
  int row = blockIdx.x;
  if (row >= m) return;
  const T* arow = a + static_cast<size_t>(row) * k;
  float maxabs = 0.f;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    maxabs = fmaxf(maxabs, fabsf(ToFloatAct(arow[i])));
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) maxabs = fmaxf(maxabs, __shfl_down_sync(0xffffffff, maxabs, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = maxabs;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  float scale = red[0] / 127.f;
  if (scale == 0.f) scale = 1e-8f;
  if (threadIdx.x == 0) ascale[row] = scale;
  float inv = 1.f / scale;
  for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int q = __float2int_rn(ToFloatAct(arow[i]) * inv);
    q = max(-127, min(127, q));
    aq[static_cast<size_t>(row) * k + i] = static_cast<int8_t>(q);
  }
}

template <typename T>
void LaunchQuantizeRowwiseInt8(const T* a, int8_t* aq, float* ascale, int m, int k, cudaStream_t stream) {
  QuantizeRowwiseInt8Kernel<T><<<m, 256, 0, stream>>>(a, aq, ascale, m, k);
}

// Batched int8 dp4a GEMV. Block (kWarpSize, kColsPerThreadBlock); grid (N/kColsPerThreadBlock, ceil(m/CtaM)).
// Mirrors MatMulFloatInt4Kernel's shared scale/zp staging and packed-weight indexing; the per-lane K chunk
// is dotted via dp4a against CtaM int8 activation rows, scaled by the block weight scale and row act scale.
template <class T, int block_size, bool has_zero_point, int CtaM>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulInt4Dp4aKernel(
    T* output,
    const int8_t* aq,
    const float* ascale,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m, int n, int k, int blocks_per_K) {
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
  const int8_t* a_row[CtaM];
#pragma unroll
  for (int r = 0; r < CtaM; r++) a_row[r] = aq + static_cast<size_t>(m_base + r) * k + (lane_id << 3);
  b_scale_vec += warp_id * blocks_per_K;

  float facc[CtaM];
#pragma unroll
  for (int r = 0; r < CtaM; r++) facc[r] = 0.f;

  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

  for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    float wscale = static_cast<float>(b_scale_vec[t_meta_k]);
    int zp = 8;
    if constexpr (has_zero_point) zp = b_zp_vec[t_meta_k];
    // unpack 8 nibbles -> 8 int8 (q - zp), packed as two int32 for dp4a
    int8_t w8[8];
#pragma unroll
    for (int i = 0; i < 8; i++) w8[i] = static_cast<int8_t>(static_cast<int>((value >> (4 * i)) & 0xF) - zp);
    int wlo = *reinterpret_cast<const int*>(w8);
    int whi = *reinterpret_cast<const int*>(w8 + 4);
#pragma unroll
    for (int r = 0; r < CtaM; r++) {
      if (r >= valid) continue;
      const int8_t* ap = a_row[r] + k_id;
      int alo = *reinterpret_cast<const int*>(ap);
      int ahi = *reinterpret_cast<const int*>(ap + 4);
      int dot = __dp4a(alo, wlo, 0);
      dot = __dp4a(ahi, whi, dot);
      facc[r] += wscale * static_cast<float>(dot);
    }
    b_data_quant += k_per_iter / 2;
    t_meta_k += k_per_iter / block_size;
  }

  // remainder (k not a multiple of k_per_iter); k is a multiple of block_size so each lane's 8-chunk is valid
  if (k_id + (lane_id << 3) < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    float wscale = static_cast<float>(b_scale_vec[t_meta_k]);
    int zp = 8;
    if constexpr (has_zero_point) zp = b_zp_vec[t_meta_k];
    int8_t w8[8];
#pragma unroll
    for (int i = 0; i < 8; i++) w8[i] = static_cast<int8_t>(static_cast<int>((value >> (4 * i)) & 0xF) - zp);
    int wlo = *reinterpret_cast<const int*>(w8);
    int whi = *reinterpret_cast<const int*>(w8 + 4);
#pragma unroll
    for (int r = 0; r < CtaM; r++) {
      if (r >= valid) continue;
      const int8_t* ap = a_row[r] + k_id;
      int alo = *reinterpret_cast<const int*>(ap);
      int ahi = *reinterpret_cast<const int*>(ap + 4);
      int dot = __dp4a(alo, wlo, 0);
      dot = __dp4a(ahi, whi, dot);
      facc[r] += wscale * static_cast<float>(dot);
    }
  }

#pragma unroll
  for (int r = 0; r < CtaM; r++) {
    if (r >= valid) continue;
    float sum = facc[r];
    for (int i = kWarpSize / 2; i > 0; i = i / 2) sum += WARP_SHFL_DOWN(sum, i);
    if (lane_id == 0) output[static_cast<size_t>(m_base + r) * n + n_id] = static_cast<T>(sum * ascale[m_base + r]);
  }
}

// Cap on M for the int8 dp4a verify path (kMatMulInt8Dp4aMaxM in matmul_nbits.cuh). dp4a keeps the batched
// GEMV memory-bound (flat) through the spec-decode verify range; beyond it the per-row weight re-read makes
// it lose to dequant+cuBLAS, so M>cap falls through. M==1 also uses it when accuracy_level=4 so the target
// stays self-consistent across phases.

template <class T>
bool TryMatMulInt8Dp4a(
    T* output,
    const int8_t* aq,
    const float* ascale,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m, int n, int k, int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream) {
  if (m < 1 || m > kMatMulInt8Dp4aMaxM || n % kColsPerThreadBlock != 0 || k % kElementsPerThreadPerIteration != 0) {
    return false;
  }
  if (k % block_size != 0) return false;  // exact per-lane block scale indexing
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;
  if (k_per_iter % block_size != 0) return false;

  int blocks_per_K = k / block_size;
  size_t shared_mem_size = sizeof(T) * blocks_per_K * kColsPerThreadBlock +
                           static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
  if (shared_mem_size > shared_mem_per_block) return false;

  const int cta_m = (m <= 2) ? 2 : ((m <= 4) ? 4 : 8);
  dim3 threads(GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  dim3 blocks(n / kColsPerThreadBlock, (m + cta_m - 1) / cta_m);

#define Int8Dp4aDispatch(BS, CM)                                                                       \
  if (nullptr != zero_points) {                                                                        \
    MatMulInt4Dp4aKernel<T, BS, true, CM><<<blocks, threads, shared_mem_size, stream>>>(               \
        output, aq, ascale, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);            \
  } else {                                                                                             \
    MatMulInt4Dp4aKernel<T, BS, false, CM><<<blocks, threads, shared_mem_size, stream>>>(              \
        output, aq, ascale, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);            \
  }
#define Int8Dp4aDispatchBlock(CM)        \
  if (16 == block_size) { Int8Dp4aDispatch(16, CM) }      \
  else if (32 == block_size) { Int8Dp4aDispatch(32, CM) } \
  else if (64 == block_size) { Int8Dp4aDispatch(64, CM) } \
  else if (128 == block_size) { Int8Dp4aDispatch(128, CM) } \
  else { return false; }

  if (cta_m == 2) { Int8Dp4aDispatchBlock(2) }
  else if (cta_m == 4) { Int8Dp4aDispatchBlock(4) }
  else { Int8Dp4aDispatchBlock(8) }
#undef Int8Dp4aDispatchBlock
#undef Int8Dp4aDispatch
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

  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);
  dim3 threads(GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  int blocks_per_K = (k + block_size - 1) / block_size;
  size_t shared_mem_size = sizeof(T) * blocks_per_K * kColsPerThreadBlock +
                           static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

  // 2 <= m <= SmallMCap<T>(): batched GEMV that reuses each dequantized weight across CtaM rows.
  // m == 1 falls through to the single-row kernel below; larger m returned false above and uses
  // the dequantize + cuBLAS path.
  if (m >= 2) {
    return TryMatMulSmallM4Bits<T>(output, a_data, b_data_quant, scales_data, zero_points,
                                   m, n, k, block_size, shared_mem_size, stream);
  }

#define MatMulFloatInt4KernelDispatch(block_size)                                              \
  if (nullptr != zero_points) {                                                                \
    MatMulFloatInt4Kernel<T, block_size, true><<<blocks, threads, shared_mem_size, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);        \
  } else {                                                                                     \
    MatMulFloatInt4Kernel<T, block_size, false><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);        \
  }

  if (16 == block_size) {
    MatMulFloatInt4KernelDispatch(16);
  } else if (32 == block_size) {
    MatMulFloatInt4KernelDispatch(32);
  } else if (64 == block_size) {
    MatMulFloatInt4KernelDispatch(64);
  } else if (128 == block_size) {
    MatMulFloatInt4KernelDispatch(128);
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }

#undef MatMulFloatInt4KernelDispatch

  return true;
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
    cudaStream_t stream);

#define INSTANTIATE_INT8_DP4A(T)                                                            \
  template void LaunchQuantizeRowwiseInt8<T>(const T*, int8_t*, float*, int, int, cudaStream_t); \
  template bool TryMatMulInt8Dp4a<T>(T*, const int8_t*, const float*, const uint8_t*, const T*, \
                                     const uint8_t*, int, int, int, int, size_t, cudaStream_t)
INSTANTIATE_INT8_DP4A(float);
INSTANTIATE_INT8_DP4A(half);
INSTANTIATE_INT8_DP4A(nv_bfloat16);
#undef INSTANTIATE_INT8_DP4A

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
