// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kColsPerThreadBlock = 8;
constexpr int kElementsPerThreadPerIteration = 8;
constexpr int kWarpSize = onnxruntime::cuda::GPU_WARP_SIZE;

template <typename T>
__device__ __forceinline__ T WarpUniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  p.asInt = onnxruntime::cuda::WARP_SHFL((unsigned)p.asInt, 0);
  return p.value;
}

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
__device__ __forceinline__ void Convert8xInt4To8xHalfs(uint32_t value, half2* half_2x4) {
  uint32_t* h = reinterpret_cast<uint32_t*>(half_2x4);
  constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
  constexpr uint32_t kBottomMask = 0x000f000f;
  constexpr uint32_t kTopMask = 0x00f000f0;
  constexpr uint32_t kI4sToF16sMagicNum = 0x64006400;
  const uint32_t top_i4s = value >> 8;
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[0])
               : "r"(value), "n"(kBottomMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[1])
               : "r"(value), "n"(kTopMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(kBottomMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(kTopMask), "n"(kI4sToF16sMagicNum), "n"(kImmLut));

  constexpr uint32_t kFp16TopMagicNum = 0x64006400;
  constexpr uint32_t kOneSixteenth = 0x2c002c00;
  constexpr uint32_t kNeg64 = 0xd400d400;
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(kFp16TopMagicNum));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(kOneSixteenth), "r"(kNeg64));
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(kFp16TopMagicNum));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(kOneSixteenth), "r"(kNeg64));
}

__device__ __forceinline__ void AccumulateEightElements4b(
    uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
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

  half2 elements[4];
  Convert8xInt4To8xHalfs(values_quant, elements);
  half2 v0 = elements[0] * scale_half2 + zp_adjust2;
  half2 v1 = elements[1] * scale_half2 + zp_adjust2;
  half2 v2 = elements[2] * scale_half2 + zp_adjust2;
  half2 v3 = elements[3] * scale_half2 + zp_adjust2;

  const half2 sum_half2_0 = __halves2half2(sums[0], sums[1]) + v0 * bit_cast<half2>(vec_permuted.x);
  const half2 sum_half2_1 = __halves2half2(sums[2], sums[3]) + v1 * bit_cast<half2>(vec_permuted.y);
  const half2 sum_half2_2 = __halves2half2(sums[4], sums[5]) + v2 * bit_cast<half2>(vec_permuted.z);
  const half2 sum_half2_3 = __halves2half2(sums[6], sums[7]) + v3 * bit_cast<half2>(vec_permuted.w);
  sums[0] = sum_half2_0.x;
  sums[1] = sum_half2_0.y;
  sums[2] = sum_half2_1.x;
  sums[3] = sum_half2_1.y;
  sums[4] = sum_half2_2.x;
  sums[5] = sum_half2_2.y;
  sums[6] = sum_half2_3.x;
  sums[7] = sum_half2_3.y;
}
#else
__device__ __forceinline__ void AccumulateEightElements4b(
    uint32_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
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

  const half2 sum_half2_0 = __halves2half2(sums[0], sums[1]) + v0 * bit_cast<half2>(vec_a.x);
  const half2 sum_half2_1 = __halves2half2(sums[2], sums[3]) + v1 * bit_cast<half2>(vec_a.y);
  const half2 sum_half2_2 = __halves2half2(sums[4], sums[5]) + v2 * bit_cast<half2>(vec_a.z);
  const half2 sum_half2_3 = __halves2half2(sums[6], sums[7]) + v3 * bit_cast<half2>(vec_a.w);
  sums[0] = sum_half2_0.x;
  sums[1] = sum_half2_0.y;
  sums[2] = sum_half2_1.x;
  sums[3] = sum_half2_1.y;
  sums[4] = sum_half2_2.x;
  sums[5] = sum_half2_2.y;
  sums[6] = sum_half2_3.x;
  sums[7] = sum_half2_3.y;
}
#endif

__device__ __forceinline__ void AccumulateEightElements4b(
    uint32_t values_quant, float scale, uint8_t zp, const float* a, float* sums) {
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
#else
  (void)value;
  (void)bf16_2x4;
#endif
}

__device__ __forceinline__ void AccumulateEightElements4b(
    uint32_t values_quant, nv_bfloat16 scale, uint8_t zp, const nv_bfloat16* a, nv_bfloat16* sums) {
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

  __nv_bfloat162 elements[4];
  Convert8xInt4To8xBF16s(values_quant, elements);
  __nv_bfloat162 v0 = __hfma2(elements[0], scale_bf162, zp_adjust2);
  __nv_bfloat162 v1 = __hfma2(elements[1], scale_bf162, zp_adjust2);
  __nv_bfloat162 v2 = __hfma2(elements[2], scale_bf162, zp_adjust2);
  __nv_bfloat162 v3 = __hfma2(elements[3], scale_bf162, zp_adjust2);

  const __nv_bfloat162 sum_bf162_0 =
      __hfma2(v0, bit_cast<__nv_bfloat162>(vec_permuted.x), __halves2bfloat162(sums[0], sums[1]));
  const __nv_bfloat162 sum_bf162_1 =
      __hfma2(v1, bit_cast<__nv_bfloat162>(vec_permuted.y), __halves2bfloat162(sums[2], sums[3]));
  const __nv_bfloat162 sum_bf162_2 =
      __hfma2(v2, bit_cast<__nv_bfloat162>(vec_permuted.z), __halves2bfloat162(sums[4], sums[5]));
  const __nv_bfloat162 sum_bf162_3 =
      __hfma2(v3, bit_cast<__nv_bfloat162>(vec_permuted.w), __halves2bfloat162(sums[6], sums[7]));
  sums[0] = sum_bf162_0.x;
  sums[1] = sum_bf162_0.y;
  sums[2] = sum_bf162_1.x;
  sums[3] = sum_bf162_1.y;
  sums[4] = sum_bf162_2.x;
  sums[5] = sum_bf162_2.y;
  sums[6] = sum_bf162_3.x;
  sums[7] = sum_bf162_3.y;
#else
  (void)values_quant;
  (void)scale;
  (void)zp;
  (void)a;
  (void)sums;
#endif
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime