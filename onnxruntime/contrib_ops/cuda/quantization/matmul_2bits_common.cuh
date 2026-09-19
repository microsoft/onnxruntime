// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Shared geometry for the 2-bit GEMV kernels. One uint32 of the packed weight blob holds 16
// consecutive codes, so a thread consumes 16 elements of K per iteration and a warp consumes 512.
// block_size is a power of two in [16, 512] for every shape these kernels accept, so a thread's
// 16 codes always belong to a single quantization block.
constexpr int kColsPerThreadBlock2b = 8;
constexpr int kElementsPerThreadPerIteration2b = 16;
constexpr int kWarpSize2b = onnxruntime::cuda::GPU_WARP_SIZE;
constexpr int kElementsPerByte2b = 4;
constexpr uint8_t kDefaultZeroPoint2b = 2;  // 1 << (bits - 1)

template <class T>
__device__ __forceinline__ float ToFloat2b(T v);
template <>
__device__ __forceinline__ float ToFloat2b<float>(float v) { return v; }
template <>
__device__ __forceinline__ float ToFloat2b<half>(half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float ToFloat2b<nv_bfloat16>(nv_bfloat16 v) { return __bfloat162float(v); }

// Reads the 2-bit zero point of quantization block `block_index` from one output channel's
// zero-point row. The uint8 row is bit-packed four blocks to a byte, low crumb first.
__device__ __forceinline__ uint8_t UnpackZeroPoint2b(const uint8_t* zero_points_row, int block_index) {
  return static_cast<uint8_t>((zero_points_row[block_index >> 2] >> ((block_index & 0x03) << 1)) & 0x03);
}

// ---------------------------------------------------------------------------------------------
// Per-activation-type traits. A 16-element chunk of dequantized weights (Weights) and the matching
// activations (Acts) are held in whatever lane order makes the dot product cheapest, so callers
// never need to know which representation was chosen.
//
// Instruction count matters even though the GEMV is nominally memory bound: at 2 bits per weight
// there are 4 multiply-accumulates per byte of B, four times the 8-bit ratio, so a naive
// per-element integer-to-float conversion makes the kernel instruction bound.
// ---------------------------------------------------------------------------------------------
template <class T>
struct Traits2b {
  struct Weights {
    float v[kElementsPerThreadPerIteration2b];
  };
  struct Acts {
    float v[kElementsPerThreadPerIteration2b];
  };
  using Acc = float;
};

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
template <>
struct Traits2b<half> {
  // Lane j holds elements (j, j + 8). That pairing falls straight out of masking the shifted
  // packed word with 0x00030003, which is what keeps the expansion at two instructions per lane.
  struct Weights {
    half2 v[8];
  };
  struct Acts {
    half2 v[8];
  };
  using Acc = half2;
};
#endif

// Generic fp32 path, used for float and bfloat16 activations.
template <class T>
__device__ __forceinline__ void DequantizeSixteenGeneric2b(uint32_t values_quant, T scale, uint8_t zp,
                                                           float (&dq)[kElementsPerThreadPerIteration2b]) {
  const float scale_f = ToFloat2b<T>(scale);
  const float zp_adjust = -scale_f * static_cast<float>(zp);
#pragma unroll
  for (int i = 0; i < kElementsPerThreadPerIteration2b; ++i) {
    dq[i] = fmaf(static_cast<float>((values_quant >> (2 * i)) & 0x03u), scale_f, zp_adjust);
  }
}

template <class T>
__device__ __forceinline__ void LoadSixteenGeneric2b(const T* a, float (&out)[kElementsPerThreadPerIteration2b]) {
#pragma unroll
  for (int i = 0; i < kElementsPerThreadPerIteration2b; ++i) {
    out[i] = ToFloat2b<T>(a[i]);
  }
}

__device__ __forceinline__ void DotAccumGeneric2b(const float (&w)[kElementsPerThreadPerIteration2b],
                                                  const float (&a)[kElementsPerThreadPerIteration2b], float& acc) {
#pragma unroll
  for (int i = 0; i < kElementsPerThreadPerIteration2b; ++i) {
    acc = fmaf(w[i], a[i], acc);
  }
}

__device__ __forceinline__ void DequantizeSixteen2b(uint32_t values_quant, float scale, uint8_t zp,
                                                    Traits2b<float>::Weights& w) {
  DequantizeSixteenGeneric2b<float>(values_quant, scale, zp, w.v);
}
__device__ __forceinline__ void LoadSixteen2b(const float* a, Traits2b<float>::Acts& out) {
  LoadSixteenGeneric2b<float>(a, out.v);
}
__device__ __forceinline__ void DotAccum2b(const Traits2b<float>::Weights& w, const Traits2b<float>::Acts& a,
                                           float& acc) {
  DotAccumGeneric2b(w.v, a.v, acc);
}

__device__ __forceinline__ void DequantizeSixteen2b(uint32_t values_quant, nv_bfloat16 scale, uint8_t zp,
                                                    Traits2b<nv_bfloat16>::Weights& w) {
  DequantizeSixteenGeneric2b<nv_bfloat16>(values_quant, scale, zp, w.v);
}
__device__ __forceinline__ void LoadSixteen2b(const nv_bfloat16* a, Traits2b<nv_bfloat16>::Acts& out) {
  LoadSixteenGeneric2b<nv_bfloat16>(a, out.v);
}
__device__ __forceinline__ void DotAccum2b(const Traits2b<nv_bfloat16>::Weights& w,
                                           const Traits2b<nv_bfloat16>::Acts& a, float& acc) {
  DotAccumGeneric2b(w.v, a.v, acc);
}

__device__ __forceinline__ float HorizontalAdd2b(float acc) { return acc; }

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530) && !defined(__HIPCC__)
// 0x6400 is half(1024). OR-ing a 2-bit code into its low mantissa bits yields half(1024 + code)
// exactly, so subtracting half(1024) recovers the code with no rounding. Masking the shifted word
// with 0x00030003 extracts codes j and j + 8 together, one LOP3 for the pair.
__device__ __forceinline__ void DequantizeSixteen2b(uint32_t values_quant, half scale, uint8_t zp,
                                                    Traits2b<half>::Weights& w) {
  constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;  // (a & b) | c
  constexpr uint32_t kCrumbMask = 0x00030003;
  constexpr uint32_t kMagicNum = 0x64006400;

  const half2 scale_h2 = __half2half2(scale);
  const half2 zp_adjust_h2 = __half2half2(__hneg(scale) * __ushort2half_rn(zp));

  uint32_t* h = reinterpret_cast<uint32_t*>(w.v);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t shifted = values_quant >> (2 * j);
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(h[j])
                 : "r"(shifted), "n"(kCrumbMask), "n"(kMagicNum), "n"(kImmLut));
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[j]) : "r"(h[j]), "r"(kMagicNum));
    w.v[j] = __hfma2(w.v[j], scale_h2, zp_adjust_h2);
  }
}

// Repacks 16 contiguous activations into the (j, j + 8) lane order used by Weights.
__device__ __forceinline__ void LoadSixteen2b(const half* a, Traits2b<half>::Acts& out) {
  constexpr uint32_t kLowHalf2 = 0x5410;
  constexpr uint32_t kHighHalf2 = 0x7632;
  const uint4 lo = *reinterpret_cast<const uint4*>(a);
  const uint4 hi = *reinterpret_cast<const uint4*>(a + 8);
  const uint32_t words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  uint32_t* o = reinterpret_cast<uint32_t*>(out.v);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(o[2 * i]) : "r"(words[i]), "r"(words[i + 4]), "r"(kLowHalf2));
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(o[2 * i + 1]) : "r"(words[i]), "r"(words[i + 4]), "r"(kHighHalf2));
  }
}

__device__ __forceinline__ void DotAccum2b(const Traits2b<half>::Weights& w, const Traits2b<half>::Acts& a,
                                           half2& acc) {
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    acc = __hfma2(w.v[j], a.v[j], acc);
  }
}

__device__ __forceinline__ float HorizontalAdd2b(half2 acc) {
  return __half2float(acc.x) + __half2float(acc.y);
}
#else
__device__ __forceinline__ void DequantizeSixteen2b(uint32_t values_quant, half scale, uint8_t zp,
                                                    Traits2b<half>::Weights& w) {
  DequantizeSixteenGeneric2b<half>(values_quant, scale, zp, w.v);
}
__device__ __forceinline__ void LoadSixteen2b(const half* a, Traits2b<half>::Acts& out) {
  LoadSixteenGeneric2b<half>(a, out.v);
}
__device__ __forceinline__ void DotAccum2b(const Traits2b<half>::Weights& w, const Traits2b<half>::Acts& a,
                                           float& acc) {
  DotAccumGeneric2b(w.v, a.v, acc);
}
#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
