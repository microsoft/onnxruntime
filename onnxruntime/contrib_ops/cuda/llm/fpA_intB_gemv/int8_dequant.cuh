// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Dequantizes symmetric group-wise INT8 weights in the SM80-interleaved layout of fpA_intB_gemv
// (ColumnMajorInterleaved, 64-wide K tiles, two real columns per interleaved row) into a row-major [n, k] matrix of
// the activation type, so that a large-M product can run as a plain cuBLAS GEMM.
//
// The CUTLASS fpA_intB GEMM dequantizes in registers inside its mainloop and reaches roughly 300 TFLOPS on H200 for
// the Qwen3.8 projections, while a BF16 cuBLAS GEMM reaches ~750. Materializing the weights once per call costs
// 3 * n * k bytes of DRAM traffic (about 35 us for the 84 MB in_proj matrix), which is below 5% of the GEMM once M is
// a few hundred rows.
//
// Layout (see int8_tc_gemv.cuh): the pair p of real columns (2p, 2p + 1) starts at byte p * 2k. K tile tau (64 wide),
// column c of the pair and 16-byte chunk t start at tau * 128 + c * 64 + t * 16. Word w of a chunk holds the logical K
// offsets (2w, 2w + 8, 2w + 1, 2w + 9) of the chunk in its bytes 0..3, i.e. the even bytes are K 0..7 and the odd bytes
// are K 8..15. The stored byte is q = value + 128; scales are [k / group_size, n].
//
// Numerics: (q - 128) * scale is exact in FP32 (an 8-bit integer times a 16-bit float), so the single rounding to the
// output type is the same rounding of (q - 128) * scale that the GEMV kernels and the CUTLASS converter apply.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace onnxruntime::llm::kernels::fpA_intB_gemv {
namespace dequant {

constexpr int kChunkK = 16;  // K values per 16-byte chunk
constexpr int kThreads = 256;

#if defined(__CUDACC__)

template <typename T>
struct Convert;

template <>
struct Convert<__nv_bfloat16> {
  using Pair = __nv_bfloat162;
  __device__ static __forceinline__ float ToFloat(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
  }
  __device__ static __forceinline__ uint32_t Pack(float lo, float hi) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  }
};

template <>
struct Convert<half> {
  using Pair = half2;
  __device__ static __forceinline__ float ToFloat(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
  }
  __device__ static __forceinline__ uint32_t Pack(float lo, float hi) {
    const half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
  }
};

__device__ __forceinline__ uint4 LoadWeights(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

// Bytes {b0, b1} of `word` as (q - 128) * scale values packed as two output-type lanes.
template <typename T>
__device__ __forceinline__ uint32_t DequantPair(uint32_t word, int shift_lo, int shift_hi, float scale) {
  const float lo = static_cast<float>(static_cast<int>((word >> shift_lo) & 0xFFu) - 128) * scale;
  const float hi = static_cast<float>(static_cast<int>((word >> shift_hi) & 0xFFu) - 128) * scale;
  return Convert<T>::Pack(lo, hi);
}

// One thread converts the 16 K values of one chunk for both columns of a pair. Consecutive threads take consecutive
// chunks of the same column pair, so a warp reads whole 128-byte lines (both columns) and writes 1 KB runs per row.
template <typename T>
__global__ void __launch_bounds__(kThreads)
    DequantizeInt8InterleavedKernel(const uint8_t* __restrict__ weight, const T* __restrict__ scales,
                                    T* __restrict__ out, int n, int k, int chunk_group_shift) {
  const int chunks_per_row = k / kChunkK;
  const int idx = blockIdx.x * kThreads + threadIdx.x;  // the launcher guarantees n / 2 * chunks_per_row < 2^31
  if (idx >= (n / 2) * chunks_per_row) {
    return;
  }
  const int pair = idx / chunks_per_row;
  const int chunk = idx - pair * chunks_per_row;
  const int tile = chunk >> 2;
  const int t = chunk & 3;

  const uint8_t* wp = weight + static_cast<int64_t>(pair) * 2 * k + static_cast<int64_t>(tile) * 128 + t * 16;
  const uint4 q0 = LoadWeights(wp);
  const uint4 q1 = LoadWeights(wp + 64);

  const uint32_t scale_pair =
      __ldg(reinterpret_cast<const uint32_t*>(scales + static_cast<int64_t>(chunk >> chunk_group_shift) * n + 2 * pair));
  const float s0 = Convert<T>::ToFloat(static_cast<uint16_t>(scale_pair & 0xFFFFu));
  const float s1 = Convert<T>::ToFloat(static_cast<uint16_t>(scale_pair >> 16));

  const uint32_t w0[4] = {q0.x, q0.y, q0.z, q0.w};
  const uint32_t w1[4] = {q1.x, q1.y, q1.z, q1.w};
  // Even bytes of a word are K (2w, 2w + 1) and odd bytes are K (2w + 8, 2w + 9) of the chunk.
  uint32_t lo0[4], hi0[4], lo1[4], hi1[4];
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    lo0[w] = DequantPair<T>(w0[w], 0, 16, s0);
    hi0[w] = DequantPair<T>(w0[w], 8, 24, s0);
    lo1[w] = DequantPair<T>(w1[w], 0, 16, s1);
    hi1[w] = DequantPair<T>(w1[w], 8, 24, s1);
  }

  T* o0 = out + static_cast<int64_t>(2 * pair) * k + static_cast<int64_t>(chunk) * kChunkK;
  T* o1 = o0 + k;
  reinterpret_cast<uint4*>(o0)[0] = make_uint4(lo0[0], lo0[1], lo0[2], lo0[3]);
  reinterpret_cast<uint4*>(o0)[1] = make_uint4(hi0[0], hi0[1], hi0[2], hi0[3]);
  reinterpret_cast<uint4*>(o1)[0] = make_uint4(lo1[0], lo1[1], lo1[2], lo1[3]);
  reinterpret_cast<uint4*>(o1)[1] = make_uint4(hi1[0], hi1[1], hi1[2], hi1[3]);
}

#endif  // __CUDACC__

}  // namespace dequant
}  // namespace onnxruntime::llm::kernels::fpA_intB_gemv
