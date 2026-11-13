// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Opt-in single-row (M == 1) fp16 GEMV for 4-bit MatMulNBits weights in the native column-major
// layout, with or without uint8-packed zero points. Only CUDA headers are included so the kernel can
// also be built standalone for micro-benchmarks.
//
// Work split: one warp owns kCols adjacent output columns. Lane l covers the 32-element slice
// [c * 1024 + l * 32, +32) of every 1024-element K chunk c, so every weight load is 16 bytes and the
// 32 activations of that slice are loaded and permuted once, then reused for all kCols columns. All
// weight loads of kChunksPerIter chunks are issued before any is consumed. A 32-element slice never
// straddles a quantization block (block_size is a multiple of 32), so one scale and zero point serve
// a whole 16-byte load. The zero point is folded into the exact int4 -> fp16 magic-number conversion,
// the scale is applied to the weights in fp16 (as the existing kernel does) and every 8-element
// partial dot product is accumulated in fp32.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace m1_wide {

constexpr int kWarpsPerBlock = 4;
constexpr int kLaneElems = 32;
constexpr int kChunkElems = 32 * kLaneElems;
constexpr int kChunksPerIter = 2;

__device__ __forceinline__ uint4 LoadWeights(const uint8_t* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
}

__device__ __forceinline__ uint32_t Lop3(uint32_t a, uint32_t b, uint32_t c) {
  constexpr uint32_t kImmLut = (0xf0 & 0xcc) | 0xaa;
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(r) : "r"(a), "r"(b), "r"(c), "n"(kImmLut));
  return r;
}

__device__ __forceinline__ half2 AsHalf2(uint32_t v) {
  return *reinterpret_cast<half2*>(&v);
}

// Returns sum_i (q_i - zp) * scale * a_i for the 8 nibbles of q. The activations are pre-permuted
// to (a0,a4), (a1,a5), (a2,a6), (a3,a7) to match the magic-number nibble order.
__device__ __forceinline__ float Dot8(uint32_t q, half2 zsub, half2 zfma, half2 scale, const half2* a) {
  constexpr uint32_t kBottomMask = 0x000f000f;
  constexpr uint32_t kTopMask = 0x00f000f0;
  constexpr uint32_t kMagic = 0x64006400;  // fp16 1024
  const half2 kOneSixteenth = AsHalf2(0x2c002c00);
  const uint32_t top = q >> 8;
  // (1024 + q) - (1024 + zp) and (1024 + 16 q) / 16 - (64 + zp) are exact in fp16.
  half2 w0 = __hsub2(AsHalf2(Lop3(q, kBottomMask, kMagic)), zsub);
  half2 w1 = __hfma2(AsHalf2(Lop3(q, kTopMask, kMagic)), kOneSixteenth, zfma);
  half2 w2 = __hsub2(AsHalf2(Lop3(top, kBottomMask, kMagic)), zsub);
  half2 w3 = __hfma2(AsHalf2(Lop3(top, kTopMask, kMagic)), kOneSixteenth, zfma);
  half2 d = __hmul2(__hmul2(w0, scale), a[0]);
  d = __hfma2(__hmul2(w1, scale), a[1], d);
  d = __hfma2(__hmul2(w2, scale), a[2], d);
  d = __hfma2(__hmul2(w3, scale), a[3], d);
  const float2 f = __half22float2(d);
  return f.x + f.y;
}

__device__ __forceinline__ void LoadPermutedActivations(const half* a, half2 (&ap)[16]) {
  constexpr uint32_t kLowHalf2 = 0x5410;
  constexpr uint32_t kHighHalf2 = 0x7632;
  const uint4* a4 = reinterpret_cast<const uint4*>(a);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint4 v = __ldg(a4 + i);
    uint32_t p[4];
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(p[0]) : "r"(v.x), "r"(v.z), "r"(kLowHalf2));
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(p[1]) : "r"(v.x), "r"(v.z), "r"(kHighHalf2));
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(p[2]) : "r"(v.y), "r"(v.w), "r"(kLowHalf2));
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(p[3]) : "r"(v.y), "r"(v.w), "r"(kHighHalf2));
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      ap[4 * i + j] = AsHalf2(p[j]);
    }
  }
}

template <int kBlockSize, bool kHasZeroPoint, int kCols>
__global__ void __launch_bounds__(kWarpsPerBlock * 32) MatMulFloat4BitsKernelM1Wide(
    half* __restrict__ output,
    const half* __restrict__ a,
    const uint8_t* __restrict__ b,
    const half* __restrict__ scales,
    const uint8_t* __restrict__ zero_points,
    int n,
    int k) {
  static_assert(kBlockSize % kLaneElems == 0, "a lane slice must not straddle a quantization block");
  const int lane = threadIdx.x & 31;
  const int col0 = (blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5)) * kCols;
  if (col0 >= n) {
    return;
  }
  const int blocks_per_k = k / kBlockSize;
  const int zp_stride = (blocks_per_k + 1) >> 1;
  const size_t col_bytes = static_cast<size_t>(k) >> 1;
  const uint8_t* b_col = b + static_cast<size_t>(col0) * col_bytes;
  const half* s_col = scales + static_cast<size_t>(col0) * blocks_per_k;
  const uint8_t* z_col = kHasZeroPoint ? zero_points + static_cast<size_t>(col0) * zp_stride : nullptr;

  float acc[kCols];
#pragma unroll
  for (int j = 0; j < kCols; ++j) {
    acc[j] = 0.f;
  }

  for (int kb = lane * kLaneElems; kb < k; kb += kChunkElems * kChunksPerIter) {
    uint4 w[kChunksPerIter][kCols];
    half2 s[kChunksPerIter][kCols];
    uint32_t zp[kChunksPerIter][kCols];
#pragma unroll
    for (int c = 0; c < kChunksPerIter; ++c) {
      const int kk = kb + c * kChunkElems;
      if (c == 0 || kk < k) {
        const int blk = kk / kBlockSize;
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
          w[c][j] = LoadWeights(b_col + j * col_bytes + (kk >> 1));
          s[c][j] = __half2half2(s_col[j * blocks_per_k + blk]);
          if constexpr (kHasZeroPoint) {
            const uint32_t packed = z_col[j * zp_stride + (blk >> 1)];
            zp[c][j] = (blk & 1) ? (packed >> 4) : (packed & 0x0f);
          } else {
            zp[c][j] = 8;
          }
        }
      }
    }
#pragma unroll
    for (int c = 0; c < kChunksPerIter; ++c) {
      const int kk = kb + c * kChunkElems;
      if (c == 0 || kk < k) {
        half2 ap[16];
        LoadPermutedActivations(a + kk, ap);
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
          const half2 zsub = __half2half2(__ushort2half_rn(static_cast<unsigned short>(1024 + zp[c][j])));
          const half2 zfma = __half2half2(__hneg(__ushort2half_rn(static_cast<unsigned short>(64 + zp[c][j]))));
          float sum = Dot8(w[c][j].x, zsub, zfma, s[c][j], ap);
          sum += Dot8(w[c][j].y, zsub, zfma, s[c][j], ap + 4);
          sum += Dot8(w[c][j].z, zsub, zfma, s[c][j], ap + 8);
          sum += Dot8(w[c][j].w, zsub, zfma, s[c][j], ap + 12);
          acc[j] += sum;
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < kCols; ++j) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[j] += __shfl_xor_sync(0xffffffff, acc[j], offset);
    }
  }
  if (lane < kCols) {
    float v = acc[0];
#pragma unroll
    for (int j = 1; j < kCols; ++j) {
      if (lane == j) {
        v = acc[j];
      }
    }
    output[col0 + lane] = __float2half_rn(v);
  }
}

// Columns per warp: keep at least ~1k warps in flight for latency hiding on 100+ SM parts.
inline int SelectColsPerWarp(int n) {
  if (n >= 4096 && n % 4 == 0) return 4;
  if (n >= 2048 && n % 2 == 0) return 2;
  return 1;
}

// Returns false (without launching) when the shape, block size or alignment is unsupported.
inline bool LaunchMatMul4BitsM1Wide(half* output, const half* a, const uint8_t* b, const half* scales,
                                    const uint8_t* zero_points, int n, int k, int block_size, int cols_per_warp,
                                    cudaStream_t stream) {
  if (n <= 0 || k <= 0 || k % kLaneElems != 0 || k % block_size != 0) {
    return false;
  }
  if (block_size != 32 && block_size != 64 && block_size != 128) {
    return false;
  }
  if ((reinterpret_cast<uintptr_t>(a) & 15) != 0 || (reinterpret_cast<uintptr_t>(b) & 15) != 0) {
    return false;
  }
  const int cols = cols_per_warp > 0 ? cols_per_warp : SelectColsPerWarp(n);
  if (n % cols != 0) {
    return false;
  }
  const int warps = n / cols;
  const dim3 grid((warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
  const dim3 block(kWarpsPerBlock * 32);

#define M1_WIDE_LAUNCH(BS, ZP, C)                                     \
  MatMulFloat4BitsKernelM1Wide<BS, ZP, C><<<grid, block, 0, stream>>>( \
      output, a, b, scales, zero_points, n, k)
#define M1_WIDE_DISPATCH_COLS(BS, ZP) \
  if (cols == 4) {                    \
    M1_WIDE_LAUNCH(BS, ZP, 4);        \
  } else if (cols == 2) {             \
    M1_WIDE_LAUNCH(BS, ZP, 2);        \
  } else if (cols == 1) {             \
    M1_WIDE_LAUNCH(BS, ZP, 1);        \
  } else {                            \
    return false;                     \
  }
#define M1_WIDE_DISPATCH_ZP(BS)        \
  if (zero_points != nullptr) {        \
    M1_WIDE_DISPATCH_COLS(BS, true)    \
  } else {                             \
    M1_WIDE_DISPATCH_COLS(BS, false)   \
  }

  if (block_size == 32) {
    M1_WIDE_DISPATCH_ZP(32)
  } else if (block_size == 64) {
    M1_WIDE_DISPATCH_ZP(64)
  } else {
    M1_WIDE_DISPATCH_ZP(128)
  }
#undef M1_WIDE_DISPATCH_ZP
#undef M1_WIDE_DISPATCH_COLS
#undef M1_WIDE_LAUNCH
  return true;
}

}  // namespace m1_wide
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
