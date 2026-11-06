// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_packed_projection.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <algorithm>
#include "contrib_ops/cuda/quantization/packed_projection_impl.cuh"

namespace onnxruntime::contrib::cuda {
namespace {

using namespace packed_projection;

// Mechanically transplanted packed-projection kernels; see packed_projection.md
// for the three source snapshots and their distinct rounding contracts.
// x is pre-scaled by 2^-6 so that fp16 partial sums of 16 products cannot overflow for |x| < ~4e4 (massive
// activations).
constexpr float kXScale = 1.0f / 64.0f;

// A warp owns ROWS weight rows; lane l handles 16-byte (32-value) chunks l + 32*it. All weight loads are issued up
// front.
template <int R, int ROWS, int IT>
__global__ void Nvfp4GemvKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ w,
                                const uint8_t* __restrict__ ws, const float* __restrict__ gscale, int N, int K,
                                bf16* __restrict__ y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
  if (n0 >= N) return;
  const int kbytes = K / 2;
  const int kscale = K / 16;
  uint4 wv[ROWS][IT];
  uint32_t sc[ROWS][IT];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int n = min(n0 + i, N - 1);
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int cb = it * 512 + lane * 16;
      wv[i][it] = *reinterpret_cast<const uint4*>(w + static_cast<int64_t>(n) * kbytes + cb);
      sc[i][it] = *reinterpret_cast<const uint16_t*>(ws + static_cast<int64_t>(n) * kscale + (cb >> 3));
    }
  }
  float acc[ROWS][R];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
#pragma unroll
  for (int it = 0; it < IT; ++it) {
    const int k0 = (it * 512 + lane * 16) * 2;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      __half2 xp[16];
      const uint4* xq = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(r) * K + k0);
#pragma unroll
      for (int g = 0; g < 4; ++g) {
        const uint4 v = xq[g];
        const bf16* b = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int p = 0; p < 4; ++p) xp[g * 4 + p] = __floats2half2_rn(B2F(b[p]) * kXScale, B2F(b[p + 4]) * kXScale);
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        const uint32_t words[4] = {wv[i][it].x, wv[i][it].y, wv[i][it].z, wv[i][it].w};
        float tot = 0.f;
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          __half2 a = __float2half2_rn(0.f);
#pragma unroll
          for (int wd = 0; wd < 2; ++wd) {
            const uint32_t word = words[blk * 2 + wd];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const uint32_t pb = Fp4PairPattern(word >> (4 * p));
              __half2 pat = *reinterpret_cast<const __half2*>(&pb);
              pat = __hmul2(pat, __float2half2_rn(16384.f));
              a = __hfma2(pat, xp[(blk * 2 + wd) * 4 + p], a);
            }
          }
          const float f = (__low2float(a) + __high2float(a)) * (1.0f / kXScale);
          tot += f * Fp8E4M3ToFloat((sc[i][it] >> (8 * blk)) & 0xFFu);
        }
        acc[i][r] += tot;
      }
    }
  }
  const float g = gscale[0];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = WarpSum(acc[i][r]);
      if (lane == 0 && n0 + i < N) y[static_cast<int64_t>(r) * N + n0 + i] = __float2bfloat16(v * g);
    }
#else
  // ComputeInternal rejects these devices; default CUDA builds still compile their targets.
  asm volatile("trap;");
#endif
}

template <int R, int IT>
void LaunchNvfp4Impl(const bf16* x, const uint8_t* w, const uint8_t* ws, const float* g, int N, int K, bf16* y,
                     cudaStream_t s) {
  if (N <= 8192) {
    constexpr int ROWS = 1, WARPS = 4;
    Nvfp4GemvKernel<R, ROWS, IT><<<(N + WARPS - 1) / WARPS, WARPS * 32, 0, s>>>(x, w, ws, g, N, K, y);
  } else {
    constexpr int ROWS = 2, WARPS = 8;
    const int per_block = ROWS * WARPS;
    Nvfp4GemvKernel<R, ROWS, IT><<<(N + per_block - 1) / per_block, WARPS * 32, 0, s>>>(x, w, ws, g, N, K, y);
  }
}
template <int R>
void LaunchNvfp4(const bf16* x, const uint8_t* w, const uint8_t* ws, const float* g, int N, int K, bf16* y,
                 cudaStream_t s) {
  if (K == 4096)
    LaunchNvfp4Impl<R, 4>(x, w, ws, g, N, K, y, s);
  else
    LaunchNvfp4Impl<R, 8>(x, w, ws, g, N, K, y, s);
}

template <int FMT, bool WIDE, int MT, int KW, int UNR, bool ALIGNED_X = false>
__global__ void __launch_bounds__(KW * 32)
    MmaGemvKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ w, const void* __restrict__ sc,
                  const float* __restrict__ gscale, int R, int N, int K, bf16* __restrict__ y) {
  constexpr int NH = WIDE ? 2 : 1;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nchunks = K / (64 * NH);
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * (MT * 16);
  const bf16* xr = x + static_cast<int64_t>(min(g, R - 1)) * K;
  const bool xvalid = g < R;
  float acc[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[m][i] = 0.f;
  MmaMainLoop<FMT, WIDE, MT, KW, UNR>(warp, nchunks, acc, [&](int c, StageT<FMT, WIDE, MT>& st) {
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t row = min(n_base + m * 16 + g + 8 * h, static_cast<int64_t>(N - 1));
        LoadWeightRow<FMT, WIDE>(w, sc, row, K, c, t, st.raw[m][h], st.sc[m][h]);
      }
    if constexpr (ALIGNED_X)
      LoadAlignedX<NH>(xr, xvalid, c, t, st.x);
    else
      LoadX<NH>(xr, xvalid, c, t, st.x);
  });
  const float gs = (gscale ? gscale[0] : 1.0f) * (FMT == kFmtFp4 ? 0x1p16f : 1.0f);
  ReduceEpilogue<MT, KW>(acc, warp, g, t, [&](int rl, int col, float v) {
    if (col < R && n_base + rl < N) y[static_cast<int64_t>(col) * N + n_base + rl] = __float2bfloat16(v * gs);
  });
}

template <int FMT, bool WIDE, int MT, int KW, int UNR = 1>
void LaunchMmaGemvImpl(const bf16* x, const uint8_t* w, const void* sc, const float* gs, int R, int N, int K, bf16* y,
                       cudaStream_t s) {
  const int blocks = (N + MT * 16 - 1) / (MT * 16);
  if constexpr (FMT == kFmtFp4) {
    if (K % 8 == 0 && (reinterpret_cast<uintptr_t>(x) & 15) == 0) {
      MmaGemvKernel<FMT, WIDE, MT, KW, UNR, true>
          <<<blocks, KW * 32, KW * MT * 16 * 8 * sizeof(float), s>>>(x, w, sc, gs, R, N, K, y);
      return;
    }
  }
  MmaGemvKernel<FMT, WIDE, MT, KW, UNR>
      <<<blocks, KW * 32, KW * MT * 16 * 8 * sizeof(float), s>>>(x, w, sc, gs, R, N, K, y);
}

template <int FMT>
constexpr bool kCanWide = FMT != kFmtQ8 && FMT != kFmtQ8F16 && FMT != kFmtIq2N;

template <int FMT>
void LaunchMmaGemv(const bf16* x, const uint8_t* w, const void* sc, const float* gs, int R, int N, int K, bf16* y,
                   cudaStream_t s) {
  // Keep the source default reduction topology independent of process tuning variables.
  const bool wide = kCanWide<FMT> && K % 128 == 0;
  if (N >= 8192) {
    if (wide)
      LaunchMmaGemvImpl<FMT, kCanWide<FMT>, 2, 4>(x, w, sc, gs, R, N, K, y, s);
    else
      LaunchMmaGemvImpl<FMT, false, 2, 4>(x, w, sc, gs, R, N, K, y, s);
  } else {
    if (wide)
      LaunchMmaGemvImpl<FMT, kCanWide<FMT>, 1, 16>(x, w, sc, gs, R, N, K, y, s);
    else
      LaunchMmaGemvImpl<FMT, false, 1, 16>(x, w, sc, gs, R, N, K, y, s);
  }
}

__device__ __forceinline__ float DotU8x16(const uint4 wv, const float* x) {
  const uint32_t w[4] = {wv.x, wv.y, wv.z, wv.w};
  float acc = 0.f;
#pragma unroll
  for (int q = 0; q < 4; ++q)
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (w[q] >> (8 * b)) & 0xFFu;
      acc += (__uint_as_float(0x4B000000u | byte) - 8388608.0f - 128.0f) * x[q * 4 + b];
    }
  return acc;
}

template <int BITS, int R, int ROWS, int IT>
__global__ void QGemvKernel(const bf16* __restrict__ x, const uint8_t* __restrict__ w, const bf16* __restrict__ sc,
                            int N, int K, bf16* __restrict__ y) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
  if (n0 >= N) return;
  const int nb = K / 32;
  constexpr int CH = BITS == 8 ? 2 : 1;  // uint4 per 32-value block
  uint4 wv[ROWS][IT][CH];
  float sv[ROWS][IT];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const int n = min(n0 + i, N - 1);
#pragma unroll
    for (int it = 0; it < IT; ++it) {
      const int b = lane + 32 * it;
      if (b < nb) {
        const uint4* wp = reinterpret_cast<const uint4*>(w + (static_cast<int64_t>(n) * nb + b) * (BITS * 4));
#pragma unroll
        for (int c = 0; c < CH; ++c) wv[i][it][c] = wp[c];
        sv[i][it] = B2F(sc[static_cast<int64_t>(n) * nb + b]);
      } else {
        sv[i][it] = 0.f;
#pragma unroll
        for (int c = 0; c < CH; ++c) wv[i][it][c] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  float acc[ROWS][R];
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) acc[i][r] = 0.f;
#pragma unroll
  for (int it = 0; it < IT; ++it) {
    const int b = lane + 32 * it;
    if (b >= nb) continue;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float xf[32];
      const uint4* xq = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(r) * K + b * 32);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const uint4 v = xq[q];
        const bf16* bb = reinterpret_cast<const bf16*>(&v);
#pragma unroll
        for (int i = 0; i < 8; ++i) xf[q * 8 + i] = B2F(bb[i]);
      }
#pragma unroll
      for (int i = 0; i < ROWS; ++i) {
        float d;
        if constexpr (BITS == 4)
          d = DotInt4x32(wv[i][it][0], xf);
        else
          d = DotU8x16(wv[i][it][0], xf) + DotU8x16(wv[i][it][1], xf + 16);
        acc[i][r] += d * sv[i][it];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < ROWS; ++i)
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float v = WarpSum(acc[i][r]);
      if (lane == 0 && n0 + i < N) y[static_cast<int64_t>(r) * N + n0 + i] = __float2bfloat16(v);
    }
}

template <int BITS, int R, int IT>
void LaunchQGemvImpl(const bf16* x, const uint8_t* w, const bf16* sc, int N, int K, bf16* y, cudaStream_t s) {
  if (N <= 4096) {  // small N: one row per warp to fill the GPU
    constexpr int ROWS = 1, WARPS = 4;
    QGemvKernel<BITS, R, ROWS, IT><<<(N + WARPS - 1) / WARPS, WARPS * 32, 0, s>>>(x, w, sc, N, K, y);
  } else {
    constexpr int ROWS = 2, WARPS = 8;
    const int per_block = ROWS * WARPS;
    QGemvKernel<BITS, R, ROWS, IT><<<(N + per_block - 1) / per_block, WARPS * 32, 0, s>>>(x, w, sc, N, K, y);
  }
}
template <int BITS, int R>
void LaunchQGemvR(const bf16* x, const uint8_t* w, const bf16* sc, int N, int K, bf16* y, cudaStream_t s) {
  const int nb = K / 32;
  if (nb <= 32)
    LaunchQGemvImpl<BITS, R, 1>(x, w, sc, N, K, y, s);
  else if (nb <= 64)
    LaunchQGemvImpl<BITS, R, 2>(x, w, sc, N, K, y, s);
  else if (nb <= 128)
    LaunchQGemvImpl<BITS, R, 4>(x, w, sc, N, K, y, s);
  else
    LaunchQGemvImpl<BITS, R, 6>(x, w, sc, N, K, y, s);
}

}  // namespace

cudaError_t LaunchPackedProjection(PackedProjectionFormat format, const void* a, const uint8_t* b, const void* scales,
                                   const float* global_scale, bool fp16_scales, int rows, int n, int k, void* y,
                                   cudaStream_t stream) {
  if (rows < 1 || rows > 8 || n <= 0 || k <= 0 || k % 64 != 0) {
    return cudaErrorInvalidValue;
  }
  const auto* x = static_cast<const bf16*>(a);
  auto* output = static_cast<bf16*>(y);
  switch (format) {
    case PackedProjectionFormat::NVFP4:
      if (global_scale == nullptr || (rows == 1 && k != 4096 && k != 8192)) {
        return cudaErrorInvalidValue;
      }
      if (rows == 1) {
        LaunchNvfp4<1>(x, b, static_cast<const uint8_t*>(scales), global_scale, n, k, output, stream);
      } else {
        LaunchMmaGemv<kFmtFp4>(x, b, scales, global_scale, rows, n, k, output, stream);
      }
      break;
    case PackedProjectionFormat::Q8:
      if (fp16_scales) {
        LaunchMmaGemv<kFmtQ8F16>(x, b, scales, nullptr, rows, n, k, output, stream);
      } else if (rows == 1) {
        if (k > 6144) return cudaErrorInvalidValue;
        LaunchQGemvR<8, 1>(x, b, static_cast<const bf16*>(scales), n, k, output, stream);
      } else {
        LaunchMmaGemv<kFmtQ8>(x, b, scales, nullptr, rows, n, k, output, stream);
      }
      break;
    case PackedProjectionFormat::IQ2XXS:
      if (k % 256 != 0) return cudaErrorInvalidValue;
      LaunchMmaGemv<kFmtIq2N>(x, b, scales, nullptr, rows, n, k, output, stream);
      break;
  }
  return cudaGetLastError();
}

}  // namespace onnxruntime::contrib::cuda
