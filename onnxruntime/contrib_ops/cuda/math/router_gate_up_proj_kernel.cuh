// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Skinny (M <= 8) 16-bit projection for a MoE layer that produces the router logits and the shared-expert
// gated activation in one kernel:
//   Y = X[M, K] * W[K, N], N = router + 2 * inter
//   router_out[M, router]  = T(Y[:, :router])
//   act_out[M, inter]      = Activation(T(Y[:, router : router + inter]), T(Y[:, router + inter :]))
// with T bfloat16 or float16 and Activation = SwiGLU.
//
// Tensor-core GEMV. The weights are the MMA "A" operand and the (up to 8) tokens are the "B" columns, so one
// m16n8k16 instruction covers 16 output columns x 8 tokens x 16 K. The MMA is free to sum the 16 K slots in any
// order as long as A and B agree, so slot pairs (2t, 2t + 1) / (2t + 8, 2t + 9) are mapped to rows of W that a lane
// loads with plain 16-byte vector loads (8 consecutive columns of four rows) and packs with byte permutes; no
// shared-memory staging or transposes are needed and every row segment a warp touches is 128 contiguous bytes.
//
// A cluster of kKSplit CTAs splits K and reduces through distributed shared memory in a fixed order, so the
// result is deterministic. All weight loads are issued before griddepcontrol.wait so that, when the launch uses
// programmatic dependent launch, they overlap the tail of the producer kernel.
//
// This header only depends on the CUDA runtime so it can be benchmarked standalone.

#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "contrib_ops/cuda/math/swiglu_activation.cuh"

namespace onnxruntime::contrib::cuda::router_gate_up {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kKSplit = 8;  // CTAs per cluster; also the maximum number of rows (one finalized per rank)
constexpr int kMaxRows = 8;
constexpr int kTilesPerCta = 2;  // 64-column MMA tiles per CTA (router: two router tiles; gated: gate and up)
constexpr int kWarpsPerTile = kWarps / kTilesPerCta;
constexpr int kTileCols = 64;
constexpr int kRouterUnit = kTilesPerCta * kTileCols;
constexpr int kGatedUnit = kTileCols;
constexpr int kKPerIter = 16 * kWarpsPerTile;  // K rows consumed per iteration by the warps of one tile
constexpr int kMaxIters = 5;

struct Params {
  const uint16_t* x;
  const uint16_t* w;
  uint16_t* router_out;
  uint16_t* act_out;
  int rows;
  int k;
  int n;
  int router;
  int inter;
  float limit;
  float alpha;
  float beta;
};

// Shapes the kernel handles; everything else must use the generic path. The kernel also needs `w` aligned to 16 bytes
// and `x` to 4 bytes (vector loads), which the caller checks.
inline bool Supports(int rows, int k, int router, int inter) {
  return rows >= 1 && rows <= kMaxRows && k > 0 && k % (kKSplit * kKPerIter) == 0 &&
         k / kKSplit / kKPerIter <= kMaxIters && router > 0 && router % kRouterUnit == 0 && inter > 0 &&
         inter % kGatedUnit == 0;
}

template <bool kBf16>
__device__ __forceinline__ float RoundToT(float v) {
  if constexpr (kBf16) {
    return __bfloat162float(__float2bfloat16_rn(v));
  } else {
    return __half2float(__float2half_rn(v));
  }
}

template <bool kBf16>
__device__ __forceinline__ uint16_t ToBits(float v) {
  if constexpr (kBf16) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(v));
  } else {
    return __half_as_ushort(__float2half_rn(v));
  }
}

template <bool kBf16>
__device__ __forceinline__ void Mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  if constexpr (kBf16) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
// Clusters and programmatic dependent launch need sm_90. Device code generated from older PTX, including a JIT for a
// newer device, lands here, so the host launches this kernel only after checking the loaded image was built for sm_90+.
template <int NI, bool kBf16>
__global__ void RouterGateUpProjKernel(Params) {}
#else
template <int NI, bool kBf16>
__global__ void __launch_bounds__(kThreads, 1) __cluster_dims__(kKSplit, 1, 1) RouterGateUpProjKernel(Params p) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int unit = blockIdx.x / kKSplit;
  const int router_units = p.router / kRouterUnit;
  const bool is_router = unit < router_units;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ti = warp / kWarpsPerTile;
  const int wq = warp % kWarpsPerTile;
  const int g = lane >> 2;
  const int t = lane & 3;

  int tile_col;
  if (is_router) {
    tile_col = unit * kRouterUnit + ti * kTileCols;
  } else {
    tile_col = p.router + ti * p.inter + (unit - router_units) * kGatedUnit;
  }

  const int kc = p.k / kKSplit;
  const int k0 = rank * kc;
  const int col0 = tile_col + g * 8;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif

  uint4 wv[NI][4];
#pragma unroll
  for (int it = 0; it < NI; ++it) {
    const int r0 = k0 + (it * kWarpsPerTile + wq) * 16 + 2 * t;
    const uint16_t* base = p.w + static_cast<size_t>(r0) * p.n + col0;
    wv[it][0] = __ldg(reinterpret_cast<const uint4*>(base));
    wv[it][1] = __ldg(reinterpret_cast<const uint4*>(base + p.n));
    wv[it][2] = __ldg(reinterpret_cast<const uint4*>(base + 8 * static_cast<size_t>(p.n)));
    wv[it][3] = __ldg(reinterpret_cast<const uint4*>(base + 9 * static_cast<size_t>(p.n)));
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

  uint32_t xb[NI][2];
#pragma unroll
  for (int it = 0; it < NI; ++it) {
    const int kb = k0 + (it * kWarpsPerTile + wq) * 16 + 2 * t;
    uint32_t b0 = 0, b1 = 0;
    if (g < p.rows) {
      const uint16_t* xr = p.x + static_cast<size_t>(g) * p.k + kb;
      b0 = *reinterpret_cast<const uint32_t*>(xr);
      b1 = *reinterpret_cast<const uint32_t*>(xr + 8);
    }
    xb[it][0] = b0;
    xb[it][1] = b1;
  }

  float acc[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int c = 0; c < 4; ++c) acc[j][c] = 0.0f;
  }

#pragma unroll
  for (int it = 0; it < NI; ++it) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t v0 = (&wv[it][0].x)[j];
      const uint32_t v1 = (&wv[it][1].x)[j];
      const uint32_t v2 = (&wv[it][2].x)[j];
      const uint32_t v3 = (&wv[it][3].x)[j];
      const uint32_t a[4] = {__byte_perm(v0, v1, 0x5410), __byte_perm(v0, v1, 0x7632), __byte_perm(v2, v3, 0x5410),
                             __byte_perm(v2, v3, 0x7632)};
      Mma16816<kBf16>(acc[j], a, xb[it][0], xb[it][1]);
    }
  }

  __shared__ float red[kTilesPerCta][kWarpsPerTile][kMaxRows][kTileCols];

  // acc[j][0..1]: column g * 8 + 2j, tokens 2t / 2t + 1; acc[j][2..3]: column g * 8 + 2j + 1.
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = g * 8 + 2 * j;
    red[ti][wq][2 * t][c] = acc[j][0];
    red[ti][wq][2 * t + 1][c] = acc[j][1];
    red[ti][wq][2 * t][c + 1] = acc[j][2];
    red[ti][wq][2 * t + 1][c + 1] = acc[j][3];
  }
  cluster.sync();

  // Rank r finalizes token r: it sums the K slices of every warp of every CTA in the cluster in a fixed order.
  const int tok = rank;
  if (tok < p.rows) {
    const float* parts[kKSplit];
#pragma unroll
    for (int r = 0; r < kKSplit; ++r) parts[r] = cluster.map_shared_rank(&red[0][0][0][0], r);
    auto slice = [&](int tile, int c) {
      float s = 0.0f;
#pragma unroll
      for (int r = 0; r < kKSplit; ++r) {
#pragma unroll
        for (int q = 0; q < kWarpsPerTile; ++q) {
          s += parts[r][((tile * kWarpsPerTile + q) * kMaxRows + tok) * kTileCols + c];
        }
      }
      return s;
    };
    if (is_router) {
      if (tid < kRouterUnit) {
        const float s = slice(tid / kTileCols, tid % kTileCols);
        p.router_out[static_cast<size_t>(tok) * p.router + unit * kRouterUnit + tid] = ToBits<kBf16>(s);
      }
    } else if (tid < kGatedUnit) {
      const float gate = slice(0, tid);
      const float up = slice(1, tid);
      const float val =
          SwiGluActivation(RoundToT<kBf16>(gate), RoundToT<kBf16>(up), p.limit, p.alpha, p.beta);
      p.act_out[static_cast<size_t>(tok) * p.inter + (unit - router_units) * kGatedUnit + tid] = ToBits<kBf16>(val);
    }
  }
  cluster.sync();
}
#endif

template <int NI, bool kBf16>
inline cudaError_t LaunchIters(const Params& p, cudaStream_t stream, bool use_pdl) {
  const int units = p.router / kRouterUnit + p.inter / kGatedUnit;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(units * kKSplit, 1, 1);
  config.blockDim = dim3(kThreads, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = use_pdl ? 1 : 0;
  return cudaLaunchKernelEx(&config, RouterGateUpProjKernel<NI, kBf16>, p);
}

template <bool kBf16>
inline cudaError_t LaunchT(const Params& p, cudaStream_t stream, bool use_pdl) {
  switch (p.k / kKSplit / kKPerIter) {
    case 1:
      return LaunchIters<1, kBf16>(p, stream, use_pdl);
    case 2:
      return LaunchIters<2, kBf16>(p, stream, use_pdl);
    case 3:
      return LaunchIters<3, kBf16>(p, stream, use_pdl);
    case 4:
      return LaunchIters<4, kBf16>(p, stream, use_pdl);
    default:
      return LaunchIters<5, kBf16>(p, stream, use_pdl);
  }
}

inline cudaError_t Launch(const Params& p, cudaStream_t stream, bool use_pdl, bool is_bf16) {
  return is_bf16 ? LaunchT<true>(p, stream, use_pdl) : LaunchT<false>(p, stream, use_pdl);
}

}  // namespace onnxruntime::contrib::cuda::router_gate_up
