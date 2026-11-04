// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/packed_moe_impl.h"
#include "contrib_ops/cuda/quantization/packed_projection_impl.cuh"

namespace onnxruntime::contrib::cuda {
namespace {
using namespace packed_projection;

// Group in first-occurrence order; one expert tile serves up to eight token rows.
__global__ void MoeGroupKernel(const int32_t* idx, int P, int topk, int32_t* meta) {
  __shared__ int sidx[128], srank[128], warp_tot[4];
  const int p = threadIdx.x, lane = p & 31, warp = p >> 5;
#pragma unroll 5
  for (int i = p; i < P * 8; i += blockDim.x) meta[1 + P + i] = -1;
  if (p < P) {
    sidx[p] = idx[p];
    meta[1 + p] = -1;
  }
  __syncthreads();
  int q = -1;
  if (p < P) {
    const int my = sidx[p];
    for (int j = 0; j <= p; ++j) {
      if (sidx[j] == my) {
        q = j;
        break;
      }
    }
  }
  const bool first = p < P && q == p;
  const unsigned b = __ballot_sync(0xffffffffu, first);
  if (lane == 0) warp_tot[warp] = __popc(b);
  __syncthreads();
  int off = 0;
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    if (w < warp) off += warp_tot[w];
  }
  if (first) {
    srank[p] = off + __popc(b & ((1u << lane) - 1u));
    meta[1 + srank[p]] = sidx[p];
  }
  if (p == 0) meta[0] = warp_tot[0] + warp_tot[1] + warp_tot[2] + warp_tot[3];
  __syncthreads();
  if (p < P) meta[1 + P + srank[q] * 8 + p / topk] = p;
}

template <int PH, int KW>
__global__ void __launch_bounds__(KW * 32)
    MoeMmaKernel(const bf16* xin, const int32_t* meta, int P, int R, int Nrows, int K,
                 const uint8_t* w, const bf16* sc, void* out) {
  constexpr int FMT = kFmtIq2N, MT = 2;
  const int u = blockIdx.y;
  if (u >= meta[0]) return;
  const int64_t e = meta[1 + u];
  const int32_t* pair_of = meta + 1 + P + u * 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3, nchunks = K / 64;
  const int n_base = blockIdx.x * (MT * 16), xpair = pair_of[g];
  const bf16* xr = xin;
  bool xvalid;
  if constexpr (PH == 1) {
    xvalid = g < R;
    xr += static_cast<int64_t>(min(g, R - 1)) * K;
  } else {
    xvalid = xpair >= 0;
    xr += static_cast<int64_t>(max(xpair, 0)) * K;
  }
  float acc[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[m][i] = 0.f;
  }
  MmaMainLoop<FMT, false, MT, KW, 1>(warp, nchunks, acc, [&](int c, StageT<FMT, false, MT>& st) {
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t row = e * Nrows + min(n_base + m * 16 + g + 8 * h, Nrows - 1);
        LoadWeightRow<FMT, false>(w, sc, row, K, c, t, st.raw[m][h], st.sc[m][h]);
      }
    }
    LoadX<1>(xr, xvalid, c, t, st.x);
  });
  ReduceEpilogue<MT, KW>(acc, warp, g, t, [&](int rl, int col, float v) {
    const int j = n_base + rl;
    if (j >= Nrows) return;
    const int pair = pair_of[col];
    if (pair < 0) return;
    v = B2F(__float2bfloat16(v));
    if constexpr (PH == 1) {
      const float r = fmaxf(v, 0.f);
      reinterpret_cast<bf16*>(out)[static_cast<int64_t>(pair) * Nrows + j] = __float2bfloat16(r * r);
    } else {
      reinterpret_cast<float*>(out)[static_cast<int64_t>(pair) * Nrows + j] = v;
    }
  });
}

__global__ void MoeFinalizeKernel(const float* part, const float* w, int topk, bf16* out) {
  const int row = blockIdx.x, vec_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (vec_idx >= 256) return;
  const float* wt = w + row * topk;
  const float4* pt = reinterpret_cast<const float4*>(part + static_cast<int64_t>(row) * topk * 1024) + vec_idx;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  if (topk == 9) {
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const float wk = wt[k];
      const float4 pv = __ldg(pt + k * 256);
      a0 += wk * pv.x;
      a1 += wk * pv.y;
      a2 += wk * pv.z;
      a3 += wk * pv.w;
    }
  } else {
    for (int k = 0; k < topk; ++k) {
      const float wk = wt[k];
      const float4 pv = __ldg(pt + k * 256);
      a0 += wk * pv.x;
      a1 += wk * pv.y;
      a2 += wk * pv.z;
      a3 += wk * pv.w;
    }
  }
  uint2 out2;
  bf16* packed = reinterpret_cast<bf16*>(&out2);
  packed[0] = __float2bfloat16(a0);
  packed[1] = __float2bfloat16(a1);
  packed[2] = __float2bfloat16(a2);
  packed[3] = __float2bfloat16(a3);
  reinterpret_cast<uint2*>(out + static_cast<int64_t>(row) * 1024)[vec_idx] = out2;
}
}  // namespace

cudaError_t LaunchMoEPackedIQ2XXS(cudaStream_t stream, int rows, int topk, int intermediate_size,
                                  const void* input, const int32_t* indices, const float* weights,
                                  const uint8_t* fc1, const void* scale1, const uint8_t* fc2, const void* scale2,
                                  void* output, void* hidden, float* expert_output, int32_t* metadata) {
  const int pairs = rows * topk;
  MoeGroupKernel<<<1, 128, 0, stream>>>(indices, pairs, topk, metadata);
  // Frozen default: FC1 uses 2 MMA tiles/4 warps, FC2 uses 2 tiles/8 warps, narrow native IQ2 chunks.
  MoeMmaKernel<1, 4><<<dim3(intermediate_size / 32, pairs), 128, 4 * 2 * 16 * 8 * sizeof(float), stream>>>(
      static_cast<const bf16*>(input), metadata, pairs, rows, intermediate_size, 1024,
      fc1, static_cast<const bf16*>(scale1), hidden);
  MoeMmaKernel<2, 8><<<dim3(1024 / 32, pairs), 256, 8 * 2 * 16 * 8 * sizeof(float), stream>>>(
      static_cast<const bf16*>(hidden), metadata, pairs, rows, 1024, intermediate_size,
      fc2, static_cast<const bf16*>(scale2), expert_output);
  MoeFinalizeKernel<<<dim3(rows, 4), 64, 0, stream>>>(expert_output, weights, topk, static_cast<bf16*>(output));
  return cudaGetLastError();
}
}  // namespace onnxruntime::contrib::cuda
