// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/moe_packed_impl.cuh"
#include "contrib_ops/cuda/quantization/packed_projection_impl.cuh"

namespace onnxruntime::contrib::cuda {
namespace {
using namespace packed_projection;

// First-occurrence grouping keeps one MMA tile's columns in token-row order.
__global__ void MoeGroupKernel(const int32_t* idx, int P, int topk, int experts, int32_t* meta,
                               int intermediate, bf16* hidden, float* part) {
  __shared__ int sidx[128], srank[128], warp_tot[4], invalid_rows[8];
  const int p = threadIdx.x, lane = p & 31, warp = p >> 5;
#pragma unroll 5
  for (int i = p; i < P * 8; i += blockDim.x) meta[1 + P + i] = -1;
  if (p < P / topk) invalid_rows[p] = 0;
  if (p < P) {
    sidx[p] = idx[p];
    meta[1 + p] = -1;
  }
  __syncthreads();
  bool invalid = false;
  if (p < P) {
    const int row = p / topk, my = sidx[p];
    invalid = my < 0 || my >= experts;
    for (int j = row * topk; j < p; ++j)
      invalid = invalid || sidx[j] == my;
    if (invalid) atomicOr(invalid_rows + row, 1);
  }
  // The vote is block-uniform; duplicate/out-of-range IDs poison only their token row.
  if (__syncthreads_or(invalid)) {
    if (p < P && invalid_rows[p / topk]) sidx[p] = -1;
    for (int row = 0; row < P / topk; ++row) {
      if (!invalid_rows[row]) continue;
      for (int k = 0; k < topk; ++k) {
        const int pair = row * topk + k;
        for (int i = p; i < intermediate; i += blockDim.x)
          hidden[static_cast<int64_t>(pair) * intermediate + i] = __float2bfloat16(NAN);
        for (int i = p; i < 1024; i += blockDim.x)
          part[static_cast<int64_t>(pair) * 1024 + i] = NAN;
      }
    }
    __syncthreads();
  }
  int q = -1;
  if (p < P && sidx[p] >= 0) {
    const int my = sidx[p];
    for (int j = 0; j <= p; ++j) {
      if (sidx[j] == my) {
        q = j;
        break;
      }
    }
  }
  const bool first = q >= 0 && q == p;
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
  if (q >= 0) meta[1 + P + srank[q] * 8 + p / topk] = p;
}

template <int PH, int KW, bool X_ALIGNED = false>
__global__ void __launch_bounds__(KW * 32)
    MoeMmaKernel(const bf16* xin, const int32_t* meta, int P, int R, int Nrows, int K,
                 const uint8_t* w, void* out) {
  constexpr int FMT = kFmtIq2Gguf, MT = 2;
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
        // Embedded 66-byte blocks cannot inherit the activation's vector alignment.
        LoadWeightRow<FMT, false, false>(w, nullptr, row, K, c, t, st.raw[m][h], st.sc[m][h]);
      }
    }
    LoadX<1, X_ALIGNED>(xr, xvalid, c, t, st.x);
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

__device__ __forceinline__ float4 LoadExpertOutput(const float* ptr) {
  if ((reinterpret_cast<uintptr_t>(ptr) & 15) == 0)
    return __ldg(reinterpret_cast<const float4*>(ptr));
  return make_float4(__ldg(ptr), __ldg(ptr + 1), __ldg(ptr + 2), __ldg(ptr + 3));
}

__global__ void MoeFinalizeKernel(const float* part, const float* w, int topk, bf16* out) {
  const int row = blockIdx.x, vec_idx = blockIdx.y * blockDim.x + threadIdx.x;
  if (vec_idx >= 256) return;
  const float* wt = w + row * topk;
  const float* pt = part + static_cast<int64_t>(row) * topk * 1024 + vec_idx * 4;
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  if (topk == 9) {
#pragma unroll
    for (int k = 0; k < 9; ++k) {
      const float wk = wt[k];
      const float4 pv = LoadExpertOutput(pt + k * 1024);
      a0 += wk * pv.x;
      a1 += wk * pv.y;
      a2 += wk * pv.z;
      a3 += wk * pv.w;
    }
  } else {
    for (int k = 0; k < topk; ++k) {
      const float wk = wt[k];
      const float4 pv = LoadExpertOutput(pt + k * 1024);
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
  bf16* dst = out + static_cast<int64_t>(row) * 1024 + vec_idx * 4;
  if ((reinterpret_cast<uintptr_t>(dst) & 7) == 0) {
    *reinterpret_cast<uint2*>(dst) = out2;
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = packed[i];
  }
}

__device__ __forceinline__ uint4 LoadRouterVector(const bf16* source) {
  if ((reinterpret_cast<uintptr_t>(source) & (alignof(uint4) - 1)) == 0)
    return *reinterpret_cast<const uint4*>(source);
  uint4 result;
  bf16* values = reinterpret_cast<bf16*>(&result);
#pragma unroll
  for (int i = 0; i < 8; ++i) values[i] = source[i];
  return result;
}

__global__ void RouterLogitsKernel(const bf16* x, const bf16* w, int K, int E, float* scores) {
  const int r = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = blockIdx.y * 8 + warp;
  if (e >= E) return;
  const bf16* xr = x + static_cast<int64_t>(r) * K;
  const bf16* wr = w + static_cast<int64_t>(e) * K;
  float acc = 0.f;
  for (int k = lane * 8; k < K; k += 256) {
    const uint4 xv = LoadRouterVector(xr + k), wv = LoadRouterVector(wr + k);
    const bf16* xp = reinterpret_cast<const bf16*>(&xv);
    const bf16* wp = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc += B2F(xp[i]) * B2F(wp[i]);
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffff, acc, o);
  if (lane == 0) scores[static_cast<int64_t>(r) * E + e] = 1.0f / (1.0f + expf(-acc));
}

__device__ __forceinline__ uint32_t RouterScoreRank(float value) {
  const uint32_t bits = __float_as_uint(value), magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return 1u;
  if (magnitude == 0) return 0x80000000u;
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

template <typename T>
__device__ __forceinline__ void RouterSelectTopk(T (&values)[16], const float (&scores)[16],
                                                 int topk, float scale, T empty,
                                                 int32_t* indices, float* weights) {
  const int lane = threadIdx.x;
  int my_idx = -1;
  float my_sc = 0.f, sum = 0.f;
  for (int j = 0; j < topk; ++j) {
    T bv = empty;
    float bsc = 0.f;
    int bi = -1;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (values[i] > bv) {
        bv = values[i];
        bsc = scores[i];
        bi = lane + 32 * i;
      }
    }
    T wv = bv;
    int wi = bi;
    for (int o = 16; o > 0; o >>= 1) {
      const T ov = __shfl_xor_sync(0xffffffff, wv, o);
      const int oi = __shfl_xor_sync(0xffffffff, wi, o);
      if (ov > wv || (ov == wv && oi >= 0 && (wi < 0 || oi < wi))) {
        wv = ov;
        wi = oi;
      }
    }
    const int win_lane = wi & 31, win_slot = wi >> 5;
    if (wi >= 0 && win_lane == lane) {
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if (i == win_slot) values[i] = empty;
      }
    }
    const float s = wi >= 0 ? __shfl_sync(0xffffffff, bsc, win_lane) : 0.f;
    if (lane == j) {
      my_idx = wi;
      my_sc = s;
    }
    sum += s;
  }
  if (lane < topk) {
    indices[lane] = my_idx;
    weights[lane] = my_sc / (sum + 1e-20f) * scale;
  }
}

__global__ void RouterTopkKernel(const float* scores, const float* bias, int E, int topk, float scale,
                                 int32_t* idx_out, float* w_out) {
  const int r = blockIdx.x, lane = threadIdx.x;
  float values[16], sc[16];
  bool finite = true;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int e = lane + 32 * i;
    sc[i] = e < E ? scores[static_cast<int64_t>(r) * E + e] : 0.f;
    values[i] = e < E ? sc[i] + bias[e] : -INFINITY;
    finite = finite && (e >= E || isfinite(values[i]));
  }
  int32_t* indices = idx_out + static_cast<int64_t>(r) * topk;
  float* weights = w_out + static_cast<int64_t>(r) * topk;
  if (__all_sync(0xffffffff, finite)) {
    RouterSelectTopk(values, sc, topk, scale, -INFINITY, indices, weights);
    return;
  }
  // Keep NaNs, numeric infinities and removed slots distinct; signed zeros tie.
  uint32_t ranks[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    ranks[i] = lane + 32 * i < E ? RouterScoreRank(values[i]) : 0u;
  RouterSelectTopk(ranks, sc, topk, scale, 0u, indices, weights);
}
}  // namespace

cudaError_t LaunchMoEPacked(cudaStream_t stream, int rows, int topk, int experts, int intermediate,
                            const void* input, const int32_t* indices, const float* weights,
                            const uint8_t* fc1, const uint8_t* fc2, void* output, void* hidden,
                            float* expert_output, int32_t* metadata) {
  if (rows < 1 || rows > 8 || topk < 1 || topk > 32 || rows * topk > 128 ||
      experts < topk || intermediate <= 0 || intermediate % 256 != 0 ||
      !input || !indices || !weights || !fc1 || !fc2 || !output || !hidden || !expert_output || !metadata)
    return cudaErrorInvalidValue;
  const int pairs = rows * topk;
  MoeGroupKernel<<<1, 128, 0, stream>>>(indices, pairs, topk, experts, metadata, intermediate,
                                        static_cast<bf16*>(hidden), expert_output);
  MoeMmaKernel<1, 4><<<dim3(intermediate / 32, pairs), 128, 4 * 2 * 16 * 8 * sizeof(float), stream>>>(
      static_cast<const bf16*>(input), metadata, pairs, rows, intermediate, 1024, fc1, hidden);
  const bool aligned_hidden = intermediate % 8 == 0 && (reinterpret_cast<uintptr_t>(hidden) & 15) == 0;
  if (aligned_hidden) {
    MoeMmaKernel<2, 8, true><<<dim3(1024 / 32, pairs), 256, 8 * 2 * 16 * 8 * sizeof(float), stream>>>(
        static_cast<const bf16*>(hidden), metadata, pairs, rows, 1024, intermediate, fc2, expert_output);
  } else {
    MoeMmaKernel<2, 8><<<dim3(1024 / 32, pairs), 256, 8 * 2 * 16 * 8 * sizeof(float), stream>>>(
        static_cast<const bf16*>(hidden), metadata, pairs, rows, 1024, intermediate, fc2, expert_output);
  }
  MoeFinalizeKernel<<<dim3(rows, 4), 64, 0, stream>>>(expert_output, weights, topk, static_cast<bf16*>(output));
  return cudaGetLastError();
}

cudaError_t LaunchMoEPackedRouter(cudaStream_t stream, int rows, int width, int experts, int topk,
                                  float scale, const void* input, const void* weight, const float* bias,
                                  int32_t* indices, float* weights, float* scores) {
  if (rows < 1 || rows > 8 || width <= 0 || width % 256 != 0 || experts < 1 || experts > 512 ||
      topk < 1 || topk > 32 || topk > experts || rows * topk > 128 || !isfinite(scale) ||
      !input || !weight || !bias || !indices || !weights || !scores)
    return cudaErrorInvalidValue;
  RouterLogitsKernel<<<dim3(rows, (experts + 7) / 8), 256, 0, stream>>>(
      static_cast<const bf16*>(input), static_cast<const bf16*>(weight), width, experts, scores);
  RouterTopkKernel<<<rows, 32, 0, stream>>>(scores, bias, experts, topk, scale, indices, weights);
  return cudaGetLastError();
}
}  // namespace onnxruntime::contrib::cuda
