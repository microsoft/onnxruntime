// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sequence/fused_state_space_impl.h"
#include <cuda_bf16.h>

namespace onnxruntime::contrib::cuda {
namespace {
using bf16 = __nv_bfloat16;
constexpr int kMaxRows = 16;
__device__ __forceinline__ float B2F(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float Silu(float v) { return v / (1.0f + expf(-v)); }

// Preserve the frozen fused path's FP32 ordering and BF16 convolution boundary.
template <int R_FIXED = 0>
__global__ void MambaConvKernel(const bf16* proj, int R_rt, int proj_stride, int di, int conv_dim,
                                const bf16* conv_state_in, const bf16* conv_w, const bf16* conv_b, bf16* conv_state_out,
                                bf16* xbc_act, const int32_t* sel, int64_t slots) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  const int R = R_FIXED > 0 ? R_FIXED : R_rt;
  if (*sel < 0 || static_cast<int64_t>(*sel) >= slots) {
    for (int r = 0; r < R; ++r) {
      xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(NAN);
      for (int k = 0; k < 3; ++k)
        conv_state_out[(static_cast<int64_t>(r) * conv_dim + c) * 3 + k] = __float2bfloat16(NAN);
    }
    return;
  }
  conv_state_in += static_cast<int64_t>(*sel) * conv_dim * 3;
  float inp[3 + (R_FIXED > 0 ? R_FIXED : kMaxRows)];
  inp[0] = B2F(conv_state_in[c * 3]);
  inp[1] = B2F(conv_state_in[c * 3 + 1]);
  inp[2] = B2F(conv_state_in[c * 3 + 2]);
  const bf16* weight = conv_w + static_cast<int64_t>(c) * 4;
  uint2 wv;
  if ((reinterpret_cast<uintptr_t>(weight) & (alignof(uint2) - 1)) == 0) {
    wv = __ldg(reinterpret_cast<const uint2*>(weight));
  } else {
    bf16* values = reinterpret_cast<bf16*>(&wv);
#pragma unroll
    for (int i = 0; i < 4; ++i) values[i] = weight[i];
  }
  const bf16* wp = reinterpret_cast<const bf16*>(&wv);
  const float w0 = B2F(wp[0]), w1 = B2F(wp[1]), w2 = B2F(wp[2]), w3 = B2F(wp[3]);
  const float bias = B2F(conv_b[c]);
  for (int r = 0; r < R; ++r) inp[3 + r] = B2F(proj[static_cast<int64_t>(r) * proj_stride + di + c]);
  for (int r = 0; r < R; ++r) {
    float acc = bias + w0 * inp[r] + w1 * inp[r + 1] + w2 * inp[r + 2] + w3 * inp[r + 3];
    xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(Silu(acc));
  }
  for (int r = 0; r < R; ++r) {
    bf16* out = conv_state_out + (static_cast<int64_t>(r) * conv_dim + c) * 3;
    out[0] = __float2bfloat16(inp[r + 1]);
    out[1] = __float2bfloat16(inp[r + 2]);
    out[2] = __float2bfloat16(inp[r + 3]);
  }
}

template <int QP, int R_FIXED = 0>
__global__ void MambaSsmKernel(const bf16* proj, int R_rt, int proj_stride, int di, int conv_dim, int heads,
                               int heads_per_group, const bf16* xbc_act, const float* ssm_in, const float* dt_bias,
                               const float* a_neg, const float* d_skip, float tmin, float* ssm_out, float* ypre,
                               const int32_t* sel, int64_t slots) {
  constexpr int DS = 128, HD = 64, NH = DS / QP;
  constexpr int MAX_R = R_FIXED > 0 ? R_FIXED : kMaxRows;
  const int h = blockIdx.x, g = h / heads_per_group;
  const int tid = threadIdx.x, p = tid % HD, half = tid / HD;
  const int R = R_FIXED > 0 ? R_FIXED : R_rt;
  // Uniform across the block and before either state read; safe during CUDA capture.
  if (*sel < 0 || static_cast<int64_t>(*sel) >= slots) {
    for (int r = 0; r < R; ++r) {
      for (int i = tid; i < DS * HD; i += blockDim.x)
        ssm_out[(static_cast<int64_t>(r) * heads + h) * DS * HD + i] = NAN;
      for (int i = tid; i < HD; i += blockDim.x) ypre[static_cast<int64_t>(r) * di + h * HD + i] = NAN;
    }
    return;
  }
  __shared__ float sB[MAX_R][DS], sC[MAX_R][DS], sPart[MAX_R][QP][HD], sDt[MAX_R];
  const int b_off = di + g * DS;
  const int c_off = di + (heads / heads_per_group) * DS + g * DS;
  for (int i = tid; i < R * DS; i += blockDim.x) {
    const int r = i / DS, n = i % DS;
    sB[r][n] = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + b_off + n]);
    sC[r][n] = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + c_off + n]);
  }
  if (tid < R) {
    const float raw = B2F(proj[static_cast<int64_t>(tid) * proj_stride + di + conv_dim + h]) + dt_bias[h];
    float dt = raw > 20.0f ? raw : log1pf(expf(raw));
    sDt[tid] = fmaxf(dt, tmin);
  }
  __syncthreads();
  float st[NH];
  const int64_t row_stride = static_cast<int64_t>(heads) * DS * HD;
  const float* sp = ssm_in + static_cast<int64_t>(*sel) * row_stride +
                    (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
  for (int n = 0; n < NH; ++n) st[n] = sp[static_cast<int64_t>(n) * HD];
  const float A = a_neg[h], D = d_skip[h];
  for (int r = 0; r < R; ++r) {
    const float dt = sDt[r];
    const float dA = expf(A * dt);
    const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + p]);
    const float dtx = dt * x;
    float acc = 0.f;
#pragma unroll
    for (int n = 0; n < NH; ++n) {
      const int nn = half * NH + n;
      st[n] = st[n] * dA + dtx * sB[r][nn];
      acc += sC[r][nn] * st[n];
    }
    sPart[r][half][p] = acc;
    float* out = ssm_out + r * row_stride + (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
    for (int n = 0; n < NH; ++n) out[static_cast<int64_t>(n) * HD] = st[n];
  }
  __syncthreads();
  for (int i = tid; i < R * HD; i += blockDim.x) {
    const int r = i / HD, pp = i % HD;
    const float x = B2F(xbc_act[static_cast<int64_t>(r) * conv_dim + h * HD + pp]);
    float sum = D * x;
#pragma unroll
    for (int q = 0; q < QP; ++q) sum += sPart[r][q][pp];
    ypre[static_cast<int64_t>(r) * di + h * HD + pp] = sum;
  }
}

__global__ void MambaNormKernel(const bf16* proj, int proj_stride, int di, int group_size, float eps,
                                const float* ypre, const float* norm_w, bf16* y) {
  const int r = blockIdx.y, g = blockIdx.x;
  const int64_t base = static_cast<int64_t>(r) * di + static_cast<int64_t>(g) * group_size;
  const bf16* gate = proj + static_cast<int64_t>(r) * proj_stride + static_cast<int64_t>(g) * group_size;
  __shared__ float red[32];
  if (group_size == 1024) {
    const int tid = threadIdx.x;
    float v[4], ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = tid + k * 256;
      const float gt = B2F(__ldg(gate + i));
      v[k] = __ldg(ypre + base + i) * (gt / (1.0f + expf(-gt)));
      ss += v[k] * v[k];
    }
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
    if ((tid & 31) == 0) red[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32) {
      float w = tid < 8 ? red[tid] : 0.f;
      for (int o = 16; o > 0; o >>= 1) w += __shfl_down_sync(0xffffffff, w, o);
      if (tid == 0) red[0] = w;
    }
    __syncthreads();
    const float rs = rsqrtf(red[0] / group_size + eps);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int i = tid + k * 256;
      y[base + i] = __float2bfloat16(v[k] * rs * __ldg(norm_w + g * 1024 + i));
    }
    return;
  }
  float ss = 0.f;
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float gt = B2F(gate[i]);
    const float v = ypre[base + i] * (gt / (1.0f + expf(-gt)));
    ss += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < 8 ? red[threadIdx.x] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float rs = rsqrtf(red[0] / group_size + eps);
  for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float gt = B2F(gate[i]);
    const float v = ypre[base + i] * (gt / (1.0f + expf(-gt)));
    y[base + i] = __float2bfloat16(v * rs * norm_w[g * group_size + i]);
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
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  // Zero is reserved for ineligible slots; NaNs follow every numeric score.
  if (magnitude > 0x7f800000u) return 1u;
  // Floating-point equality treats both signed zeros as the same score.
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
  // The decision is warp-uniform before either path's full-warp shuffles.
  if (__all_sync(0xffffffff, finite)) {
    RouterSelectTopk(values, sc, topk, scale, -INFINITY, indices, weights);
    return;
  }
  // Exceptional scores need distinct ranks for NaNs, -infinity and removed slots.
  uint32_t ranks[16];
#pragma unroll
  for (int i = 0; i < 16; ++i)
    ranks[i] = lane + 32 * i < E ? RouterScoreRank(values[i]) : 0u;
  RouterSelectTopk(ranks, sc, topk, scale, 0u, indices, weights);
}

__global__ void ReluSquaredKernel(const bf16* x, bf16* y, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = fmaxf(B2F(x[i]), 0.f);
    y[i] = __float2bfloat16(v * v);
  }
}
}  // namespace

cudaError_t LaunchMamba2VerifyNormalization(cudaStream_t stream, int rows, int width, int groups, int projection_stride,
                                            float epsilon, const void* projection, const float* readout,
                                            const float* norm_weight, void* output) {
  MambaNormKernel<<<dim3(groups, rows), 256, 0, stream>>>(static_cast<const bf16*>(projection), projection_stride,
                                                          width, width / groups, epsilon, readout, norm_weight,
                                                          static_cast<bf16*>(output));
  return cudaGetLastError();
}

cudaError_t LaunchMamba2Verify(cudaStream_t stream, int rows, int64_t slots, int heads, int groups, float epsilon,
                               float time_step_min, const void* projection, const void* conv_state,
                               const float* ssm_state, const void* conv_weight, const void* conv_bias,
                               const float* dt_bias, const float* a_negative, const float* d_skip,
                               const float* norm_weight, const int32_t* state_row, void* output, void* conv_snapshots,
                               float* ssm_snapshots, void* xbc, float* readout) {
  const int di = heads * 64, channels = di + 2 * groups * 128, stride = di + channels + heads;
  const auto* proj = static_cast<const bf16*>(projection);
  const auto* conv_in = static_cast<const bf16*>(conv_state);
  const auto* cw = static_cast<const bf16*>(conv_weight);
  const auto* cb = static_cast<const bf16*>(conv_bias);
  auto* conv_out = static_cast<bf16*>(conv_snapshots);
  auto* act = static_cast<bf16*>(xbc);
  if (rows == 8) {
    MambaConvKernel<8><<<(channels + 63) / 64, 64, 0, stream>>>(proj, rows, stride, di, channels, conv_in, cw, cb,
                                                                conv_out, act, state_row, slots);
    MambaSsmKernel<4, 8><<<heads, 256, 0, stream>>>(proj, rows, stride, di, channels, heads, heads / groups, act,
                                                    ssm_state, dt_bias, a_negative, d_skip, time_step_min,
                                                    ssm_snapshots, readout, state_row, slots);
  } else {
    MambaConvKernel<<<(channels + 127) / 128, 128, 0, stream>>>(proj, rows, stride, di, channels, conv_in, cw, cb,
                                                                conv_out, act, state_row, slots);
    if (rows > 1) {
      MambaSsmKernel<4><<<heads, 256, 0, stream>>>(proj, rows, stride, di, channels, heads, heads / groups, act,
                                                   ssm_state, dt_bias, a_negative, d_skip, time_step_min, ssm_snapshots,
                                                   readout, state_row, slots);
    } else {
      MambaSsmKernel<2><<<heads, 128, 0, stream>>>(proj, rows, stride, di, channels, heads, heads / groups, act,
                                                   ssm_state, dt_bias, a_negative, d_skip, time_step_min, ssm_snapshots,
                                                   readout, state_row, slots);
    }
  }
  return LaunchMamba2VerifyNormalization(stream, rows, di, groups, stride, epsilon, proj, readout, norm_weight, output);
}

cudaError_t LaunchBiasCorrectedMoERoute(cudaStream_t stream, int rows, int width, int experts, int topk, float scale,
                                        const void* input, const void* weight, const float* bias, int32_t* indices,
                                        float* weights, float* scores) {
  RouterLogitsKernel<<<dim3(rows, (experts + 7) / 8), 256, 0, stream>>>(
      static_cast<const bf16*>(input), static_cast<const bf16*>(weight), width, experts, scores);
  RouterTopkKernel<<<rows, 32, 0, stream>>>(scores, bias, experts, topk, scale, indices, weights);
  return cudaGetLastError();
}

cudaError_t LaunchReluSquared(cudaStream_t stream, const void* input, void* output, int64_t count) {
  if (count > 0) {
    ReluSquaredKernel<<<static_cast<unsigned int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const bf16*>(input), static_cast<bf16*>(output), count);
  }
  return cudaGetLastError();
}
}  // namespace onnxruntime::contrib::cuda
