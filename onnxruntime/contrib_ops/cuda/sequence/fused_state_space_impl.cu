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

// FP32 convolution accumulation with one BF16 rounding after SiLU.
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

__global__ void MambaConvPrefillKernel(
    const bf16* __restrict__ proj, int R, int proj_stride, int di, int conv_dim,
    const bf16* __restrict__ conv_state_in, const bf16* __restrict__ conv_w,
    const bf16* __restrict__ conv_b, bf16* __restrict__ xbc_act,
    const int32_t* __restrict__ sel, int64_t slots) {
  constexpr int kConvTileR = 32;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int r_start = blockIdx.x * kConvTileR;
  if (c >= conv_dim || r_start >= R) return;
  const int r_end = (R - r_start) < kConvTileR ? R : (r_start + kConvTileR);
  const int32_t s = *sel;
  if (s < 0 || static_cast<int64_t>(s) >= slots) {
    for (int r = r_start; r < r_end; ++r) {
      xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(NAN);
    }
    return;
  }
  const bf16* cs = conv_state_in + (static_cast<int64_t>(s) * conv_dim + c) * 3;
  const auto load_tap = [&](int idx) -> float {
    if (idx < 0) return B2F(__ldg(cs + (3 + idx)));
    return B2F(__ldg(proj + static_cast<int64_t>(idx) * proj_stride + di + c));
  };
  float p0 = load_tap(r_start - 3);
  float p1 = load_tap(r_start - 2);
  float p2 = load_tap(r_start - 1);
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
  const float bias = B2F(__ldg(conv_b + c));
  for (int r = r_start; r < r_end; ++r) {
    const float p3 = B2F(__ldg(proj + static_cast<int64_t>(r) * proj_stride + di + c));
    const float acc = bias + w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3;
    xbc_act[static_cast<int64_t>(r) * conv_dim + c] = __float2bfloat16(Silu(acc));
    p0 = p1;
    p1 = p2;
    p2 = p3;
  }
}

// conv_state_in may alias conv_state_out (MayInplace), so neither state pointer is __restrict__.
__global__ void MambaConvWriteSnapshotsKernel(
    const bf16* __restrict__ proj, int R, int proj_stride, int di, int conv_dim,
    const bf16* conv_state_in, bf16* conv_state_out,
    const int32_t* __restrict__ sel, int64_t slots) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  const int snap_count = static_cast<int64_t>(R) < slots ? R : static_cast<int>(slots);
  const int first_snap_r = R - snap_count;
  const int32_t s = *sel;
  if (s < 0 || static_cast<int64_t>(s) >= slots) {
    for (int sr = 0; sr < snap_count; ++sr) {
      for (int k = 0; k < 3; ++k) {
        conv_state_out[(static_cast<int64_t>(sr) * conv_dim + c) * 3 + k] = __float2bfloat16(NAN);
      }
    }
    return;
  }
  const bf16* cs = conv_state_in + (static_cast<int64_t>(s) * conv_dim + c) * 3;
  const bf16 cs0 = cs[0], cs1 = cs[1], cs2 = cs[2];
  const auto load_hist = [&](int idx) -> bf16 {
    if (idx == -3) return cs0;
    if (idx == -2) return cs1;
    if (idx == -1) return cs2;
    return __ldg(proj + static_cast<int64_t>(idx) * proj_stride + di + c);
  };
  for (int sr = 0; sr < snap_count; ++sr) {
    const int r = first_snap_r + sr;
    bf16* out = conv_state_out + (static_cast<int64_t>(sr) * conv_dim + c) * 3;
    out[0] = load_hist(r - 2);
    out[1] = load_hist(r - 1);
    out[2] = load_hist(r);
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
    sDt[tid] = dt < tmin ? tmin : dt;
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

// ssm_in may alias ssm_out (MayInplace), so neither state pointer is __restrict__.
template <int QP, int TILE_R = 16>
__global__ __launch_bounds__(QP * 64) void MambaSsmPrefillKernel(
    const bf16* __restrict__ proj, int R, int proj_stride, int di, int conv_dim, int heads,
    int heads_per_group, const bf16* __restrict__ xbc_act, const float* ssm_in,
    const float* __restrict__ dt_bias, const float* __restrict__ a_neg, const float* __restrict__ d_skip,
    float tmin, float* ssm_out, float* __restrict__ ypre,
    const int32_t* __restrict__ sel, int64_t slots) {
  constexpr int DS = 128, HD = 64, NH = DS / QP, kThreads = QP * HD;
  const int h = blockIdx.x, g = h / heads_per_group;
  const int tid = threadIdx.x, p = tid % HD, half = tid / HD;
  const int snap_count = static_cast<int64_t>(R) < slots ? R : static_cast<int>(slots);
  const int first_snap_r = R - snap_count;
  const int32_t s = *sel;
  if (s < 0 || static_cast<int64_t>(s) >= slots) {
    for (int sr = 0; sr < snap_count; ++sr) {
      for (int i = tid; i < DS * HD; i += kThreads) {
        ssm_out[(static_cast<int64_t>(sr) * heads + h) * DS * HD + i] = NAN;
      }
    }
    for (int r = 0; r < R; ++r) {
      for (int i = tid; i < HD; i += kThreads) {
        ypre[static_cast<int64_t>(r) * di + h * HD + i] = NAN;
      }
    }
    return;
  }
  __shared__ float sB[TILE_R][DS];
  __shared__ float sC[TILE_R][DS];
  __shared__ float sX[TILE_R][HD];
  __shared__ float sPart[TILE_R][QP][HD];
  __shared__ float sDt[TILE_R];
  __shared__ float sdA[TILE_R];

  float st[NH];
  const int64_t row_stride = static_cast<int64_t>(heads) * DS * HD;
  const float* sp = ssm_in + static_cast<int64_t>(s) * row_stride +
                    (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
  for (int n = 0; n < NH; ++n) st[n] = sp[static_cast<int64_t>(n) * HD];

  const float A = a_neg[h], D = d_skip[h], dt_b = dt_bias[h];
  const int b_off = di + g * DS;
  const int c_off = di + (heads / heads_per_group) * DS + g * DS;

  for (int r0 = 0; r0 < R; r0 += TILE_R) {
    const int tile_rows = (R - r0) < TILE_R ? (R - r0) : TILE_R;
    for (int i = tid; i < tile_rows * DS; i += kThreads) {
      const int tr = i / DS, n = i % DS;
      const int64_t base = static_cast<int64_t>(r0 + tr) * conv_dim;
      sB[tr][n] = B2F(__ldg(xbc_act + base + b_off + n));
      sC[tr][n] = B2F(__ldg(xbc_act + base + c_off + n));
    }
    for (int i = tid; i < tile_rows * HD; i += kThreads) {
      const int tr = i / HD, pp = i % HD;
      sX[tr][pp] = B2F(__ldg(xbc_act + static_cast<int64_t>(r0 + tr) * conv_dim + h * HD + pp));
    }
    if (tid < tile_rows) {
      const float raw = B2F(__ldg(proj + static_cast<int64_t>(r0 + tid) * proj_stride + di + conv_dim + h)) + dt_b;
      float dt = raw > 20.0f ? raw : log1pf(expf(raw));
      dt = dt < tmin ? tmin : dt;
      sDt[tid] = dt;
      sdA[tid] = expf(A * dt);
    }
    __syncthreads();

    for (int tr = 0; tr < tile_rows; ++tr) {
      const float dt = sDt[tr];
      const float dA = sdA[tr];
      const float x = sX[tr][p];
      const float dtx = dt * x;
      float acc = 0.f;
#pragma unroll
      for (int n = 0; n < NH; ++n) {
        const int nn = half * NH + n;
        st[n] = st[n] * dA + dtx * sB[tr][nn];
        acc += sC[tr][nn] * st[n];
      }
      sPart[tr][half][p] = acc;
      const int r = r0 + tr;
      if (r >= first_snap_r) {
        float* out = ssm_out + static_cast<int64_t>(r - first_snap_r) * row_stride +
                     (static_cast<int64_t>(h) * DS + half * NH) * HD + p;
#pragma unroll
        for (int n = 0; n < NH; ++n) out[static_cast<int64_t>(n) * HD] = st[n];
      }
    }
    __syncthreads();

    for (int i = tid; i < tile_rows * HD; i += kThreads) {
      const int tr = i / HD, pp = i % HD;
      float sum = D * sX[tr][pp];
#pragma unroll
      for (int q = 0; q < QP; ++q) sum += sPart[tr][q][pp];
      ypre[static_cast<int64_t>(r0 + tr) * di + h * HD + pp] = sum;
    }
    if (r0 + TILE_R < R) {
      __syncthreads();
    }
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

}  // namespace

cudaError_t LaunchMamba2VerifyNormalization(cudaStream_t stream, int rows, int width, int groups, int projection_stride,
                                            float epsilon, const void* projection, const float* readout,
                                            const float* norm_weight, void* output) {
  // Groups stay on grid-x (up to 2^31-1); rows are sliced to respect the 65535 grid-y limit.
  constexpr int64_t kMaxGridRows = 65535;
  const auto* proj = static_cast<const bf16*>(projection);
  auto* y = static_cast<bf16*>(output);
  for (int64_t r0 = 0; r0 < rows; r0 += kMaxGridRows) {
    const int64_t remaining = rows - r0;
    const unsigned int chunk = static_cast<unsigned int>(remaining < kMaxGridRows ? remaining : kMaxGridRows);
    MambaNormKernel<<<dim3(groups, chunk), 256, 0, stream>>>(proj + r0 * projection_stride, projection_stride, width,
                                                             width / groups, epsilon, readout + r0 * width,
                                                             norm_weight, y + r0 * width);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
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
  if (rows == 8 && slots >= 8) {
    MambaConvKernel<8><<<(channels + 63) / 64, 64, 0, stream>>>(proj, rows, stride, di, channels, conv_in, cw, cb,
                                                                conv_out, act, state_row, slots);
    MambaSsmKernel<4, 8><<<heads, 256, 0, stream>>>(proj, rows, stride, di, channels, heads, heads / groups, act,
                                                    ssm_state, dt_bias, a_negative, d_skip, time_step_min,
                                                    ssm_snapshots, readout, state_row, slots);
  } else if (rows <= kMaxRows && static_cast<int64_t>(rows) <= slots) {
    MambaConvKernel<<<(channels + 127) / 128, 128, 0, stream>>>(proj, rows, stride, di, channels, conv_in, cw, cb,
                                                                conv_out, act, state_row, slots);
    // Single-token replay of a multi-row snapshot cache must retain its readout reduction order.
    if (rows > 1 || slots > 1) {
      MambaSsmKernel<4><<<heads, 256, 0, stream>>>(proj, rows, stride, di, channels, heads, heads / groups, act,
                                                   ssm_state, dt_bias, a_negative, d_skip, time_step_min, ssm_snapshots,
                                                   readout, state_row, slots);
    } else {
      MambaSsmKernel<2><<<heads, 128, 0, stream>>>(proj, rows, stride, di, channels, heads, heads / groups, act,
                                                   ssm_state, dt_bias, a_negative, d_skip, time_step_min, ssm_snapshots,
                                                   readout, state_row, slots);
    }
  } else {
    const dim3 conv_grid(static_cast<unsigned int>((rows + 31) / 32),
                         static_cast<unsigned int>((channels + 127) / 128));
    MambaConvPrefillKernel<<<conv_grid, 128, 0, stream>>>(
        proj, rows, stride, di, channels, conv_in, cw, cb, act, state_row, slots);
    MambaConvWriteSnapshotsKernel<<<(channels + 127) / 128, 128, 0, stream>>>(
        proj, rows, stride, di, channels, conv_in, conv_out, state_row, slots);
    MambaSsmPrefillKernel<4, 16><<<heads, 256, 0, stream>>>(
        proj, rows, stride, di, channels, heads, heads / groups, act, ssm_state, dt_bias, a_negative, d_skip,
        time_step_min, ssm_snapshots, readout, state_row, slots);
  }
  return LaunchMamba2VerifyNormalization(stream, rows, di, groups, stride, epsilon, proj, readout, norm_weight, output);
}

}  // namespace onnxruntime::contrib::cuda
