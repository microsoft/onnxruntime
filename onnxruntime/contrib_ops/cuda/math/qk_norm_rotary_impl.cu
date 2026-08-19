// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/qk_norm_rotary_impl.h"

#include "contrib_ops/cuda/math/quant_sim_common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;

// Block-wide sum of `v`, left in s_red[0]. Every thread must call.
__device__ __forceinline__ float BlockSum(float v, float* s_red, int tid) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
  if ((tid & 31) == 0) s_red[tid >> 5] = v;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < kWarps; ++w) total += s_red[w];
    s_red[0] = total;
  }
  __syncthreads();
  return s_red[0];
}

// Rotate the trailing `rd` channels of a row held in shared memory. The rotary tables are
// stored pre-interleaved, so `x @ R` is the signed swap of each adjacent pair.
__device__ __forceinline__ void RopeLast(float* s_row, float* s_rope, int nope_dim, int rd,
                                         const float* cos_row, const float* sin_row, int tid) {
  for (int t = tid; t < rd; t += kThreads) s_rope[t] = s_row[nope_dim + t];
  __syncthreads();
  for (int t = tid; t < rd; t += kThreads) {
    const float rot = (t & 1) ? s_rope[t - 1] : -s_rope[t + 1];
    s_row[nope_dim + t] = s_rope[t] * cos_row[t] + rot * sin_row[t];
  }
  __syncthreads();
}

// Q: a weightless RMS norm followed by the partial rotation, one block per (token, head).
//
// The graph rounds the reciprocal to the activation dtype and then multiplies in that dtype,
// so both roundings are reproduced here rather than folded into one fp32 multiply.
template <typename CudaT>
__global__ void QueryNormRotaryKernel(const QKNormRotaryEmbeddingParams p,
                                      const CudaT* __restrict__ query,
                                      const float* __restrict__ cos_table,
                                      const float* __restrict__ sin_table,
                                      CudaT* __restrict__ out) {
  extern __shared__ float smem[];
  float* s_row = smem;
  float* s_red = s_row + p.head_dim;
  float* s_rope = s_red + kWarps;

  const int tid = threadIdx.x;
  const int d = p.head_dim;
  const int row = blockIdx.x;  // (b * seq_len + s) * num_heads + h
  const int token = row / p.num_heads;
  const CudaT* src = query + static_cast<int64_t>(row) * d;
  CudaT* dst = out + static_cast<int64_t>(row) * d;

  float ss = 0.0f;
  for (int c = tid; c < d; c += kThreads) {
    const float v = QuantSimConv<CudaT>::ToFloat(src[c]);
    s_row[c] = v;
    ss += v * v;
  }
  const float total = BlockSum(ss, s_red, tid);
  const float rs = QuantSimConv<CudaT>::Round(1.0f / sqrtf(total / d + p.epsilon));
  for (int c = tid; c < d; c += kThreads) s_row[c] = QuantSimConv<CudaT>::Round(s_row[c] * rs);
  __syncthreads();

  const int rd = p.rope_head_dim;
  if (rd > 0) {
    RopeLast(s_row, s_rope, p.nope_dim, rd, cos_table + static_cast<int64_t>(token) * rd,
             sin_table + static_cast<int64_t>(token) * rd, tid);
  }
  for (int c = tid; c < d; c += kThreads) dst[c] = QuantSimConv<CudaT>::FromFloat(s_row[c]);
}

// KV: a weighted RMS norm, the partial rotation, and the simulated FP8 round trip on the
// non-rotary half -- one block per token, since the latent KV row is shared by every head.
template <typename CudaT>
__global__ void KvNormRotaryKernel(const QKNormRotaryEmbeddingParams p,
                                   const CudaT* __restrict__ kv,
                                   const float* __restrict__ norm_weight,
                                   const float* __restrict__ cos_table,
                                   const float* __restrict__ sin_table,
                                   CudaT* __restrict__ out) {
  extern __shared__ float smem[];
  float* s_row = smem;
  float* s_red = s_row + p.head_dim;
  float* s_rope = s_red + kWarps;
  float* s_scale = s_rope + p.rope_head_dim;

  const int tid = threadIdx.x;
  const int d = p.head_dim;
  const int token = blockIdx.x;
  const CudaT* src = kv + static_cast<int64_t>(token) * d;
  CudaT* dst = out + static_cast<int64_t>(token) * d;

  float ss = 0.0f;
  for (int c = tid; c < d; c += kThreads) {
    const float v = QuantSimConv<CudaT>::ToFloat(src[c]);
    s_row[c] = v;
    ss += v * v;
  }
  const float total = BlockSum(ss, s_red, tid);
  const float rs = 1.0f / sqrtf(total / d + p.epsilon);
  // The norm itself stays in fp32 (its ONNX node emits FLOAT); the cast that follows it does not.
  for (int c = tid; c < d; c += kThreads) {
    s_row[c] = QuantSimConv<CudaT>::Round(s_row[c] * rs * norm_weight[c]);
  }
  __syncthreads();

  const int rd = p.rope_head_dim;
  if (rd > 0) {
    RopeLast(s_row, s_rope, p.nope_dim, rd, cos_table + static_cast<int64_t>(token) * rd,
             sin_table + static_cast<int64_t>(token) * rd, tid);
  }

  // Only the non-rotary half is quantised; the rotary half reaches the store still in fp32 and
  // is rounded there, which is where the graph's own cast back to the activation dtype lands.
  if (p.simulate_fp8) {
    const int blocks = p.nope_dim / 64;
    for (int i = tid; i < blocks; i += kThreads) {
      float amax = 0.0f;
      for (int j = 0; j < 64; ++j) amax = fmaxf(amax, fabsf(s_row[i * 64 + j]));
      s_scale[i] = QuantSimBlockScale(amax, kQuantSimFp8Max, 1e-30f);
    }
    __syncthreads();
    for (int c = tid; c < p.nope_dim; c += kThreads) {
      const float scale = s_scale[c >> 6];
      const float q = fminf(fmaxf(s_row[c] / scale, -kQuantSimFp8Max), kQuantSimFp8Max);
      s_row[c] = QuantSimConv<CudaT>::Round(QuantSimRoundE4M3(q) * scale);
    }
    __syncthreads();
  }

  for (int c = tid; c < d; c += kThreads) dst[c] = QuantSimConv<CudaT>::FromFloat(s_row[c]);
}

}  // namespace

int QueryNormRotarySharedFloats(const QKNormRotaryEmbeddingParams& p) {
  return p.head_dim + kWarps + p.rope_head_dim;
}

int KvNormRotarySharedFloats(const QKNormRotaryEmbeddingParams& p) {
  return p.head_dim + kWarps + p.rope_head_dim + (p.simulate_fp8 ? p.nope_dim / 64 : 0);
}

template <typename T>
Status LaunchQKNormRotaryEmbedding(cudaStream_t stream, const QKNormRotaryEmbeddingParams& p,
                                   const T* query, const T* kv, const float* kv_norm_weight,
                                   const float* cos_table, const float* sin_table,
                                   T* query_out, T* kv_out) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const int tokens = p.batch * p.seq_len;
  if (tokens == 0) return Status::OK();

  QueryNormRotaryKernel<CudaT>
      <<<tokens * p.num_heads, kThreads, QueryNormRotarySharedFloats(p) * sizeof(float),
         stream>>>(p, reinterpret_cast<const CudaT*>(query), cos_table, sin_table,
                   reinterpret_cast<CudaT*>(query_out));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  KvNormRotaryKernel<CudaT>
      <<<tokens, kThreads, KvNormRotarySharedFloats(p) * sizeof(float), stream>>>(
          p, reinterpret_cast<const CudaT*>(kv), kv_norm_weight, cos_table, sin_table,
          reinterpret_cast<CudaT*>(kv_out));
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

#define INSTANTIATE(T)                                                                             \
  template Status LaunchQKNormRotaryEmbedding<T>(cudaStream_t, const QKNormRotaryEmbeddingParams&, \
                                                 const T*, const T*, const float*, const float*,   \
                                                 const float*, T*, T*)

INSTANTIATE(float);
INSTANTIATE(MLFloat16);
INSTANTIATE(BFloat16);

#undef INSTANTIATE

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
