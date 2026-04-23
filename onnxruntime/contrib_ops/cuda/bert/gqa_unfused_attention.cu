// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GQA-capable unfused CUDA attention kernel. See header for contract.

#include <math_constants.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "core/common/safeint.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/gqa_unfused_attention.h"

using onnxruntime::cuda::OrtToCudaType;

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr size_t kAlign = 256;

inline SafeInt<size_t> AlignTo(SafeInt<size_t> a, size_t b) { return ((a + (b - 1)) / b) * b; }

// Device helper: convert T to float. Specialised for __half and __nv_bfloat16
// to keep conversions consistent with the rest of the codebase.
template <typename T>
__device__ __forceinline__ float ToFloat(T v);
template <>
__device__ __forceinline__ float ToFloat<float>(float v) { return v; }
template <>
__device__ __forceinline__ float ToFloat<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float ToFloat<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

inline size_t QkElementCount(int batch_size, int num_heads, int q_seq, int total_kv) {
  return SafeInt<size_t>(batch_size) * num_heads * q_seq * total_kv;
}

// ---------------------------------------------------------------------------
// Softmax kernel: reads FP32 QK scores, writes T softmax output.
//
// Applies (in this order):
//   1. scale: x = scale * qk
//   2. softcap (if > 0):            x = softcap * tanh(x / softcap)
//   3. attn_bias (if provided):     x += bias
//   4. mask (causal + sliding window + per-batch seqlens_k)
//   5. stable softmax across [start, end) for each row
//
// Uses 3-pass strided reads to avoid shared memory size limits for large
// total_kv_length. Handles fully-masked rows by emitting zeros (no NaN).
// ---------------------------------------------------------------------------
template <typename T, int TPB>
__global__ void GqaUnfusedSoftmaxKernel(
    const int q_sequence_length,
    const int total_kv_length,
    const int num_heads,  // N_q
    const float* __restrict__ qk_in,
    const T* __restrict__ attn_bias,
    const bool has_bias,
    const bool bcast_bias_dim_0,
    const bool bcast_bias_dim_1,
    const int* __restrict__ seqlens_k,
    const bool is_causal,
    const int local_window_size,
    const float scale,
    const float softcap,
    T* __restrict__ softmax_out) {
  // Grid: (N_q * S_q, B, 1). Block: (TPB, 1, 1).
  const int q_in_head = blockIdx.x % q_sequence_length;
  const int head = blockIdx.x / q_sequence_length;  // 0..N_q-1
  const int batch = blockIdx.y;

  int kv_end = total_kv_length;
  if (seqlens_k != nullptr) {
    int v = seqlens_k[batch];
    if (v < kv_end) kv_end = v;
    if (v < 0) kv_end = 0;
  }
  // past (number of KV positions before the current query tokens) must be
  // per-batch when seqlens_k is provided, since different batches can have
  // different amounts of valid past context. Using the global total_kv_length
  // would over-estimate past for short batches and shift the sliding-window
  // start past kv_end, producing an all-masked (zero) row.
  const int past = kv_end - q_sequence_length;
  const int q_pos = past + q_in_head;

  int end = kv_end;
  if (is_causal) {
    const int c = q_pos + 1;
    if (c < end) end = c;
  }
  int start = 0;
  if (local_window_size >= 0) {
    const int s = q_pos - local_window_size;
    if (s > start) start = s;
  }
  if (end < 0) end = 0;
  if (start > end) start = end;

  // Row offsets
  const int64_t row_idx = (static_cast<int64_t>(batch) * gridDim.x) + blockIdx.x;
  const int64_t row_offset = row_idx * total_kv_length;

  int64_t bias_row_offset = 0;
  if (has_bias) {
    const int b_eff = bcast_bias_dim_0 ? 0 : batch;
    const int n_stride = bcast_bias_dim_1 ? 1 : num_heads;
    const int h_eff = bcast_bias_dim_1 ? 0 : head;
    bias_row_offset = ((static_cast<int64_t>(b_eff) * n_stride + h_eff) * q_sequence_length + q_in_head) *
                      static_cast<int64_t>(total_kv_length);
  }

  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  __shared__ float s_max;
  __shared__ float s_inv_sum;

  // Pass 1: compute max of masked values.
  float thread_max = -CUDART_INF_F;
  for (int i = threadIdx.x; i < total_kv_length; i += TPB) {
    if (i < start || i >= end) continue;
    float x = qk_in[row_offset + i] * scale;
    if (softcap > 0.f) {
      x = softcap * tanhf(x / softcap);
    }
    if (has_bias) {
      x += ToFloat(attn_bias[bias_row_offset + i]);
    }
    if (x > thread_max) thread_max = x;
  }
  float block_max = BlockReduce(tmp_storage).Reduce(thread_max, cub::Max());
  if (threadIdx.x == 0) s_max = block_max;
  __syncthreads();

  // If the row is fully masked, emit zeros (match existing mask-of-zeros behavior).
  if (s_max == -CUDART_INF_F) {
    for (int i = threadIdx.x; i < total_kv_length; i += TPB) {
      softmax_out[row_offset + i] = T(0.f);
    }
    return;
  }

  // Pass 2: compute sum of exp.
  float thread_sum = 0.f;
  for (int i = threadIdx.x; i < total_kv_length; i += TPB) {
    if (i < start || i >= end) continue;
    float x = qk_in[row_offset + i] * scale;
    if (softcap > 0.f) {
      x = softcap * tanhf(x / softcap);
    }
    if (has_bias) {
      x += ToFloat(attn_bias[bias_row_offset + i]);
    }
    thread_sum += expf(x - s_max);
  }
  float block_sum = BlockReduce(tmp_storage).Reduce(thread_sum, cub::Sum());
  if (threadIdx.x == 0) s_inv_sum = (block_sum > 0.f) ? (1.f / block_sum) : 0.f;
  __syncthreads();

  // Pass 3: write softmax output in type T.
  for (int i = threadIdx.x; i < total_kv_length; i += TPB) {
    float y = 0.f;
    if (i >= start && i < end) {
      float x = qk_in[row_offset + i] * scale;
      if (softcap > 0.f) {
        x = softcap * tanhf(x / softcap);
      }
      if (has_bias) {
        x += ToFloat(attn_bias[bias_row_offset + i]);
      }
      y = expf(x - s_max) * s_inv_sum;
    }
    softmax_out[row_offset + i] = T(y);
  }
}

template <typename T>
void LaunchGqaUnfusedSoftmax(
    cudaStream_t stream,
    const GqaUnfusedAttentionParams& params,
    const float* qk_in,
    const T* attn_bias,
    T* softmax_out) {
  const dim3 grid(params.num_heads * params.q_sequence_length, params.batch_size, 1);
  const bool has_bias = (attn_bias != nullptr);
  constexpr int TPB = 256;
  GqaUnfusedSoftmaxKernel<T, TPB><<<grid, TPB, 0, stream>>>(
      params.q_sequence_length,
      params.total_kv_length,
      params.num_heads,
      qk_in,
      attn_bias,
      has_bias,
      params.broadcast_attn_bias_dim_0,
      params.broadcast_attn_bias_dim_1,
      params.seqlens_k,
      params.is_causal,
      params.local_window_size,
      params.scale,
      params.softcap,
      softmax_out);
}

// ---------------------------------------------------------------------------
// QK GEMM: FP32 accumulate into FP32 scratch (fixes #28195 fp16 overflow).
//
// Reshape-Q trick for GQA: batch_count = B * N_kv, and within each batch the
// Q matrix is the concatenation of `group_size` Q heads. No K/V replication.
//
// Per-batch matrices (row-major view):
//   Q sub-matrix: [group_size * S_q, H]  (contiguous in memory: BNSH layout
//                                         with heads in the same KV group
//                                         being contiguous).
//   K sub-matrix: [S_kv,            H]
//   C sub-matrix: [group_size * S_q, S_kv]
//
// cuBLAS is column-major: the row-major (M, K) matrix is column-major (K, M).
// We issue: C = op_A(A) * op_B(B) where
//   A = K  (col-major (H, S_kv), op_A = T)   →  M_cublas = S_kv, K_cublas = H
//   B = Q  (col-major (H, group_size*S_q), op_B = N)
//                                             →  N_cublas = group_size*S_q
//   C (col-major (S_kv, group_size*S_q)) == row-major (group_size*S_q, S_kv).
// ---------------------------------------------------------------------------
template <typename T>
cudaDataType CudaTypeFor();
template <>
cudaDataType CudaTypeFor<__half>() { return CUDA_R_16F; }
template <>
cudaDataType CudaTypeFor<__nv_bfloat16>() { return CUDA_R_16BF; }
template <>
cudaDataType CudaTypeFor<float>() { return CUDA_R_32F; }

template <typename T>
common::Status LaunchQkGemmFp32(
    const cudaDeviceProp& /*device_prop*/,
    cublasHandle_t cublas,
    const GqaUnfusedAttentionParams& params,
    const T* query,
    const T* key,
    float* qk_out) {
  const int B = params.batch_size;
  const int N_kv = params.kv_num_heads;
  const int group = params.num_heads / params.kv_num_heads;
  const int S_q = params.q_sequence_length;
  const int S_kv = params.total_kv_length;
  const int H = params.head_size;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Strides between (b, n_kv) blocks:
  //   Q is BNSH with heads grouped: element (b, n_kv, g, s_q, h) at offset
  //     ((b * N_kv + n_kv) * group + g) * S_q * H + s_q * H + h
  //   so stride per (b, n_kv) = group * S_q * H.
  //   K is BNSH: per (b, n_kv) = max_kv_length * H.
  //   C (fp32 scratch) is packed [B, N_q, S_q, S_kv]; per (b, n_kv) block is
  //     group * S_q * S_kv.
  const int64_t stride_q = static_cast<int64_t>(group) * S_q * H;
  const int64_t stride_k = static_cast<int64_t>(params.max_kv_length) * H;
  const int64_t stride_c = static_cast<int64_t>(group) * S_q * S_kv;

  cudaDataType ab_type = CudaTypeFor<T>();
  // compute + scale type is FP32 → no fp16 overflow of raw QK even at
  // head_size=512, scale=1.0 (direct fix for issue #28195).
  cublasStatus_t status = cublasGemmStridedBatchedEx(
      cublas,
      CUBLAS_OP_T, CUBLAS_OP_N,
      /*m=*/S_kv, /*n=*/group * S_q, /*k=*/H,
      &alpha,
      /*A=*/key, ab_type, /*lda=*/H, stride_k,
      /*B=*/query, ab_type, /*ldb=*/H, stride_q,
      &beta,
      /*C=*/qk_out, CUDA_R_32F, /*ldc=*/S_kv, stride_c,
      /*batch_count=*/B * N_kv,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);

  if (status != CUBLAS_STATUS_SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GqaUnfusedAttention QK GEMM failed: ", status);
  }
  return common::Status::OK();
}

// ---------------------------------------------------------------------------
// Attn * V GEMM: C = P * V, where P is the softmax output in type T.
//
// Per-batch matrices (row-major view, batch_count = B * N_kv):
//   P sub-matrix: [group_size * S_q, S_kv]  (packed, leading dim = total_kv)
//   V sub-matrix: [S_kv,            H_v]
//   Y sub-matrix: [group_size * S_q, H_v]
//
// cuBLAS column-major: issue C = A * B with
//   A = V  (col-major (H_v, S_kv), op_A = N)       → M = H_v, K = S_kv
//   B = P  (col-major (S_kv, group*S_q), op_B = N) → N = group * S_q
// ---------------------------------------------------------------------------
template <typename T>
common::Status LaunchAttnVGemm(
    cublasHandle_t cublas,
    const GqaUnfusedAttentionParams& params,
    const T* softmax_out,
    const T* value,
    T* output) {
  const int B = params.batch_size;
  const int N_kv = params.kv_num_heads;
  const int group = params.num_heads / params.kv_num_heads;
  const int S_q = params.q_sequence_length;
  const int S_kv = params.total_kv_length;
  const int H_v = params.v_head_size;

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const int64_t stride_v = static_cast<int64_t>(params.max_kv_length) * H_v;
  const int64_t stride_p = static_cast<int64_t>(group) * S_q * S_kv;
  const int64_t stride_y = static_cast<int64_t>(group) * S_q * H_v;

  // Use cublasGemmStridedBatchedEx directly with FP32 alpha/beta + FP32 compute.
  // The helper has no __nv_bfloat16 overload and __half overload depends on
  // global HalfGemmOptions; going direct gives deterministic behavior.
  cudaDataType ab_type = CudaTypeFor<T>();
  cublasStatus_t status = cublasGemmStridedBatchedEx(
      cublas,
      CUBLAS_OP_N, CUBLAS_OP_N,
      /*m=*/H_v, /*n=*/group * S_q, /*k=*/S_kv,
      &alpha,
      /*A=*/value, ab_type, /*lda=*/H_v, stride_v,
      /*B=*/softmax_out, ab_type, /*ldb=*/S_kv, stride_p,
      &beta,
      /*C=*/output, ab_type, /*ldc=*/H_v, stride_y,
      /*batch_count=*/B * N_kv,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GqaUnfusedAttention AV GEMM failed: ", status);
  }
  return common::Status::OK();
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
size_t GetGqaUnfusedAttentionWorkspaceSize(int batch_size,
                                           int num_heads,
                                           int q_sequence_length,
                                           int total_kv_length) {
  const size_t elems = QkElementCount(batch_size, num_heads, q_sequence_length, total_kv_length);
  // FP32 QK scratch + T softmax scratch. We always allocate sizeof(float) per
  // element for the T scratch too (upper bound); caller can cast appropriately.
  const size_t qk_bytes = AlignTo(SafeInt<size_t>(elems) * sizeof(float), kAlign);
  const size_t softmax_bytes = AlignTo(SafeInt<size_t>(elems) * sizeof(float), kAlign);
  return SafeInt<size_t>(qk_bytes) + softmax_bytes;
}

template <typename T>
common::Status LaunchGqaUnfusedAttention(
    const cudaDeviceProp& device_prop,
    cublasHandle_t cublas,
    cudaStream_t stream,
    const GqaUnfusedAttentionParams& params,
    const T* query,
    const T* key,
    const T* value,
    const T* attn_bias,
    T* output,
    void* workspace) {
  ORT_RETURN_IF_NOT(params.batch_size > 0 && params.num_heads > 0 && params.kv_num_heads > 0 &&
                        params.head_size > 0 && params.v_head_size > 0 &&
                        params.q_sequence_length > 0 && params.total_kv_length > 0 &&
                        params.max_kv_length >= params.total_kv_length,
                    "GqaUnfusedAttention: invalid params.");
  ORT_RETURN_IF_NOT(params.num_heads % params.kv_num_heads == 0,
                    "GqaUnfusedAttention: num_heads (", params.num_heads,
                    ") must be a multiple of kv_num_heads (", params.kv_num_heads, ").");
  ORT_RETURN_IF(workspace == nullptr, "GqaUnfusedAttention: workspace is null.");

  const size_t elems = QkElementCount(params.batch_size, params.num_heads,
                                      params.q_sequence_length, params.total_kv_length);
  const size_t qk_bytes = AlignTo(SafeInt<size_t>(elems) * sizeof(float), kAlign);

  auto* qk_fp32 = reinterpret_cast<float*>(workspace);
  auto* softmax_T = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(workspace) + qk_bytes);

  ORT_RETURN_IF_ERROR((LaunchQkGemmFp32<T>(device_prop, cublas, params, query, key, qk_fp32)));

  LaunchGqaUnfusedSoftmax<T>(stream, params, qk_fp32, attn_bias, softmax_T);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  ORT_RETURN_IF_ERROR((LaunchAttnVGemm<T>(cublas, params, softmax_T, value, output)));

  return common::Status::OK();
}

// Explicit template instantiations.
template common::Status LaunchGqaUnfusedAttention<__half>(
    const cudaDeviceProp&, cublasHandle_t, cudaStream_t,
    const GqaUnfusedAttentionParams&, const __half*, const __half*, const __half*,
    const __half*, __half*, void*);
template common::Status LaunchGqaUnfusedAttention<__nv_bfloat16>(
    const cudaDeviceProp&, cublasHandle_t, cudaStream_t,
    const GqaUnfusedAttentionParams&, const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, void*);
template common::Status LaunchGqaUnfusedAttention<float>(
    const cudaDeviceProp&, cublasHandle_t, cudaStream_t,
    const GqaUnfusedAttentionParams&, const float*, const float*, const float*,
    const float*, float*, void*);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
