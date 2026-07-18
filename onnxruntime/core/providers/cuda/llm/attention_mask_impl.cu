// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/llm/attention_mask_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void ConvertBoolMaskToAttentionBiasKernel(
    const bool* __restrict__ attn_mask,
    T* __restrict__ attention_bias,
    const int64_t num_elements,
    const float mask_filter_value) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < num_elements;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    attention_bias[idx] = attn_mask[idx] ? T(0.0f) : T(mask_filter_value);
  }
}

template <typename T>
Status LaunchConvertBoolMaskToAttentionBias(
    const bool* attn_mask_bool,
    T* attention_bias,
    int64_t num_elements,
    float mask_filter_value,
    cudaStream_t stream,
    int max_threads_per_block) {
  if (num_elements == 0) {
    return Status::OK();
  }

  int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), num_elements));
  int64_t blocks = (num_elements + threads - 1) / threads;
  // Cap grid size to avoid exceeding CUDA gridDim.x limit (2^31 - 1).
  // The grid-stride loop in the kernel handles the overflow.
  constexpr int64_t kMaxGridDimX = 65535;
  unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

  ConvertBoolMaskToAttentionBiasKernel<T><<<grid_size, threads, 0, stream>>>(
      attn_mask_bool, attention_bias, num_elements, mask_filter_value);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConvertBoolMaskToAttentionBias<float>(
    const bool*, float*, int64_t, float, cudaStream_t, int);
template Status LaunchConvertBoolMaskToAttentionBias<__half>(
    const bool*, __half*, int64_t, float, cudaStream_t, int);
template Status LaunchConvertBoolMaskToAttentionBias<__nv_bfloat16>(
    const bool*, __nv_bfloat16*, int64_t, float, cudaStream_t, int);

// Convert nonpad_kv_seqlen (int64) to seqlens_k (int32) as actual token count.
// Flash attention's mha_fwd_kvcache expects seqlens_k_ = number of valid tokens.
__global__ void ConvertNonpadKvSeqlenToFlashSeqlensKKernel(
    const int64_t* __restrict__ nonpad_kv_seqlen,
    int* __restrict__ seqlens_k,
    const int batch_size,
    const int total_sequence_length) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < batch_size) {
    int64_t val = nonpad_kv_seqlen[idx];
    CUDA_KERNEL_ASSERT(val >= 0);
    CUDA_KERNEL_ASSERT(val <= static_cast<int64_t>(total_sequence_length));
    val = max(static_cast<int64_t>(0), min(val, static_cast<int64_t>(total_sequence_length)));
    seqlens_k[idx] = static_cast<int>(val);  // count, not index
  }
}

Status LaunchConvertNonpadKvSeqlenToFlashSeqlensK(
    const int64_t* nonpad_kv_seqlen,
    int* seqlens_k,
    int batch_size,
    int total_sequence_length,
    cudaStream_t stream,
    int max_threads_per_block) {
  if (batch_size == 0) {
    return Status::OK();
  }

  int threads = std::min(batch_size, max_threads_per_block);
  int blocks = (batch_size + threads - 1) / threads;

  ConvertNonpadKvSeqlenToFlashSeqlensKKernel<<<blocks, threads, 0, stream>>>(
      nonpad_kv_seqlen, seqlens_k, batch_size, total_sequence_length);

  return CUDA_CALL(cudaGetLastError());
}

// Zero output elements for batches where seqlens_k == 0 (fully masked).
// CUTLASS MEA epilogue computes 1/s_prime where s_prime=0 → NaN for fully-masked
// batches. The unfused path produces uniform softmax weights (finite mask_filter_value,
// not -inf) so output is valid but non-zero; we still zero for Flash parity.
// Flash handles this natively with an early-exit for empty sequences.
template <typename T>
__global__ void ZeroOutputForFullyMaskedBatchesKernel(
    T* __restrict__ output,
    const int* __restrict__ seqlens_k,
    const int batch_size,
    const int64_t elements_per_batch) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(batch_size) * elements_per_batch;
  for (; idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int b = static_cast<int>(idx / elements_per_batch);
    if (seqlens_k[b] == 0) {
      output[idx] = T(0.0f);
    }
  }
}

template <typename T>
Status LaunchZeroOutputForFullyMaskedBatches(
    T* output,
    const int* seqlens_k,
    int batch_size,
    int64_t elements_per_batch,
    cudaStream_t stream,
    int max_threads_per_block) {
  int64_t total = static_cast<int64_t>(batch_size) * elements_per_batch;
  if (total == 0) {
    return Status::OK();
  }

  int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), total));
  int64_t blocks = (total + threads - 1) / threads;
  constexpr int64_t kMaxGridDimX = 65535;
  unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

  ZeroOutputForFullyMaskedBatchesKernel<T><<<grid_size, threads, 0, stream>>>(
      output, seqlens_k, batch_size, elements_per_batch);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchZeroOutputForFullyMaskedBatches<float>(
    float*, const int*, int, int64_t, cudaStream_t, int);
template Status LaunchZeroOutputForFullyMaskedBatches<__half>(
    __half*, const int*, int, int64_t, cudaStream_t, int);
template Status LaunchZeroOutputForFullyMaskedBatches<__nv_bfloat16>(
    __nv_bfloat16*, const int*, int, int64_t, cudaStream_t, int);

// Zero output rows fully masked by the intersection of the causal frontier and an
// explicit additive attention bias (composed is_causal + attn_mask), per onnx#8068
// "fully-masked-row -> 0" (Bug-2). The MEA/CUTLASS path uses a finite mask sentinel,
// so a row with no allowed key softmaxes to mean-of-V; this overwrites it with zeros.
// One (batch, head, query) row per loop iteration via a grid-stride loop, so every row
// is covered even when batch_size * num_heads * q_sequence_length exceeds
// gridDim.x * blockDim.x (the grid is capped at kMaxGridDimX). Output is BSNH.
template <typename T>
__global__ void ZeroFullyMaskedRowsKernel(
    T* __restrict__ output,
    const T* __restrict__ attn_bias,
    const int* __restrict__ seqlens_k,
    const int batch_size,
    const int num_heads,
    const int q_sequence_length,
    const int total_sequence_length,
    const int v_head_size,
    const bool is_causal,
    const bool causal_from_top_left,
    const int bias_dim0,  // attn_bias extent along batch: broadcast ? 1 : batch_size
    const int bias_dim1,  // attn_bias extent along head:  broadcast ? 1 : num_heads
    const float masked_bias_value) {
  const int64_t total_rows =
      static_cast<int64_t>(batch_size) * num_heads * q_sequence_length;
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < total_rows;
       row += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // Decode flattened index into (batch b, head n, query i) for BNS ordering.
    const int i = static_cast<int>(row % q_sequence_length);
    const int n = static_cast<int>((row / q_sequence_length) % num_heads);
    const int b = static_cast<int>(row / (static_cast<int64_t>(q_sequence_length) * num_heads));

    const int num_keys = (seqlens_k != nullptr) ? seqlens_k[b] : total_sequence_length;

    // Causally-allowed keys are j in [0, key_upper). Bottom-right anchors the diagonal at
    // offset = num_keys - q_sequence_length (== nonpad_kv_seqlen[b] - q_seq); top-left uses 0.
    int key_upper;
    if (is_causal) {
      const int offset = causal_from_top_left ? 0 : (num_keys - q_sequence_length);
      const int causal_last = i + offset;  // inclusive last causally-allowed key index
      key_upper = min(num_keys, causal_last + 1);
    } else {
      key_upper = num_keys;
    }

    bool any_allowed = false;
    if (key_upper > 0) {
      if (attn_bias == nullptr) {
        any_allowed = true;  // no explicit mask -> at least one causally-allowed key
      } else {
        const int b_idx = (bias_dim0 == 1) ? 0 : b;
        const int n_idx = (bias_dim1 == 1) ? 0 : n;
        const int64_t base =
            ((static_cast<int64_t>(b_idx) * bias_dim1 + n_idx) * q_sequence_length + i) *
            total_sequence_length;
        for (int j = 0; j < key_upper; ++j) {
          if (static_cast<float>(attn_bias[base + j]) > masked_bias_value) {
            any_allowed = true;
            break;
          }
        }
      }
    }

    if (!any_allowed) {
      // Output is BSNH: zero the [b, i, n, :] row (select, not multiply).
      const int64_t out_base =
          ((static_cast<int64_t>(b) * q_sequence_length + i) * num_heads + n) * v_head_size;
      for (int d = 0; d < v_head_size; ++d) {
        output[out_base + d] = T(0.0f);
      }
    }
  }
}

template <typename T>
Status LaunchZeroFullyMaskedRows(
    T* output,
    const T* attn_bias,
    const int* seqlens_k,
    int batch_size,
    int num_heads,
    int q_sequence_length,
    int total_sequence_length,
    int v_head_size,
    bool is_causal,
    bool causal_from_top_left,
    bool broadcast_bias_dim_0,
    bool broadcast_bias_dim_1,
    float masked_bias_value,
    cudaStream_t stream,
    int max_threads_per_block) {
  const int64_t total_rows =
      static_cast<int64_t>(batch_size) * num_heads * q_sequence_length;
  if (total_rows == 0) {
    return Status::OK();
  }

  const int bias_dim0 = broadcast_bias_dim_0 ? 1 : batch_size;
  const int bias_dim1 = broadcast_bias_dim_1 ? 1 : num_heads;

  int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), total_rows));
  int64_t blocks = (total_rows + threads - 1) / threads;
  // Cap the grid at kMaxGridDimX; the kernel grid-strides over total_rows, so any rows
  // beyond grid_size * threads are still covered (matches ZeroOutputForFullyMaskedBatches).
  constexpr int64_t kMaxGridDimX = 65535;
  unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

  ZeroFullyMaskedRowsKernel<T><<<grid_size, threads, 0, stream>>>(
      output, attn_bias, seqlens_k, batch_size, num_heads, q_sequence_length,
      total_sequence_length, v_head_size, is_causal, causal_from_top_left,
      bias_dim0, bias_dim1, masked_bias_value);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchZeroFullyMaskedRows<float>(
    float*, const float*, const int*, int, int, int, int, int, bool, bool, bool, bool, float,
    cudaStream_t, int);
template Status LaunchZeroFullyMaskedRows<__half>(
    __half*, const __half*, const int*, int, int, int, int, int, bool, bool, bool, bool, float,
    cudaStream_t, int);
template Status LaunchZeroFullyMaskedRows<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const int*, int, int, int, int, int, bool, bool, bool, bool,
    float, cudaStream_t, int);

// Simple kernel to fill an int32 buffer with a constant value on device.
// Used for CUDA-graph-capturable seqlens_k initialization (no host memory).
__global__ void FillInt32Kernel(int* __restrict__ output, const int value, const int count) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < count) {
    output[idx] = value;
  }
}

Status LaunchFillInt32(int* output, int value, int count, cudaStream_t stream, int max_threads_per_block) {
  if (count == 0) {
    return Status::OK();
  }

  int threads = std::min(count, max_threads_per_block);
  int blocks = (count + threads - 1) / threads;

  FillInt32Kernel<<<blocks, threads, 0, stream>>>(output, value, count);

  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace onnxruntime
