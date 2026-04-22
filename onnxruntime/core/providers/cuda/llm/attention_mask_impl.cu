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

// CUDA kernel to convert nonpad_kv_seqlen to an additive attention bias.
// Generates (batch_size, q_seq_len, total_seq_len) output where:
//   position t < nonpad_kv_seqlen[b] → 0.0 (attend)
//   position t >= nonpad_kv_seqlen[b] → mask_filter_value (mask out)
template <typename T>
__global__ void ConvertNonpadKvSeqlenToAttentionBiasKernel(
    const int64_t* __restrict__ nonpad_kv_seqlen,
    T* __restrict__ attention_bias,
    const int batch_size,
    const int q_seq_len,
    const int total_seq_len,
    const float mask_filter_value) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(batch_size) * q_seq_len * total_seq_len;
  for (; idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int b = static_cast<int>(idx / (static_cast<int64_t>(q_seq_len) * total_seq_len));
    int t = static_cast<int>(idx % total_seq_len);
    int64_t valid_len = nonpad_kv_seqlen[b];
    CUDA_KERNEL_ASSERT(valid_len >= 0 && valid_len <= static_cast<int64_t>(total_seq_len));
    valid_len = max(static_cast<int64_t>(0), min(valid_len, static_cast<int64_t>(total_seq_len)));
    attention_bias[idx] = (t < static_cast<int>(valid_len)) ? T(0.0f) : T(mask_filter_value);
  }
}

template <typename T>
Status LaunchConvertNonpadKvSeqlenToAttentionBias(
    const int64_t* nonpad_kv_seqlen,
    T* attention_bias,
    int batch_size,
    int q_seq_len,
    int total_seq_len,
    float mask_filter_value,
    cudaStream_t stream,
    int max_threads_per_block) {
  int64_t total = static_cast<int64_t>(batch_size) * q_seq_len * total_seq_len;
  if (total == 0) {
    return Status::OK();
  }

  int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), total));
  int64_t blocks = (total + threads - 1) / threads;
  constexpr int64_t kMaxGridDimX = 65535;
  unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

  ConvertNonpadKvSeqlenToAttentionBiasKernel<T><<<grid_size, threads, 0, stream>>>(
      nonpad_kv_seqlen, attention_bias, batch_size, q_seq_len, total_seq_len, mask_filter_value);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConvertNonpadKvSeqlenToAttentionBias<float>(
    const int64_t*, float*, int, int, int, float, cudaStream_t, int);
template Status LaunchConvertNonpadKvSeqlenToAttentionBias<__half>(
    const int64_t*, __half*, int, int, int, float, cudaStream_t, int);
template Status LaunchConvertNonpadKvSeqlenToAttentionBias<__nv_bfloat16>(
    const int64_t*, __nv_bfloat16*, int, int, int, float, cudaStream_t, int);

// Add an addend bias into an existing bias buffer using cyclic broadcasting.
// Used to compose nonpad_kv_seqlen bias [B, q, t] with an attn_mask bias that
// is smaller or equal (e.g. 2D [q, t] cyclic-broadcasts over batch dimension).
template <typename T>
__global__ void AddBiasInPlaceKernel(
    T* __restrict__ bias,
    const T* __restrict__ addend,
    int64_t total_elements,
    int64_t addend_elements) {
  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_elements;
       idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    float sum = static_cast<float>(bias[idx]) + static_cast<float>(addend[idx % addend_elements]);
    bias[idx] = T(sum);
  }
}

template <typename T>
Status LaunchAddBiasInPlace(
    T* bias,
    const T* addend,
    int64_t total_elements,
    int64_t addend_elements,
    cudaStream_t stream,
    int max_threads_per_block) {
  if (total_elements == 0 || addend_elements == 0) {
    return Status::OK();
  }

  int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), total_elements));
  int64_t blocks = (total_elements + threads - 1) / threads;
  constexpr int64_t kMaxGridDimX = 65535;
  unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

  AddBiasInPlaceKernel<T><<<grid_size, threads, 0, stream>>>(
      bias, addend, total_elements, addend_elements);

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchAddBiasInPlace<float>(float*, const float*, int64_t, int64_t, cudaStream_t, int);
template Status LaunchAddBiasInPlace<__half>(__half*, const __half*, int64_t, int64_t, cudaStream_t, int);
template Status LaunchAddBiasInPlace<__nv_bfloat16>(__nv_bfloat16*, const __nv_bfloat16*, int64_t, int64_t, cudaStream_t, int);

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
