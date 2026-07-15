// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

// Macro to help compute index of flatten 4D matrix, note that dim1 is not used so it is excluded.
#define INDEX_4D(dim2, dim3, dim4, i, j, k, l) (int64_t(i) * (dim2) * (dim3) * (dim4) + int64_t(j) * (dim3) * (dim4) + int64_t(k) * (dim4) + int64_t(l))

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Matrix index constants
enum class QKV : int {
  Q = 0,
  K = 1,
  V = 2,
  COUNT = 3
};

// KV Cache Layout Documentation:
// BSNH format: [batch_size, sequence_length, num_heads, head_size]
//   - Preferred for most operations due to better memory coalescing for typical access patterns
//   - Adjacent threads in a warp (h dimension) access contiguous memory
//   - Used when is_bsnh=true
//
// BNSH format: [batch_size, num_heads, sequence_length, head_size]
//   - Used when sequence dimension needs to be contiguous
//   - May suffer from worse coalescing if head_size is small
//   - Used when is_bsnh=false (or explicit bnsh flags)

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const float* tensor_in,
                                  const float* tensor_add,
                                  float* tensor_out);

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const half* tensor_in,
                                  const half* tensor_add,
                                  half* tensor_out);

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const BFloat16* tensor_in,
                                  const BFloat16* tensor_add,
                                  BFloat16* tensor_out);

template <typename T>
Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                           const int max_sequence_length,
                                           const int past_sequence_length,
                                           const int sequence_length,
                                           const int batch_size,
                                           const int head_size,
                                           const int num_heads,
                                           const int max_threads_per_block,
                                           const T* biases,
                                           const T* qkv_buffer,
                                           T* present);

// Fused KV Append for Separate Buffer Mode: Appends New K & V to Past in one kernel
// Uses blockIdx.z to distinguish between K and V
template <typename T>
Status LaunchConcatNewToPastKV(const int batch_size,
                               const int kv_num_heads,
                               const int head_size,
                               const int kv_sequence_length,
                               const int past_sequence_length,
                               const int present_sequence_length,
                               const bool is_bsnh,
                               const int* past_seq_lens,
                               const int* total_seq_lens,
                               const T* past_key,
                               const T* past_value,
                               const T* new_key,
                               const T* new_value,
                               T* present_key,
                               T* present_value,
                               cudaStream_t stream,
                               const int max_threads_per_block,
                               const bool past_only,
                               const T* cos_cache = nullptr,
                               const T* sin_cache = nullptr,
                               const int rotary_dim = 0,
                               const int64_t* position_ids = nullptr,
                               const bool interleaved = false);

template <typename T>
Status LaunchConcatKVInPlace(int batch_size,
                             int kv_num_heads,
                             int head_size,
                             int max_sequence_length,  // max sequence length of present_key or present_value.
                             const int* past_seq_lens,
                             const int* total_seq_lens,
                             int new_seq_len,
                             const T* new_key,
                             const T* new_value,
                             T* present_key,
                             T* present_value,
                             bool is_past_kv_bnsh_format,
                             bool is_new_kv_bnsh_format,
                             cudaStream_t stream,
                             const int max_threads_per_block);

// Truly fused K+V In-Place Append with RoPE
// Single kernel that appends K (with RoPE rotation) and V (without rotation) to KV cache.
// This eliminates a separate kernel launch for V, saving kernel overhead.
template <typename T>
Status LaunchConcatKVInPlaceFused(int batch_size,
                                  int kv_num_heads,
                                  int head_size,
                                  int max_sequence_length,
                                  const int* past_seq_lens,
                                  const int* total_seq_lens,
                                  int new_seq_len,
                                  const T* new_key,
                                  const T* new_value,
                                  T* present_key,
                                  T* present_value,
                                  bool is_past_kv_bnsh_format,
                                  bool is_new_kv_bnsh_format,
                                  cudaStream_t stream,
                                  const int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
