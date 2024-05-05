// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "contrib_ops/cuda/bert/transformer_cuda_common.h"

using onnxruntime::cuda::tunable::CudaTuningContext;

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct BlockLayout {
  const int32_t* mask;  // shape (num_layout, num_rows, num_cols), where num_rows = num_cols = max_seq_len / block_size.
  int num_layout;
  int block_size;  // kernel block size, which is <= sparse_block_size

  const int32_t* csr_col_indices;
  const int32_t* csr_row_indices;
  int num_rows;
  int num_cols;
};

template <typename T>
struct SparseAttentionData {
  // Input Tensors
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;

  const int32_t* block_mask = nullptr;
  const int32_t* seqlens_k_total = nullptr;

  // Temporary buffers
  T* transposed_q_buffer = nullptr;
  T* rotary_buffer = nullptr;
  T* unpacked_qkv_buffer = nullptr;

  // This is sparse layout used in kernel.
  BlockLayout kernel_layout;

  // Output Tensors
  T* output = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;

  // Data for sparse attention v2 kernel.
  bool use_v2_kernel = false;
  int* q_batch_starts = nullptr;  // shape (batch_size)
  int* q_batch_ends = nullptr;    // shape (batch_size)
  int* k_batch_starts = nullptr;  // shape (batch_size)
  int* k_batch_ends = nullptr;    // shape (batch_size)
  int* q_batch_ids = nullptr;     // shape (G)
  int* q_start_sids = nullptr;    // shape (G)
  int active_q_blocks = 0;        // G: number of blocks in q that are not masked out

  // Data for mask to CSR conversion
  IAllocatorUniquePtr<int> csr_col_indices_buffer;
  IAllocatorUniquePtr<int> csr_row_indices_buffer;

  // Data for dense flash attention
  IAllocatorUniquePtr<char> softmax_lse_accum;
  IAllocatorUniquePtr<char> out_accum;
  IAllocatorUniquePtr<char> softmax_lse;
  int num_splits = 0;

  // Data for sparse attention v2 kernel.
  IAllocatorUniquePtr<int32_t> pinned_buffer;                   // buffer to copy total_k_seq_len from GPU to CPU.
  AutoDestoryCudaEvent is_copy_done;                            // CUDA event to syncronize the copy of total_k_seq_len.
  IAllocatorUniquePtr<int32_t> v2_kernel_inputs_pinned_buffer;  // v2 kernel inputs in CPU (will be copied to GPU)
  IAllocatorUniquePtr<int32_t> v2_kernel_buffer;                // v2 kernel inputs in GPU.
};

template <typename T>
Status QkvToContext_Sparse(
    const cudaDeviceProp& device_prop,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<T>& data);

template <typename T>
Status QkvToContext_Dense(
    const cudaDeviceProp& device_prop,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<T>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
