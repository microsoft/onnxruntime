// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/tunable/cuda_tunable.h"

using onnxruntime::cuda::tunable::CudaTuningContext;

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct BlockLayout {
  int num_layout;
  int block_size;              // kernel block size, which is <= sparse_block_size
  const int* csr_row_indices;  // shape [num_layout, stride_row_indices]
  const int* csr_col_indices;  // shape [num_layout, stride_col_indices]
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
};

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    Stream* ort_stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<T>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
