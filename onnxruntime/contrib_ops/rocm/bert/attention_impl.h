// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {
size_t GetAttentionScratchSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int sequence_length,
    int all_sequence_length);

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int head_size,
    int sequence_length,
    int past_sequence_length);

Status LaunchAttentionKernel(
    const hipDeviceProp_t& prop,               // Device Properties
    hipStream_t stream,                        // Hip stream
    rocblas_handle& rocblas,                   // Rocblas handle
    const size_t element_size,                 // Element size of input tensor
    int batch_size,                            // Batch size (B)
    int sequence_length,                       // Sequence length (S)
    int num_heads,                             // Number of attention heads (N)
    int head_size,                             // Hidden layer size per head (H)
    int past_sequence_length,                  // Sequence length in past state
    bool is_unidirectional,                    // Whether there is unidirectional mask.
    const void* input,                         // Input tensor
    const int* mask_index,                     // Attention mask raw data or index. NULL means no mask.
    gsl::span<const int64_t> mask_index_dims,  // Mask index shape
    const void* past,                          // Past state input
    const void* extra_add_qk,                  // Additional Add
    void* workspace,                           // Temporary buffer
    void* output,                              // Output tensor
    void* present                              // Present state output
);

Status LaunchDecoderAttentionKernel(
    const hipDeviceProp_t& prop,      // Device Properties
    hipStream_t stream,               // Hip stream
    rocblas_handle& rocblas,          // Rocblas handle
    const size_t element_size,        // Element size of input tensor
    const int batch_size,             // Batch size (B)
    const int sequence_length,        // Sequence length (S)
    const int kv_sequence_length,     // Key/Value/Cache sequence length
    const int num_heads,              // Number of attention heads (N)
    const int head_size,              // Hidden layer size per head (H)
    const bool static_kv,             // Whether cross attention or not
    const bool use_past,              // Whether use cache or not
    const bool has_layer_state,       // Whether output cache or not
    const bool has_key_padding_mask,  // Whether use key_padding_mask or not
    const void* gemm_query_buffer,    // Query buffer
    const void* gemm_kv_buffer,       // Key and value buffer
    const bool* key_padding_mask,     // Key padding mask
    const void* key_cache,            // Input key cache
    const void* value_cache,          // Input value cache
    void* qkv_buffer,                 // Temporary buffer
    void* workspace_buffer,           // Temporary buffer
    void* output,                     // Output tensor
    void* new_key_cache,              // New_key_cache tensor
    void* new_value_cache             // New_value_cache tensor
);

Status LaunchTransCtx(hipStream_t stream,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const float* input, float* output);

Status LaunchTransCtx(hipStream_t stream,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const half* input, half* output);

Status LaunchTransQkv(hipStream_t stream, const int matrix_num,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const float* input, float* output,
                      int total_matrix_count = -1);

Status LaunchTransQkv(hipStream_t stream, const int matrix_num,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const half* input, half* output,
                      int total_matrix_count = -1);

Status LaunchConcatTensorToTensor(hipStream_t stream,
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

Status LaunchConcatTensorToTensor(hipStream_t stream,
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

Status LaunchConcatPastToPresent(hipStream_t stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int head_size,
                                 const int num_heads,
                                 const int max_threads_per_block,
                                 const float* past,
                                 const float* k_v,
                                 float* present);

Status LaunchConcatPastToPresent(hipStream_t stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int head_size,
                                 const int num_heads,
                                 const int max_threads_per_block,
                                 const half* past,
                                 const half* k_v,
                                 half* present);

inline rocblas_status _compat_rocblas_gemm_strided_batched_ex(rocblas_handle handle,
                                                              rocblas_operation transa,
                                                              rocblas_operation transb,
                                                              int m,
                                                              int n,
                                                              int k,
                                                              const void* alpha,
                                                              const void* A,
                                                              rocblas_datatype a_type,
                                                              rocblas_int lda,
                                                              rocblas_stride stride_A,
                                                              const void* b,
                                                              rocblas_datatype b_type,
                                                              rocblas_int ldb,
                                                              rocblas_stride stride_b,
                                                              const void* beta,
                                                              void* c,
                                                              rocblas_datatype c_type,
                                                              rocblas_int ldc,
                                                              rocblas_stride stride_c,
                                                              rocblas_int batch_count,
                                                              rocblas_datatype compute_type,
                                                              rocblas_gemm_algo algo) {
  return rocblas_gemm_strided_batched_ex(handle,
                                         transa,
                                         transb,
                                         m,            // m
                                         n,            // n
                                         k,            // k
                                         alpha,        // alpha
                                         A,            // A
                                         a_type,       // A type
                                         lda,          // lda
                                         stride_A,     // strideA
                                         b,            // B
                                         b_type,       // B type
                                         ldb,          // ldb
                                         stride_b,     // strideB
                                         beta,         // beta
                                         c,            // C
                                         c_type,       // C type
                                         ldc,          // ldc
                                         stride_c,     // strideC
                                         c,            // D = C
                                         c_type,       // D type = C type
                                         ldc,          // ldd = ldc
                                         stride_c,     // strideD = strideC
                                         batch_count,  // batch count
                                         compute_type,
                                         algo,
                                         0, 0);
}

// Compatible for CublasMathModeSetter
class CompatRocblasMathModeSetter {
 public:
  CompatRocblasMathModeSetter(const hipDeviceProp_t&,
                              rocblas_handle,
                              int) {
  }
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
