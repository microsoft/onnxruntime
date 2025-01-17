// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"

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

inline hipblasStatus_t _compat_hipblas_gemm_strided_batched_ex(hipblasHandle_t handle,
                                                               hipblasOperation_t transa,
                                                               hipblasOperation_t transb,
                                                               int m,
                                                               int n,
                                                               int k,
                                                               const void* alpha,
                                                               const void* A,
                                                               hipDataType a_type,
                                                               int lda,
                                                               hipblasStride stride_A,
                                                               const void* b,
                                                               hipDataType b_type,
                                                               int ldb,
                                                               hipblasStride stride_b,
                                                               const void* beta,
                                                               void* c,
                                                               hipDataType c_type,
                                                               int ldc,
                                                               hipblasStride stride_c,
                                                               int batch_count,
                                                               hipblasComputeType_t compute_type,
                                                               hipblasGemmAlgo_t algo) {
  return hipblasGemmStridedBatchedEx(handle,
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
                                     batch_count,  // batch count
                                     compute_type,
                                     algo);
}

// Compatible for CublasMathModeSetter
class CompatHipblasMathModeSetter {
 public:
  CompatHipblasMathModeSetter(const hipDeviceProp_t&,
                              hipblasHandle_t,
                              int) {
  }
};

enum AttentionMode {
  // Q,K,V,PastK,PastV,PresentK,PresentV
  QFMT_KFMT_VFMT_NONE_NONE_NONE_NONE,
  QFMT_KFMT_VFMT_NONE_NONE_2BNTH_NONE,
  QFMT_KFMT_VFMT_NONE_NONE_2BNMH_NONE,
  QFMT_KFMT_VFMT_2BNPH_NONE_2BNTH_NONE,
  QFMT_KFMT_VFMT_2BNMH_NONE_2BNMH_NONE,
  BSNH_BLNH_BLNH_NONE_NONE_NONE_NONE,
  BSNH_BNLH_BNLH_NONE_NONE_NONE_NONE,
  BSNH_BLNH_BLNH_NONE_NONE_BNTH_BNTH,
  BSNH_BNLH_BNLH_NONE_NONE_BNTH_BNTH,
  BSNH_BLNH_BLNH_NONE_NONE_BNMH_BNMH,
  BSNH_BNLH_BNLH_NONE_NONE_BNMH_BNMH,
  BSNH_BLNH_BLNH_BNPH_BNPH_BNTH_BNTH,
  BSNH_BNLH_BNLH_BNPH_BNPH_BNTH_BNTH,
  BSNH_BLNH_BLNH_BNMH_BNMH_BNMH_BNMH,
  BSNH_BNLH_BNLH_BNMH_BNMH_BNMH_BNMH,
  BLN3H_NONE_NONE_NONE_NONE_NONE_NONE,
  BSNH_BLN2H_NONE_NONE_NONE_NONE_NONE,
};

struct RocmAttentionParameters : AttentionParameters {
  AttentionMode mode;
};

Status ClassifyAttentionMode(AttentionType type,
                             RocmAttentionParameters* attn,
                             const std::vector<const Tensor*>& qkv,
                             const std::vector<const Tensor*>& past,
                             const std::vector<Tensor*>& present);

template <typename T>
Status LaunchStridedCopy(
    hipStream_t stream,
    const T* in, int4 in_shape, longlong4 in_strides, const int* in_seqlens_offset,  // coord (b,n,s,h)
    T* out, longlong4 out_strides, const int* out_seqlens_offset,                    // coord (b,n,s,h)
    int max_threads_per_block);

template <typename T>
Status LaunchStridedCopy(hipStream_t stream,
                         const T* in, int4 in_shape, longlong4 in_strides,  // coord (b,n,s,h)
                         T* out, longlong4 out_strides,                     // coord (b,n,s,h)
                         int max_threads_per_block);
}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
