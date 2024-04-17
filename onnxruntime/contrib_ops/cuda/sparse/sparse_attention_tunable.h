// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <math.h>
#include "contrib_ops/cuda/sparse/sparse_attention_triton.cuh"
#include "core/providers/cuda/tunable/cuda_tuning_context.h"

using ::onnxruntime::cuda::tunable::CudaTuningContext;
using ::onnxruntime::cuda::tunable::OpParams;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
struct SparseAttentionParams {
  T* output;
  const T* q;
  const T* k;
  const T* v;

  int batch_size;
  int num_heads;
  int kv_num_heads;
  int head_size;

  int sequence_length;
  int past_sequence_length;
  int total_sequence_length;

  float softmax_scale;

  int kernel_block_size;

  // CSR format of block mask
  const int* layout_crow;
  const int* layout_col;
  int layout_crow_stride_h;
  int layout_col_stride_h;
  int num_layout;

  SparseAttentionParams(
      T* output,
      const T* q,
      const T* k,
      const T* v,
      int batch_size,
      int sequence_length,
      int num_heads,
      int kv_num_heads,
      int head_size,
      int total_sequence_length,
      float softmax_scale,
      int kernel_block_size,
      const int* layout_crow,
      const int* layout_col,
      int layout_crow_stride_h,
      int layout_col_stride_h,
      int num_layout) {
    this->output = output;
    this->q = q;
    this->k = k;
    this->v = v;
    this->batch_size = batch_size;
    this->sequence_length = sequence_length;
    this->num_heads = num_heads;
    this->kv_num_heads = kv_num_heads;
    this->head_size = head_size;
    this->past_sequence_length = total_sequence_length - sequence_length;
    this->total_sequence_length = total_sequence_length;
    this->softmax_scale = softmax_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(head_size)) : softmax_scale;
    this->kernel_block_size = kernel_block_size;
    this->layout_crow = layout_crow;
    this->layout_col = layout_col;
    this->layout_crow_stride_h = layout_crow_stride_h;
    this->layout_col_stride_h = layout_col_stride_h;
    this->num_layout = num_layout;
  }
};

template <typename T>
struct SparseAttentionTunableParams : OpParams, SparseAttentionParams<T> {
  SparseAttentionTunableParams(
      CudaTuningContext* tuning_ctx,
      onnxruntime::Stream* ort_stream,
      T* output,
      const T* q,
      const T* k,
      const T* v,
      int batch_size,
      int sequence_length,
      int num_heads,
      int kv_num_heads,
      int head_size,
      int total_sequence_length,
      float softmax_scale,
      int kernel_block_size,
      const int* layout_crow,
      const int* layout_col,
      int layout_crow_stride_h,
      int layout_col_stride_h,
      int num_layout)
      : OpParams(tuning_ctx, ort_stream),
        SparseAttentionParams<T>(
            output, q, k, v,
            batch_size, sequence_length, num_heads, kv_num_heads, head_size, total_sequence_length, softmax_scale,
            kernel_block_size,
            layout_crow, layout_col,
            layout_crow_stride_h,
            layout_col_stride_h,
            num_layout) {}

  std::string Signature() const override {
    return std::to_string(this->kernel_block_size) + "_" +
           std::to_string(this->batch_size) + "_" +
           std::to_string(this->sequence_length) + "_" +
           std::to_string(this->num_heads) + "_" +
           std::to_string(this->kv_num_heads) + "_" +
           std::to_string(this->head_size) + "_" +
           std::to_string(this->total_sequence_length);
  }
};

template <typename T>
class SparseAttentionTunableOp : public TunableOp<SparseAttentionTunableParams<T>> {
 public:
  SparseAttentionTunableOp() {
#ifdef USE_TRITON_KERNEL
    for (auto&& [_, op] : GetTritonBlockSparseAttentionTypeStringAndOps<T>()) {
      ORT_UNUSED_PARAMETER(_);
      this->RegisterOp(std::move(op));
    }
#endif
  }
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
