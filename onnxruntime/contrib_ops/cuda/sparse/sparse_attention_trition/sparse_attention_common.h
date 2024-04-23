// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include "core/providers/cuda/cuda_common.h"
// #include "core/framework/stream_handles.h"
// #include "core/providers/cuda/shared_inc/cuda_call.h"

#define CU_CHECK(expr) ORT_RETURN_IF_ERROR(CU_CALL(expr))

namespace onnxruntime {
namespace contrib {
namespace cuda {

struct SparseAttentionParams {
  onnxruntime::Stream* ort_stream;
  void* output;
  const void* q;
  const void* k;
  const void* v;

  int batch_size;
  int num_heads;
  int kv_num_heads;
  int head_size;

  int sequence_length;
  int past_sequence_length;
  int total_sequence_length;
  int max_sequence_length;

  float softmax_scale;

  int kernel_block_size;

  // CSR format of block mask
  const int* layout_crow;
  const int* layout_col;
  int layout_crow_stride_h;
  int layout_col_stride_h;
  int num_layout;

  // strides
  int stride_qb;
  int stride_qh;
  int stride_qm;
  int stride_kb;
  int stride_kh;
  int stride_kn;
  int stride_vb;
  int stride_vh;
  int stride_vn;
  int stride_ob;
  int stride_oh;
  int stride_om;

  SparseAttentionParams(
      onnxruntime::Stream* ort_stream,
      void* output,
      const void* q,
      const void* k,
      const void* v,
      int batch_size,
      int sequence_length,
      int num_heads,
      int kv_num_heads,
      int head_size,
      int total_sequence_length,
      int max_sequence_length,
      float softmax_scale,
      int kernel_block_size,
      const int* layout_crow,
      const int* layout_col,
      int layout_crow_stride_h,
      int layout_col_stride_h,
      int num_layout) {
    this->ort_stream = ort_stream;
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
    this->max_sequence_length = max_sequence_length;
    this->softmax_scale = softmax_scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(head_size)) : softmax_scale;
    this->kernel_block_size = kernel_block_size;
    this->layout_crow = layout_crow;
    this->layout_col = layout_col;
    this->layout_crow_stride_h = layout_crow_stride_h;
    this->layout_col_stride_h = layout_col_stride_h;
    this->num_layout = num_layout;

    // When kv buffer has max sequence length, stride should match max sequence length.
    int kv_buffer_sequence_length = max_sequence_length;

    this->stride_qb = this->num_heads * this->sequence_length * this->head_size;
    this->stride_qh = this->sequence_length * this->head_size;
    this->stride_qm = this->head_size;
    this->stride_kb = this->kv_num_heads * kv_buffer_sequence_length * this->head_size;
    this->stride_kh = kv_buffer_sequence_length * this->head_size;
    this->stride_kn = this->head_size;
    this->stride_vb = this->kv_num_heads * kv_buffer_sequence_length * this->head_size;
    this->stride_vh = kv_buffer_sequence_length * this->head_size;
    this->stride_vn = this->head_size;
    this->stride_ob = this->num_heads * this->sequence_length * this->head_size;
    this->stride_oh = this->sequence_length * this->head_size;
    this->stride_om = this->head_size;
  }

  Status LaunchKernel(CUfunction f, int block_m, int threads_per_block, unsigned int sharedMemBytes) {
    ORT_ENFORCE(f != nullptr, "Kernel shall be loaded before calling LaunchKernel.");

    if (!Valididate()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SparseAttentionParams is not valid.");
    }

    void* args[26] = {
        &output, &q, &k, &v,
        &layout_crow, &layout_col, &layout_crow_stride_h, &layout_col_stride_h, &num_layout, &softmax_scale,
        &stride_qb, &stride_qh, &stride_qm, &stride_kb, &stride_kh, &stride_kn,
        &stride_vb, &stride_vh, &stride_vn, &stride_ob, &stride_oh, &stride_om,
        &num_heads, &kv_num_heads, &total_sequence_length, &past_sequence_length};

    unsigned int gridDimX = (sequence_length + block_m - 1) / block_m;
    unsigned int gridDimY = batch_size * num_heads;
    constexpr unsigned int gridDimZ = 1;

    return CU_CALL(cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, threads_per_block, 1, 1, sharedMemBytes,
                                  static_cast<CUstream>(this->ort_stream->GetHandle()),
                                  args, NULL));
  }

  bool Valididate() {
    return (reinterpret_cast<size_t>(output) % 16 == 0 &&
            reinterpret_cast<size_t>(q) % 16 == 0 &&
            reinterpret_cast<size_t>(k) % 16 == 0 &&
            reinterpret_cast<size_t>(v) % 16 == 0 &&
            reinterpret_cast<size_t>(layout_crow) % 16 == 0 &&
            reinterpret_cast<size_t>(layout_col) % 16 == 0 &&
            this->head_size % 16 == 0);
  }
};

inline void SetKernelSharedMemory(CUfunction func) {
  int device = 0;
  CUDA_CALL_THROW(cudaGetDevice(&device));

  int shared_optin = 0;
  CU_CALL_THROW(cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
  if (shared_optin > 49152) {
    CU_CALL_THROW(cuFuncSetCacheConfig(func, CU_FUNC_CACHE_PREFER_SHARED));
    CU_CALL_THROW(cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
