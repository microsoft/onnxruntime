// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cudaDriverWrapper.h"

#define CU_CHECK(expr, driver) cuErrCheck(expr, *driver)

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace sparse_attention_v1 {

struct SparseAttentionParams {
  onnxruntime::Stream* ort_stream;
  int sm;  // compute capability like 80 for A100

  void* output;
  const void* q;
  const void* k;
  const void* v;

  bool is_q_bnsh;

  int batch_size;
  int num_heads;
  int kv_num_heads;
  int head_size;

  int sequence_length;
  int past_sequence_length;
  int total_sequence_length;
  int max_cache_sequence_length;

  float scale;

  int kernel_block_size;

  // CSR format of block mask
  const int* layout_csr_row_indices;
  const int* layout_csr_col_indices;
  int layout_row_stride_h;
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
      int sm,
      void* output,
      const void* q,
      const void* k,
      const void* v,
      bool is_q_bnsh,
      int batch_size,
      int sequence_length,
      int num_heads,
      int kv_num_heads,
      int head_size,
      int total_sequence_length,
      int max_cache_sequence_length,
      float scale,
      int kernel_block_size,
      const int* layout_csr_row_indices,
      const int* layout_csr_col_indices,
      int layout_row_stride_h,
      int layout_col_stride_h,
      int num_layout) {
    this->ort_stream = ort_stream;
    this->sm = sm;
    this->output = output;
    this->q = q;
    this->k = k;
    this->v = v;
    this->is_q_bnsh = is_q_bnsh;
    this->batch_size = batch_size;
    this->sequence_length = sequence_length;
    this->num_heads = num_heads;
    this->kv_num_heads = kv_num_heads;
    this->head_size = head_size;
    this->past_sequence_length = total_sequence_length - sequence_length;
    this->total_sequence_length = total_sequence_length;
    this->max_cache_sequence_length = max_cache_sequence_length;
    this->scale = scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(head_size)) : scale;
    this->kernel_block_size = kernel_block_size;
    this->layout_csr_row_indices = layout_csr_row_indices;
    this->layout_csr_col_indices = layout_csr_col_indices;
    this->layout_row_stride_h = layout_row_stride_h;
    this->layout_col_stride_h = layout_col_stride_h;
    this->num_layout = num_layout;

    this->stride_qb = this->num_heads * this->sequence_length * this->head_size;
    this->stride_qh = (is_q_bnsh ? this->sequence_length : this->num_heads) * this->head_size;
    this->stride_qm = this->head_size;

    // When kv buffer has max sequence length, stride should match max sequence length.
    // KV cache is in BNSH format
    this->stride_kb = this->kv_num_heads * max_cache_sequence_length * this->head_size;
    this->stride_kh = max_cache_sequence_length * this->head_size;
    this->stride_kn = this->head_size;
    this->stride_vb = this->kv_num_heads * max_cache_sequence_length * this->head_size;
    this->stride_vh = max_cache_sequence_length * this->head_size;
    this->stride_vn = this->head_size;

    // Output is BSNH format
    this->stride_ob = this->sequence_length * this->num_heads * this->head_size;
    this->stride_oh = this->head_size;
    this->stride_om = this->num_heads * this->head_size;
  }

  Status LaunchKernel(CUfunction f, int block_m, int threads_per_block, unsigned int sharedMemBytes) {
    ORT_ENFORCE(f != nullptr, "Kernel shall be loaded before calling LaunchKernel.");

    if (!Valididate()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SparseAttentionParams is not valid.");
    }

    void* args[26] = {
        &output, &q, &k, &v,
        &layout_csr_row_indices, &layout_csr_col_indices, &layout_row_stride_h, &layout_col_stride_h, &num_layout, &scale,
        &stride_qb, &stride_qh, &stride_qm, &stride_kb, &stride_kh, &stride_kn,
        &stride_vb, &stride_vh, &stride_vn, &stride_ob, &stride_oh, &stride_om,
        &num_heads, &kv_num_heads, &total_sequence_length, &past_sequence_length};

    unsigned int gridDimX = (sequence_length + block_m - 1) / block_m;
    unsigned int gridDimY = batch_size * num_heads;
    constexpr unsigned int gridDimZ = 1;

#if DUMP_TENSOR_LEVEL > 0
    DUMP_TENSOR_INIT();
    DUMP_TENSOR("q", reinterpret_cast<const half*>(q), batch_size, num_heads, sequence_length, head_size);
    DUMP_TENSOR("k", reinterpret_cast<const half*>(k), batch_size, kv_num_heads, max_cache_sequence_length, head_size);
    DUMP_TENSOR("v", reinterpret_cast<const half*>(v), batch_size, kv_num_heads, max_cache_sequence_length, head_size);
    DUMP_TENSOR("csr_col_indices",
                layout_csr_col_indices,
                num_layout,
                layout_col_stride_h);

    DUMP_TENSOR("csr_row_indices",
                layout_csr_row_indices,
                num_layout,
                layout_row_stride_h);
    printf(
        "layout_row_stride_h=%d, layout_col_stride_h=%d, num_layout=%d, scale=%f,\n"
        "stride_qb=%d, stride_qh=%d, stride_qm=%d, stride_kb=%d, stride_kh=%d, stride_kn=%d,\n"
        "stride_vb=%d, stride_vh=%d, stride_vn=%d, stride_ob=%d, stride_oh=%d, stride_om=%d,\n"
        "num_heads=%d, kv_num_heads=%d, total_sequence_length=%d, past_sequence_length=%d\n"
        "output=%p, q=%p, k=%p, v=%p, layout_csr_row_indices=%p, layout_csr_col_indices=%p\n",
        layout_row_stride_h, layout_col_stride_h, num_layout, scale,
        stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn,
        stride_vb, stride_vh, stride_vn, stride_ob, stride_oh, stride_om,
        num_heads, kv_num_heads, total_sequence_length, past_sequence_length,
        output, q, k, v, layout_csr_row_indices, layout_csr_col_indices);

    printf("block_m=%d gridDimX=%d gridDimY=%d threads_per_block=%d sharedMemBytes=%d\n",
           block_m, gridDimX, gridDimY, threads_per_block, sharedMemBytes);
#endif

    const CUDADriverWrapper* driver = CUDADriverWrapper::GetInstance();
    CU_CHECK(driver->cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, threads_per_block, 1, 1, sharedMemBytes,
                                    static_cast<CUstream>(this->ort_stream->GetHandle()),
                                    args, NULL),
             driver);
    return Status::OK();
  }

  bool Valididate() {
    return (reinterpret_cast<size_t>(output) % 16 == 0 &&
            reinterpret_cast<size_t>(q) % 16 == 0 &&
            reinterpret_cast<size_t>(k) % 16 == 0 &&
            reinterpret_cast<size_t>(v) % 16 == 0 &&
            reinterpret_cast<size_t>(layout_csr_col_indices) % 16 == 0 &&
            reinterpret_cast<size_t>(layout_csr_row_indices) % 16 == 0 &&
            this->head_size % 16 == 0 &&
            this->past_sequence_length == 0);  // This kernel is for prompt only.
  }
};
}  // namespace sparse_attention_v1

inline void SetKernelSharedMemory(const CUDADriverWrapper* driver, CUfunction func) {
  int device = 0;
  CUDA_CALL_THROW(cudaGetDevice(&device));

  int shared_optin = 0;
  CU_CHECK(driver->cuDeviceGetAttribute(&shared_optin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device), driver);
  if (shared_optin > 49152) {
    CU_CHECK(driver->cuFuncSetCacheConfig(func, CU_FUNC_CACHE_PREFER_SHARED), driver);
    CU_CHECK(driver->cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_optin), driver);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
