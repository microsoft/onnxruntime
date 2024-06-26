// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "contrib_ops/cuda/sparse/sparse_attention_v1/sparse_attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace sparse_attention_v2 {

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
  int stride_qt;
  int stride_qh;
  int stride_kb;
  int stride_kt;
  int stride_kh;
  int stride_vb;
  int stride_vt;
  int stride_vh;
  int stride_ob;
  int stride_ot;
  int stride_oh;

  int q_k_ratio;

  int active_q_blocks;
  const int* q_batch_starts;
  const int* q_batch_ends;
  const int* k_batch_starts;
  const int* k_batch_ends;
  const int* q_batch_ids;
  const int* q_start_sids;

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
      int num_layout,
      int active_q_blocks,
      const int* q_batch_starts,
      const int* q_batch_ends,
      const int* k_batch_starts,
      const int* k_batch_ends,
      const int* q_batch_ids,
      const int* q_start_sids) {
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

    // Q can be either BNSH or BSNH format
    this->stride_qb = this->num_heads * this->sequence_length * this->head_size;
    this->stride_qh = (is_q_bnsh ? this->sequence_length : this->num_heads) * this->head_size;
    this->stride_qt = this->head_size;

    // When kv buffer has max sequence length, stride should match max sequence length.
    // KV cache is in BNSH format
    this->stride_kb = this->kv_num_heads * max_cache_sequence_length * this->head_size;
    this->stride_kh = max_cache_sequence_length * this->head_size;
    this->stride_kt = this->head_size;
    this->stride_vb = this->kv_num_heads * max_cache_sequence_length * this->head_size;
    this->stride_vh = max_cache_sequence_length * this->head_size;
    this->stride_vt = this->head_size;

    // Output is BSNH format
    this->stride_ob = this->sequence_length * this->num_heads * this->head_size;
    this->stride_oh = this->head_size;
    this->stride_ot = this->num_heads * this->head_size;

    this->q_k_ratio = this->num_heads / this->kv_num_heads;

    this->active_q_blocks = active_q_blocks;
    this->q_batch_starts = q_batch_starts;
    this->q_batch_ends = q_batch_ends;
    this->k_batch_starts = k_batch_starts;
    this->k_batch_ends = k_batch_ends;
    this->q_batch_ids = q_batch_ids;
    this->q_start_sids = q_start_sids;
  }

  Status LaunchKernel(CUfunction f, int threads_per_block, unsigned int sharedMemBytes) {
    ORT_ENFORCE(f != nullptr, "Kernel shall be loaded before calling LaunchKernel.");

    if (!Valididate()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SparseAttentionParams is not valid.");
    }

    void* args[29] = {
        &output, &q, &k, &v,
        &q_batch_starts, &q_batch_ends, &k_batch_starts, &k_batch_ends, &q_batch_ids, &q_start_sids,
        &layout_csr_row_indices, &layout_csr_col_indices, &layout_row_stride_h, &layout_col_stride_h,
        &stride_qb, &stride_qt, &stride_qh, &stride_kb, &stride_kt, &stride_kh,
        &stride_vb, &stride_vt, &stride_vh, &stride_ob, &stride_ot, &stride_oh,
        &q_k_ratio, &num_layout, &scale};

    unsigned int gridDimX = active_q_blocks;
    unsigned int gridDimY = num_heads;
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

    DUMP_TENSOR("q_batch_starts", q_batch_starts, 1, batch_size);
    DUMP_TENSOR("q_batch_ends", q_batch_ends, 1, batch_size);
    DUMP_TENSOR("k_batch_starts", k_batch_starts, 1, batch_size);
    DUMP_TENSOR("k_batch_ends", k_batch_ends, 1, batch_size);
    DUMP_TENSOR("q_batch_ids", q_batch_ids, 1, active_q_blocks);
    DUMP_TENSOR("q_start_sids", q_start_sids, 1, active_q_blocks);

    printf(
        "layout_row_stride_h=%d, layout_col_stride_h=%d, num_layout=%d, scale=%f, is_q_bnsh=%d,\n"
        "stride_qb=%d, stride_qt=%d, stride_qh=%d, stride_kb=%d, stride_kt=%d, stride_kh=%d,\n"
        "stride_vb=%d, stride_vt=%d, stride_vh=%d, stride_ob=%d, stride_ot=%d, stride_oh=%d,\n"
        "num_heads=%d, kv_num_heads=%d, total_sequence_length=%d, past_sequence_length=%d\n"
        "output=%p, q=%p, k=%p, v=%p, layout_csr_row_indices=%p, layout_csr_col_indices=%p\n"
        "q_batch_starts=%p, q_batch_ends=%p, k_batch_starts=%p, k_batch_ends=%p, q_batch_ids=%p, q_start_sids=%p active_q_blocks=%d\n",
        layout_row_stride_h, layout_col_stride_h, num_layout, scale, static_cast<int>(is_q_bnsh),
        stride_qb, stride_qt, stride_qh, stride_kb, stride_kt, stride_kh,
        stride_vb, stride_vt, stride_vh, stride_ob, stride_ot, stride_oh,
        num_heads, kv_num_heads, total_sequence_length, past_sequence_length,
        output, q, k, v, layout_csr_row_indices, layout_csr_col_indices,
        q_batch_starts, q_batch_ends, k_batch_starts, k_batch_ends, q_batch_ids, q_start_sids, active_q_blocks);

    printf("gridDimX=%d gridDimY=%d threads_per_block=%d sharedMemBytes=%d\n",
           gridDimX, gridDimY, threads_per_block, sharedMemBytes);
#endif

    const CUDADriverWrapper* driver = CUDADriverWrapper::GetInstance();
    CU_CHECK(driver->cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, threads_per_block, 1, 1, sharedMemBytes,
                                    static_cast<CUstream>(this->ort_stream->GetHandle()),
                                    args, NULL),
             driver);
    return Status::OK();
  }

  bool Valididate() {
    // Check pointers are aligned to 16 bytes (we used that to hint the compiler to generate aligned loads/stores)
    return (reinterpret_cast<size_t>(output) % 16 == 0 &&
            reinterpret_cast<size_t>(q) % 16 == 0 &&
            reinterpret_cast<size_t>(k) % 16 == 0 &&
            reinterpret_cast<size_t>(v) % 16 == 0 &&
            reinterpret_cast<size_t>(layout_csr_col_indices) % 16 == 0 &&
            reinterpret_cast<size_t>(layout_csr_row_indices) % 16 == 0 &&
            reinterpret_cast<size_t>(q_batch_starts) % 16 == 0 &&
            reinterpret_cast<size_t>(q_batch_ends) % 16 == 0 &&
            reinterpret_cast<size_t>(k_batch_starts) % 16 == 0 &&
            reinterpret_cast<size_t>(k_batch_ends) % 16 == 0 &&
            reinterpret_cast<size_t>(q_batch_ids) % 16 == 0 &&
            reinterpret_cast<size_t>(q_start_sids) % 16 == 0 &&
            this->head_size % 16 == 0);
  }
};

}  // namespace sparse_attention_v2
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
