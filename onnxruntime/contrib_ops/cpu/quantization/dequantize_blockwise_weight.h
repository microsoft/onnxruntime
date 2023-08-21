// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "sub_byte_quant_blob.h"

#include "core/framework/float16.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

template <typename T, int32_t block_size, int32_t bits>
void QuantizeBlockwiseWeight(
    SubByteBlob<block_size, bits>* dst,  // [N, blob_per_K]
    T* scale,                            // [N, blob_per_K]
    uint8_t* zero_points,                // [N, blob_per_K]
    const T* src,                        // [K, N]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  int32_t blob_per_K = (K + block_size - 1) / block_size;
  int32_t task_count = N * blob_per_K;

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      task_count,
      [&](ptrdiff_t task_idx) {
        int32_t n = static_cast<int32_t>(task_idx / blob_per_K);
        int32_t k_block_idx = static_cast<int32_t>(task_idx % blob_per_K);
        int32_t k = k_block_idx * block_size;
        SubByteBlob<block_size, bits>* blob_ptr = dst + task_idx;
        if (nullptr != zero_points) {
          blob_ptr->quant(src + k * N + n, scale[task_idx], zero_points[task_idx], k, K, N);
        } else {
          blob_ptr->quant(src + k * N + n, scale[task_idx], k, K, N);
        }
      },
      0);
}

template <typename T, int32_t block_size, int32_t bits>
void DequantizeBlockwiseWeight(
    T* dst,                                    // [N, K]
    const SubByteBlob<block_size, bits>* src,  // [N, blob_per_K]
    const T* scale,                            // [N, blob_per_K]
    const uint8_t* zero_points,                // [N, blob_per_K]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  int32_t blob_per_K = (K + block_size - 1) / block_size;
  int32_t task_count = N * blob_per_K;

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      task_count,
      [&](ptrdiff_t task_idx) {
        int32_t n = static_cast<int32_t>(task_idx / blob_per_K);
        int32_t k_block_idx = static_cast<int32_t>(task_idx % blob_per_K);
        int32_t k = k_block_idx * block_size;
        const SubByteBlob<block_size, bits>* blob_ptr = src + task_idx;
        if (nullptr != zero_points) {
          blob_ptr->dequant(dst + n * K + k, scale[task_idx], zero_points[task_idx], k, K);
        } else {
          blob_ptr->dequant(dst + n * K + k, scale[task_idx], k, K);
        }
      },
      0);
}

}  // namespace contrib
}  // namespace onnxruntime
