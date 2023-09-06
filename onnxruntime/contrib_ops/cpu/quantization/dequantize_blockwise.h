// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "blockwise_quant_block.h"

#include "core/framework/float16.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

template <typename T, int32_t block_size, int32_t bits>
void QuantizeBlockwise(
    uint8_t* dst,          // shape: [ N, block_per_K, block_blob_size ]
    const T* src,          // shape: [K, N]
    T* scale,              // shape: [N, block_per_K]
    uint8_t* zero_points,  // shape: [N, block_per_K]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  BlockwiseQuantBlock<T, block_size, bits>* dst_blob =
      reinterpret_cast<BlockwiseQuantBlock<T, block_size, bits>*>(dst);

  int32_t block_per_K = (K + block_size - 1) / block_size;
  int32_t task_count = N * block_per_K;

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      task_count,
      [&](ptrdiff_t task_idx) {
        int32_t n = static_cast<int32_t>(task_idx / block_per_K);
        int32_t k_block_idx = static_cast<int32_t>(task_idx % block_per_K);
        int32_t k = k_block_idx * block_size;
        BlockwiseQuantBlock<T, block_size, bits>* blob_ptr = dst_blob + task_idx;
        if (nullptr != zero_points) {
          blob_ptr->quant(src + k * N + n, scale[task_idx], zero_points[task_idx], k, K, N);
        } else {
          blob_ptr->quant(src + k * N + n, scale[task_idx], k, K, N);
        }
      },
      0);
}

#define QuantizeBlockwise4Bits(block_size) \
  QuantizeBlockwise<T, block_size, 4>(dst, src, scale, zero_points, N, K, thread_pool);

template <typename T>
void QuantizeBlockwise(
    uint8_t* dst,          // shape: [ N, block_per_K, block_blob_size ]
    const T* src,          // shape: [K, N]
    T* scale,              // shape: [N, block_per_K]
    uint8_t* zero_points,  // shape: [N, block_per_K]
    int32_t block_size,
    int32_t bits,
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  ORT_ENFORCE(bits == 4, "only 4 bits is supported now");

  if (16 == block_size) {
    QuantizeBlockwise4Bits(16);
  } else if (32 == block_size) {
    QuantizeBlockwise4Bits(32);
  } else if (64 == block_size) {
    QuantizeBlockwise4Bits(64);
  } else if (128 == block_size) {
    QuantizeBlockwise4Bits(128);
  } else if (256 == block_size) {
    QuantizeBlockwise4Bits(256);
  } else {
    ORT_NOT_IMPLEMENTED("only block size 16, 32, 64, 128, 256 are supported.");
  }
}

#undef QuantizeBlockwise4Bits

template <typename T, int32_t block_size, int32_t bits>
void DequantizeBlockwise(
    T* dst,                      // [N, K]
    const uint8_t* src,          // [N, block_per_K, block_blob_size]
    const T* scale,              // [N, block_per_K]
    const uint8_t* zero_points,  // [N, block_per_K]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  int32_t block_per_K = (K + block_size - 1) / block_size;
  int32_t task_count = N * block_per_K;

  const BlockwiseQuantBlock<T, block_size, bits>* src_blob =
      reinterpret_cast<const BlockwiseQuantBlock<T, block_size, bits>*>(src);

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      task_count,
      [&](ptrdiff_t task_idx) {
        int32_t n = static_cast<int32_t>(task_idx / block_per_K);
        int32_t k_block_idx = static_cast<int32_t>(task_idx % block_per_K);
        int32_t k = k_block_idx * block_size;
        const BlockwiseQuantBlock<T, block_size, bits>* blob_ptr = src_blob + task_idx;
        if (nullptr != zero_points) {
          blob_ptr->dequant(dst + n * K + k, scale[task_idx], zero_points[task_idx], k, K);
        } else {
          blob_ptr->dequant(dst + n * K + k, scale[task_idx], k, K);
        }
      },
      0);
}

#define DequantizeBlockwise4Bits(block_size) \
  DequantizeBlockwise<T, block_size, 4>(dst, src, scale, zero_points, N, K, thread_pool);

template <typename T>
void DequantizeBlockwise(
    T* dst,                      // [N, K]
    const uint8_t* src,          // [N, block_per_K, block_blob_size]
    const T* scale,              // [N, block_per_K]
    const uint8_t* zero_points,  // [N, block_per_K]
    int32_t block_size,
    int32_t bits,
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  ORT_ENFORCE(bits == 4, "only 4 bits is supported now");

  if (16 == block_size) {
    DequantizeBlockwise4Bits(16);
  } else if (32 == block_size) {
    DequantizeBlockwise4Bits(32);
  } else if (64 == block_size) {
    DequantizeBlockwise4Bits(64);
  } else if (128 == block_size) {
    DequantizeBlockwise4Bits(128);
  } else if (256 == block_size) {
    DequantizeBlockwise4Bits(256);
  } else {
    ORT_NOT_IMPLEMENTED("only block size 16, 32, 64, 128, 256 are supported.");
  }
}

#undef DequantizeBlockwise4Bits

}  // namespace contrib
}  // namespace onnxruntime
