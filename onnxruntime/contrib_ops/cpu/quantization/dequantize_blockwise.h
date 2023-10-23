// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "blockwise_quant_block.h"

#include <vector>

#include "core/common/safeint.h"
#include "core/framework/float16.h"
#include "core/platform/threadpool.h"
#include <iostream>

namespace onnxruntime {
namespace contrib {

template <typename T, int32_t block_size, int32_t bits>
void QuantizeBlockwise(
    uint8_t* dst,          // shape: [ N, block_per_K, block_blob_size ]
    const T* src,          // shape: [K, N]
    T* scale,              // shape: [N * block_per_K]
    uint8_t* zero_points,  // shape: [N * block_per_K] if bits > 4 else [(N *block_per_K + 1) / 2]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  BlockwiseQuantBlock<T, block_size, bits>* dst_blob =
      reinterpret_cast<BlockwiseQuantBlock<T, block_size, bits>*>(dst);

  int32_t block_per_K = (K + block_size - 1) / block_size;
  int32_t total_block_count = N * block_per_K;

  std::vector<uint8_t> zero_points_tmp;  // to avoid race condition
  (void)zero_points_tmp;
  uint8_t* zero_points_tmp_ptr = zero_points;
  if (bits <= 4 && zero_points != nullptr) {
    zero_points_tmp.resize(total_block_count, 0);
    zero_points_tmp_ptr = zero_points_tmp.data();
  }

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      total_block_count,
      [&](ptrdiff_t block_idx) {
        int32_t n = static_cast<int32_t>(block_idx / block_per_K);
        int32_t k_block_idx = static_cast<int32_t>(block_idx % block_per_K);
        int32_t k = k_block_idx * block_size;
        BlockwiseQuantBlock<T, block_size, bits>* blob_ptr = dst_blob + block_idx;
        size_t offset = SafeInt<size_t>(k) * N + n;
        if (nullptr != zero_points_tmp_ptr) {
          blob_ptr->quant(src + offset, scale[block_idx], zero_points_tmp_ptr[block_idx], k, K, N);
        } else {
          blob_ptr->quant(src + offset, scale[block_idx], k, K, N);
        }
      },
      0);

  if (bits <= 4 && zero_points != nullptr) {  // compact zero points
    for (int32_t zp_idx = 0; zp_idx < total_block_count / 2; zp_idx++) {
      zero_points[zp_idx] = ((zero_points_tmp[zp_idx * 2]) | (zero_points_tmp[zp_idx * 2 + 1] << 4));
    }
    if (total_block_count & 1) {
      zero_points[total_block_count / 2] = (zero_points[total_block_count / 2] & 0xf0) | zero_points_tmp[total_block_count - 1];
    }
  }
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
    T* dst,                      // shape: [N, K]
    const uint8_t* src,          // shape: [N, block_per_K, block_blob_size]
    const T* scale,              // shape: [N, block_per_K]
    const uint8_t* zero_points,  // shape: [N, block_per_K] if bits > 4 else [N, (block_per_K + 1) / 2]
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
        size_t offset = SafeInt<size_t>(n) * K + k;
        if (nullptr != zero_points) {
          abort();
          if constexpr (bits > 4) {  // zero point is stored with a byte
            blob_ptr->dequant(dst + offset, scale[task_idx], zero_points[task_idx], k, K);
          } else {  // zero points is stored with 4bits
            uint8_t zp = zero_points[task_idx / 2];
            zp = (task_idx & 1) ? (zp >> 4) : (zp & 0xf);
            blob_ptr->dequant(dst + offset, scale[task_idx], zp, k, K);
          }
        } else {
          blob_ptr->dequant(dst + offset, scale[task_idx], k, K);
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
