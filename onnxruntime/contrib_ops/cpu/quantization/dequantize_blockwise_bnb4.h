// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "blockwise_quant_block_bnb4.h"

#include <vector>

#include "core/common/safeint.h"
#include "core/framework/float16.h"
#include "core/platform/threadpool.h"
#include <iostream>

namespace onnxruntime {
namespace contrib {

template <typename T, int32_t block_size, int32_t DATA_TYPE>
void QuantizeBlockwiseBnb4(
    uint8_t* dst,  // shape: [(N * K + 1) / 2]
    const T* src,  // shape: [N, K]
    T* absmax,     // shape: [(N * K + block_size - 1) / block_size]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  int32_t numel = N * K;
  int32_t total_block_count = (numel + block_size - 1) / block_size;

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      total_block_count,
      [&](ptrdiff_t block_idx) {
        QuantizeBlockBnb4<T, block_size, DATA_TYPE>(src, dst, absmax[block_idx], block_idx, numel);
      },
      0);
}

#define QuantizeBlockwiseBn4DataTyped(block_size, quant_type)                       \
  if (quant_type == FP4)                                                            \
    QuantizeBlockwiseBnb4<T, block_size, FP4>(dst, src, absmax, N, K, thread_pool); \
  else                                                                              \
    QuantizeBlockwiseBnb4<T, block_size, NF4>(dst, src, absmax, N, K, thread_pool);

template <typename T>
void QuantizeBlockwiseBnb4(
    uint8_t* dst,  // shape: [(N * K + 1) / 2]
    const T* src,  // shape: [N, K]
    T* absmax,     // shape: [(N * K + block_size - 1) / block_size]
    int32_t block_size,
    int32_t quant_type,
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  ORT_ENFORCE(
      quant_type == FP4 || quant_type == NF4,
      "Invalid quant_type, only 0 (FP4) and 1 (NF4) are supported.");

  if (block_size == 16) {
    QuantizeBlockwiseBn4DataTyped(16, quant_type);
  } else if (block_size == 32) {
    QuantizeBlockwiseBn4DataTyped(32, quant_type);
  } else if (block_size == 64) {
    QuantizeBlockwiseBn4DataTyped(64, quant_type);
  } else if (block_size == 128) {
    QuantizeBlockwiseBn4DataTyped(128, quant_type);
  } else if (block_size == 256) {
    QuantizeBlockwiseBn4DataTyped(256, quant_type);
  } else {
    ORT_NOT_IMPLEMENTED("only block size 16, 32, 64, 128, 256 are supported.");
  }
}

#undef QuantizeBlockwiseBn4DataTyped

template <typename T, int32_t block_size, int32_t DATA_TYPE>
void DequantizeBlockwiseBnb4(
    T* dst,              // shape: [N, K]
    const uint8_t* src,  // shape: [(N * K + 1) / 2)]
    const T* absmax,     // shape: [(N * K + block_size - 1) / block_size]
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  int32_t numel = N * K;
  int32_t total_block_count = (numel + block_size - 1) / block_size;

  concurrency::ThreadPool::TryBatchParallelFor(
      thread_pool,
      total_block_count,
      [&](ptrdiff_t block_idx) {
        DequantizeBlockBnb4<T, block_size, DATA_TYPE>(src, dst, absmax[block_idx], block_idx, numel);
      },
      0);
}

#define DequantizeBlockwiseBn4DataTyped(block_size, quant_type)                       \
  if (quant_type == FP4)                                                              \
    DequantizeBlockwiseBnb4<T, block_size, FP4>(dst, src, absmax, N, K, thread_pool); \
  else                                                                                \
    DequantizeBlockwiseBnb4<T, block_size, NF4>(dst, src, absmax, N, K, thread_pool);

template <typename T>
void DequantizeBlockwiseBnb4(
    T* dst,              // shape: [N, K]
    const uint8_t* src,  // shape: [(N * K + 1) / 2)]
    const T* absmax,     // shape: [(N * K + block_size - 1) / block_size]
    int32_t block_size,
    int32_t quant_type,
    int32_t N,
    int32_t K,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  ORT_ENFORCE(
      quant_type == FP4 || quant_type == NF4,
      "Invalid quant_type, only 0 (FP4) and 1 (NF4) are supported.");

  if (block_size == 16) {
    DequantizeBlockwiseBn4DataTyped(16, quant_type);
  } else if (block_size == 32) {
    DequantizeBlockwiseBn4DataTyped(32, quant_type);
  } else if (block_size == 64) {
    DequantizeBlockwiseBn4DataTyped(64, quant_type);
  } else if (block_size == 128) {
    DequantizeBlockwiseBn4DataTyped(128, quant_type);
  } else if (block_size == 256) {
    DequantizeBlockwiseBn4DataTyped(256, quant_type);
  } else {
    ORT_NOT_IMPLEMENTED("only block size 16, 32, 64, 128, 256 are supported.");
  }
}

#undef DequantizeBlockwiseBn4DataTyped

}  // namespace contrib
}  // namespace onnxruntime
