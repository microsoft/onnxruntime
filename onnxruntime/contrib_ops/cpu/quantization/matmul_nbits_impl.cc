// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cpu/quantization/matmul_nbits_impl.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "core/common/common.h"
#include "core/framework/float16.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

template <class T, class zeroT>
void Dequantize4BitsKernelReOrder(
    T* output, const uint8_t* quant_data, const T* scale_data,
    const zeroT* zero_points, const int32_t* reorder_idx, int block_size,
    int groups_per_threadblock, int total_groups, int out_rows, int out_cols,
    int blockIdx_x, int threadIdx_x) {
  const int group_id = blockIdx_x * groups_per_threadblock + ((threadIdx_x * 8) / block_size);
  if (group_id >= total_groups) {
    return;
  }
  const int scales_shape_x = (out_cols + block_size - 1) / block_size;
  const int zero_point_shape_x = (scales_shape_x + 1) / 2;

  int n_idx = group_id / scales_shape_x;
  int kb_idx = group_id % scales_shape_x;
  int element_offset = group_id * block_size + ((threadIdx_x * 8) & (block_size - 1));

  const int out_x = element_offset % (scales_shape_x * block_size);
  const int out_y = element_offset / (scales_shape_x * block_size);
  if (out_y >= out_rows || out_x >= out_cols) {
    return;
  }
  T* output_i = output + out_y * out_cols + out_x;
  uint32_t quant_value = *(reinterpret_cast<const uint32_t*>(quant_data + element_offset / 2));
  const int remain_x = std::min(8, out_cols - out_x);
  const int32_t* reorder_idx_with_off = reorder_idx + kb_idx * block_size + ((threadIdx_x * 8) & (block_size - 1));
  for (int i = 0; i < remain_x; i++) {
    int32_t rid = reorder_idx ? reorder_idx_with_off[i] : kb_idx;
    T scale = *(scale_data + n_idx * scales_shape_x + rid);
    float zp_f = 8;
    if (zero_points) {
      if constexpr (std::is_same_v<zeroT, T>) {
        zp_f = *(zero_points + n_idx * scales_shape_x + rid);
      } else {
        uint8_t zp = 8;
        zp = zero_points[n_idx * zero_point_shape_x + rid / 2];
        zp = (rid & 0x01) ? (zp >> 4) : (zp & 0x0f);
      }
    }

    if constexpr (std::is_same_v<T, MLFloat16>) {
      T zp_adjust = -scale * MLFloat16(zp_f);
      output_i[i] = static_cast<float>((quant_value >> (4 * i)) & 0xF) * scale + zp_adjust;
    } else {
      T zp_adjust = -scale * zp_f;
      output_i[i] = T((quant_value >> (4 * i)) & 0xF) * scale + zp_adjust;
    }
  }
}

template <typename inputT, typename zeroT>
void DequantizeBlockwise(
    inputT* output,              // dequantized output
    const uint8_t* quant_data,   // quantized input
    const inputT* scales_data,   // quantization scales
    const zeroT* zero_points,    // quantization zero points
    const int32_t* reorder_idx,  // reorder_idx for groupwise quantization
    int32_t block_size,          // quantization block size
    bool,                        // columnwise quantization or row-wise
    int32_t K,                   // number of rows in quantized input
    int32_t N,                   // number of columns in quantized input
    onnxruntime::concurrency::ThreadPool* pool) {
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  constexpr int element_per_thread = 8;
  int groups_per_threadblock = 256 * element_per_thread / block_size;
  int groups_per_K = ceildiv(K, block_size);
  int total_groups = N * groups_per_K;  // total elemenets in quant_data
  int blocks_per_grid = static_cast<int>(ceildiv(total_groups, groups_per_threadblock));
  concurrency::ThreadPool::TrySimpleParallelFor(
      pool, static_cast<std::ptrdiff_t>(blocks_per_grid),
      [&](std::ptrdiff_t block_id) {
        for (int j = 0; j < 256; j++) {
          Dequantize4BitsKernelReOrder(output, quant_data, scales_data, zero_points,
                                       reorder_idx, block_size, groups_per_threadblock,
                                       total_groups, N, K, static_cast<int>(block_id), j);
        }
      });
}

template void DequantizeBlockwise<float, uint8_t>(
    float* output, const uint8_t* quant_data, const float* scales_data,
    const uint8_t* zero_points, const int32_t* reorder_idx, int32_t block_size,
    bool columnwise, int32_t K, int32_t N, onnxruntime::concurrency::ThreadPool* thread_pool);

template void DequantizeBlockwise<float, float>(
    float* output, const uint8_t* quant_data, const float* scales_data,
    const float* zero_points, const int32_t* reorder_idx, int32_t block_size,
    bool columnwise, int32_t K, int32_t N, onnxruntime::concurrency::ThreadPool* thread_pool);

}  // namespace contrib
}  // namespace onnxruntime
