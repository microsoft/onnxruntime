// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "qlinear_global_average_pool.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"
#include <functional>

namespace onnxruntime {
namespace contrib {

template <typename T8Bits>
Status ComputeQLinearGlobalAvgPool(
    const T8Bits* x,
    float x_scale,
    T8Bits x_zero_point,
    T8Bits* y,
    float y_scale,
    T8Bits y_zero_point,
    int64_t N,
    int64_t C,
    int64_t image_size,
    bool channels_last,
    concurrency::ThreadPool* tp) {
  if (!channels_last || C == 1) {
    auto worker = [=](std::ptrdiff_t first, std::ptrdiff_t last) {
      const T8Bits* input = (const T8Bits*)(x + (first * image_size));
      T8Bits* output = (T8Bits*)(y + first);
      std::vector<int32_t> acc_buffer(MlasQLinearSafePaddingElementCount(sizeof(int32_t), last - first));
      MlasQLinearGlobalAveragePoolNchw(input, x_scale, x_zero_point, output, y_scale, y_zero_point, last - first, image_size, acc_buffer.data());
    };
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(N * C), {1.0 * image_size, 1.0, 8.0 * image_size}, worker);
  } else {
    auto worker = [=](std::ptrdiff_t first, std::ptrdiff_t last) {
      const T8Bits* input = x + first * C * image_size;
      T8Bits* output = y + first * C;
      std::vector<int32_t> acc_buffer(MlasQLinearSafePaddingElementCount(sizeof(int32_t), C));
      std::vector<T8Bits> zero_buffer(MlasQLinearSafePaddingElementCount(sizeof(T8Bits), C), 0);
      MlasQLinearGlobalAveragePoolNhwc(
          input, x_scale, x_zero_point, output, y_scale, y_zero_point,
          last - first, image_size, C, C, acc_buffer.data(), zero_buffer.data());
    };
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(N),
        {1.0 * image_size * C, 1.0 * C, 8.0 * image_size * C},
        worker);
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
