// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/common/narrow.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/float8.h"
#include <cmath>

namespace onnxruntime {

inline float RoundHalfToEven(float input) {
  if (!std::isfinite(input)) {
    return input;
  }
  // std::remainder returns x - n, where n is the integral value nearest to x. When |x - n| = 0.5, n is chosen to be even
  return input - std::remainderf(input, 1.f);
}

template <typename T>
struct is_quant_type : std::false_type {};

template <>
struct is_quant_type<int8_t> : std::true_type {};

template <>
struct is_quant_type<uint8_t> : std::true_type {};

// Define max number of parallel threads for computing min max values
// This is hacky. Ideally we should let the thread pool handle work partition.
// Unfortunately I can't find an elegant way for aggregation at this point.
#define MAX_DEGREE_OF_PAR_FOR_MINMAX 32

struct FloatMinMax {
  float min;
  float max;
};

// ReduceRange and Symmetric is for test only
template <typename QType,
          bool ReduceRange = false,
          bool Symmetric = false,
          typename std::enable_if<is_quant_type<QType>::value, int>::type = 0>
void GetQuantizationParameter(const float* data, int64_t num_of_elements, float& scale, QType& zp, concurrency::ThreadPool* thread_pool) {
  FloatMinMax aggregate[MAX_DEGREE_OF_PAR_FOR_MINMAX];

  // Min max operation granularity: AVX512 can potentially handle 64 ~ 128 floats
  // per iteration.
  constexpr int granularity = 128;
  std::ptrdiff_t block_size;
  std::ptrdiff_t num_blocks;
  if (concurrency::ThreadPool::ShouldParallelize(thread_pool) && num_of_elements > granularity) {
    block_size = onnxruntime::narrow<std::ptrdiff_t>((num_of_elements + MAX_DEGREE_OF_PAR_FOR_MINMAX - 1) / MAX_DEGREE_OF_PAR_FOR_MINMAX);
    block_size = (block_size + granularity - 1) / granularity * granularity;
    num_blocks = onnxruntime::narrow<std::ptrdiff_t>((num_of_elements + block_size - 1) / block_size);
  } else {
    num_blocks = 1;
    block_size = onnxruntime::narrow<std::ptrdiff_t>(num_of_elements);
  }

  for (int i = 0; i < num_blocks; i++) {
    aggregate[i].min = std::numeric_limits<float>::max();
    aggregate[i].max = std::numeric_limits<float>::lowest();
  }

  const TensorOpCost unit_cost{static_cast<double>(block_size) * sizeof(float), 2.0, static_cast<double>(block_size)};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(std::ptrdiff_t(num_of_elements), end * block_size);
    auto agg_idx = begin % num_blocks;
    MlasFindMinMaxElement(&(data[begin_idx]), &aggregate[agg_idx].min, &aggregate[agg_idx].max, end_idx - begin_idx);
  });

  float& min = aggregate[0].min;
  float& max = aggregate[0].max;
  for (int i = 1; i < num_blocks; i++) {
    min = std::min(min, aggregate[i].min);
    max = std::max(max, aggregate[i].max);
  }
  // ensure the input range includes zero
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);

  // find scale and zero point
  QType qmin = std::numeric_limits<QType>::min();
  QType qmax = std::numeric_limits<QType>::max();
  if (std::is_same<QType, int8_t>::value) {
    if (ReduceRange) {
      qmin = static_cast<QType>(-64);
      qmax = static_cast<QType>(64);
    }

    if (Symmetric) {
      zp = 0;
      float max_value = std::max(max, -min);
      scale = max_value > 0 ? max_value / qmax : 1.f;
      return;
    }
  }
  scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);

  float initial_zero_point = qmin - min / scale;
  zp = static_cast<QType>(RoundHalfToEven(std::max(float(qmin), std::min(float(qmax), initial_zero_point))));
}

/**
 * @brief Run MlasQuantizeLinear in parallel, with provided thread pool
 */

template <typename OutputType>
#if !defined(DISABLE_FLOAT8_TYPES)
typename std::enable_if<!boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputType>::value, void>::type
#else
void
#endif
ParQuantizeLinearStd(const float* Input,
                     OutputType* Output,
                     size_t N,
                     float Scale,
                     OutputType ZeroPoint,
                     concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(float)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);
    MlasQuantizeLinear(&(Input[begin_idx]), &(Output[begin_idx]), end_idx - begin_idx, Scale, ZeroPoint);
  });
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename OutputFloat8Type>
typename std::enable_if<boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputFloat8Type>::value, void>::type
ParQuantizeLinearSat(const float* Input,
                     OutputFloat8Type* Output,
                     size_t N,
                     float Scale,
                     const OutputFloat8Type& /* ORT_UNUSED_PARAMETER(ZeroPoint) */,
                     bool saturate,
                     concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(float)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);
    for (; begin_idx < end_idx; ++begin_idx) {
      Output[begin_idx] = OutputFloat8Type(Input[begin_idx] / Scale, saturate);
    }
  });
}

#endif

}  // namespace onnxruntime
