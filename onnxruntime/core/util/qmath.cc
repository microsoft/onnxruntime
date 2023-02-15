// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qmath.h"

namespace onnxruntime {

template<>
void ParQuantizeLinear(const float* Input,
                       FloatE4M3* Output,
                       size_t N,
                       float Scale,
                       FloatE4M3 /*ZeroPoint*/,
                       concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(float)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);
    for(;begin_idx<end_idx;++begin_idx) {
      Output[begin_idx] = FloatE4M3(Input[begin_idx] / Scale);
    }
  });
}

template<>
void ParQuantizeLinear(const float* Input,
                       FloatE5M2* Output,
                       size_t N,
                       float Scale,
                       FloatE5M2 /*ZeroPoint*/,
                       concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(float)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);
    for(;begin_idx<end_idx;++begin_idx) {
      Output[begin_idx] = FloatE5M2(Input[begin_idx] / Scale);
    }
  });
}

}  // namespace onnxruntime
