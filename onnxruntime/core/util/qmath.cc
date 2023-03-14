// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qmath.h"

namespace onnxruntime {

#define PAR_QUANTIZE_LINEAR(FLOAT8_TYPE) \
    template<> \
    void ParQuantizeLinear(const float* Input, \
                          FLOAT8_TYPE* Output, \
                          size_t N, \
                          float Scale, \
                          FLOAT8_TYPE /*ZeroPoint*/, \
                          concurrency::ThreadPool* thread_pool) { \
      constexpr std::ptrdiff_t block_size = 128; \
      const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size; \
      const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(float)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0}; \
      concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) { \
        auto begin_idx = begin * block_size; \
        auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size); \
        for(;begin_idx<end_idx;++begin_idx) { \
          Output[begin_idx] = FLOAT8_TYPE(Input[begin_idx] / Scale); \
        } \
      }); \
    }

PAR_QUANTIZE_LINEAR(Float8E4M3FN)
PAR_QUANTIZE_LINEAR(Float8E4M3FNUZ)
PAR_QUANTIZE_LINEAR(Float8E5M2)
PAR_QUANTIZE_LINEAR(Float8E5M2FNUZ)

}  // namespace onnxruntime
