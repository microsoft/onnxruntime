// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bench_util.h"
#include <numeric>
#include <stdexcept>

std::vector<int64_t> BenchArgsVector(benchmark::State& state, size_t& start, size_t count) {
  std::vector<int64_t> shape;
  shape.reserve(count);
  for (size_t axis = 0; axis < count; ++axis) {
    shape.emplace_back(state.range(start + axis));
  }
  start += count;
  return shape;
}

std::vector<float> RandomVectorUniform(std::vector<int64_t> shape, float min_value, float max_value) {
  int64_t sz = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  if (sz <= 0) {
    throw std::invalid_argument("shape gives size must greater than 0!");
  }
  return RandomVectorUniform(static_cast<size_t>(sz), min_value, max_value);
}
