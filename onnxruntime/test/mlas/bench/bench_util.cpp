// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bench_util.h"
#include <numeric>
#include <random>
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

std::vector<float> RandomVectorUniform(size_t N, float min_value, float max_value) {
  if (min_value >= max_value) {
    return std::vector<float>(N, min_value);
  }
  std::default_random_engine generator(static_cast<unsigned>(N));
  std::uniform_real_distribution<float> distribution(min_value, max_value);
  std::vector<float> r(N);
  for (size_t i = 0; i < N; i++) {
    r[i] = distribution(generator);
  }
  return r;
}

std::vector<float> RandomVectorUniform(std::vector<int64_t> shape, float min_value, float max_value) {
  int64_t sz = std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
  if (sz <= 0) {
    throw std::invalid_argument("shape gives size must greater than 0!");
  }
  return RandomVectorUniform(static_cast<size_t>(sz), min_value, max_value);
}

// The Benchmark used here do not contains this as in newer version.
// Use the code from newer version.
void ArgsProduct(benchmark::internal::Benchmark* bench,
                 const std::vector<std::vector<int64_t>>& arglists) {
  std::vector<std::size_t> indices(arglists.size(), 0);
  const std::size_t total = std::accumulate(
      std::begin(arglists), std::end(arglists), std::size_t{1},
      [](const std::size_t res, const std::vector<int64_t>& arglist) {
        return res * arglist.size();
      });
  std::vector<int64_t> args;
  args.reserve(arglists.size());
  for (std::size_t i = 0; i < total; i++) {
    for (std::size_t arg = 0; arg < arglists.size(); arg++) {
      args.push_back(arglists[arg][indices[arg]]);
    }
    bench->Args(args);
    args.clear();

    std::size_t arg = 0;
    do {
      indices[arg] = (indices[arg] + 1) % arglists[arg].size();
    } while (indices[arg++] == 0 && arg < arglists.size());
  }
}
