// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <benchmark/benchmark.h>

#include <functional>
#include <random>

// Specifies benchmark arguments from the cartesian product of `arglists`, like Benchmark::ArgsProduct().
// `include_filter` is called to determine whether a given set of arguments should be included.
void ArgsProductWithFilter(benchmark::internal::Benchmark* bench,
                           const std::vector<std::vector<int64_t>>& arglists,
                           std::function<bool(const std::vector<int64_t>& args)> include_filter);

template <typename ElementType>
std::vector<ElementType> RandomVectorUniform(
    size_t N,
    ElementType min_value = std::numeric_limits<ElementType>::lowest(),
    ElementType max_value = std::numeric_limits<ElementType>::max()) {
  if (min_value >= max_value) {
    return std::vector<ElementType>(N, min_value);
  }
  std::default_random_engine generator(static_cast<unsigned>(N));
  std::uniform_real_distribution<double> distribution(static_cast<double>(min_value), static_cast<double>(max_value));

  std::vector<ElementType> r(N);
  for (size_t i = 0; i < N; i++) {
    r[i] = static_cast<ElementType>(distribution(generator));
  }
  return r;
}

std::vector<float> RandomVectorUniform(std::vector<int64_t> shape, float min_value, float max_value);

std::vector<int64_t> BenchArgsVector(benchmark::State& state, size_t& start, size_t count);
