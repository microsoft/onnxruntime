// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <benchmark/benchmark.h>

void ArgsProduct(benchmark::internal::Benchmark* bench,
                 const std::vector<std::vector<int64_t>>& arglists);

std::vector<float> RandomVectorUniform(size_t N, float min_value, float max_value);

std::vector<float> RandomVectorUniform(std::vector<int64_t> shape, float min_value, float max_value);

std::vector<int64_t> BenchArgsVector(benchmark::State& state, size_t& start, size_t count);
