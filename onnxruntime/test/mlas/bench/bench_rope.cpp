// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "benchmark/benchmark.h"
#include "bench_util.h"

void RunRoPEBenchmark(size_t rotary_emb_dim, bool interleaved, benchmark::State& state) {
  const float Pi = 2 * std::acos(0.0f);

  std::vector<float> input(rotary_emb_dim);
  size_t table_len = interleaved ? rotary_emb_dim / 2 : rotary_emb_dim;
  std::vector<float> sin_data(table_len);
  std::vector<float> cos_data(table_len);
  std::vector<float> output_ref(rotary_emb_dim), output_impl(rotary_emb_dim);

  for (size_t i = 0; i < rotary_emb_dim; ++i) {
    input[i] = static_cast<float>(i + 1);
  }
  for (size_t i = 0; i < table_len; ++i) {
    float theta = (float)i / 1000 * Pi;
    sin_data[i] = std::sin(theta);
    cos_data[i] = std::cos(theta);
  }

  // warm up run
  MlasRotaryEmbedOneRow<float>(&input[0], &sin_data[0], &cos_data[0], rotary_emb_dim, interleaved, &output_impl[0]);

  for (auto _ : state) {
    MlasRotaryEmbedOneRow<float>(&input[0], &sin_data[0], &cos_data[0], rotary_emb_dim, interleaved, &output_impl[0]);
  }
}

void RoPE(benchmark::State& state) {
  using onnxruntime::narrow;

  const auto rotary_emb_dim = narrow<size_t>(state.range(0));
  const auto interleaved = narrow<bool>(state.range(1));

  RunRoPEBenchmark(rotary_emb_dim, interleaved, state);
}

static void RoPEArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"rotary_emb_dim", "interleaved"});

  b->ArgsProduct({
      {128, 256, 512, 1024},            // rotary_emb_dim
      {int64_t{false}, int64_t{true}},  // interleaved
  });
}

BENCHMARK(RoPE)->Apply(RoPEArgs)->UseRealTime();
