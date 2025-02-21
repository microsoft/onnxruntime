// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "benchmark/benchmark.h"
#include "bench_util.h"
#include "core/framework/float16.h"

using namespace onnxruntime;

template <typename T>
void RunRoPEBenchmark(size_t rotary_emb_dim, bool interleaved, benchmark::State& state) {
  const float Pi = 2 * std::acos(0.0f);

  std::vector<T> input(rotary_emb_dim);
  size_t table_len = interleaved ? rotary_emb_dim / 2 : rotary_emb_dim;
  std::vector<T> sin_data(table_len);
  std::vector<T> cos_data(table_len);
  std::vector<T> output_ref(rotary_emb_dim), output_impl(rotary_emb_dim);

  for (size_t i = 0; i < rotary_emb_dim; ++i) {
    input[i] = static_cast<T>(1.f * i + 1);
  }
  for (size_t i = 0; i < table_len; ++i) {
    float theta = (float)i / 1000 * Pi;
    sin_data[i] = (T)std::sin(theta);
    cos_data[i] = (T)std::cos(theta);
  }

  // warm up run
  MlasRotaryEmbedOneRow<T>(&input[0], &sin_data[0], &cos_data[0], rotary_emb_dim, interleaved, &output_impl[0]);

  for (auto _ : state) {
    MlasRotaryEmbedOneRow<T>(&input[0], &sin_data[0], &cos_data[0], rotary_emb_dim, interleaved, &output_impl[0]);
  }
}

template <typename T>
void RoPE(benchmark::State& state) {
  using onnxruntime::narrow;

  const auto rotary_emb_dim = narrow<size_t>(state.range(0));
  const auto interleaved = narrow<bool>(state.range(1));

  RunRoPEBenchmark<T>(rotary_emb_dim, interleaved, state);
}

template <typename T>
static void RoPEArgs(benchmark::internal::Benchmark* b) {
  b->ArgNames({"rotary_emb_dim", "interleaved"});

  b->ArgsProduct({
      {128, 256, 512, 1024},            // rotary_emb_dim
      {int64_t{false}, int64_t{true}},  // interleaved
  });
}

BENCHMARK(RoPE<float>)->Apply(RoPEArgs<float>)->UseRealTime();
BENCHMARK(RoPE<MLFloat16>)->Apply(RoPEArgs<MLFloat16>)->UseRealTime();
