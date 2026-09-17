// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <chrono>

#include "contrib_ops/cuda/sparse/sparse_attention_indexer_impl.h"

namespace onnxruntime {
namespace test {
namespace {

using contrib::cuda::GetQsaWorkspaceFloatCount;
using contrib::cuda::GetQsaWorkspaceIntCount;
using contrib::cuda::LaunchQsaSparseAttentionIndexer;
using contrib::cuda::SparseAttentionIndexerParams;

struct QsaBuffers {
  float* query{};
  float* key{};
  float* weight{};
  float* cosine{};
  float* sine{};
  bool* mask{};
  float* key_cache{};
  int32_t* selected{};
  float* float_workspace{};
  int32_t* int_workspace{};

  ~QsaBuffers() {
    cudaFree(int_workspace);
    cudaFree(float_workspace);
    cudaFree(selected);
    cudaFree(key_cache);
    cudaFree(mask);
    cudaFree(sine);
    cudaFree(cosine);
    cudaFree(weight);
    cudaFree(key);
    cudaFree(query);
  }
};

bool Allocate(void** output, size_t bytes, benchmark::State& state) {
  const cudaError_t error = cudaMalloc(output, bytes);
  if (error != cudaSuccess) {
    state.SkipWithError(cudaGetErrorString(error));
    return false;
  }
  return true;
}

void BM_SparseAttentionIndexerQsaDecode(benchmark::State& state) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    state.SkipWithError("CUDA device unavailable");
    return;
  }

  constexpr int kHeadSize = 128;
  constexpr int kNumHeads = 4;
  constexpr int kTokenBudget = 2048;
  constexpr int kCompressRatio = 4;
  const int context_length = static_cast<int>(state.range(0));

  SparseAttentionIndexerParams params;
  params.batch_size = 1;
  params.sequence_length = 1;
  params.num_heads = kNumHeads;
  params.head_size = kHeadSize;
  params.rotary_width = kHeadSize;
  params.max_rotary_length = context_length;
  params.compress_ratio = kCompressRatio;
  params.capacity = kTokenBudget + kCompressRatio - 1;
  params.scale = 1.0f;
  params.past_sequence_length = context_length - 1;
  params.total_sequence_length = context_length;
  params.past_key_capacity = context_length;
  params.key_cache_capacity = context_length;
  params.max_block_count = context_length / kCompressRatio;
  params.block_topk = kTokenBudget / kCompressRatio;

  QsaBuffers buffers;
  if (!Allocate(reinterpret_cast<void**>(&buffers.query), kNumHeads * kHeadSize * sizeof(float), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.key), kHeadSize * sizeof(float), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.weight), kHeadSize * sizeof(float), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.cosine),
                static_cast<size_t>(context_length) * kHeadSize * sizeof(float), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.sine),
                static_cast<size_t>(context_length) * kHeadSize * sizeof(float), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.mask), context_length * sizeof(bool), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.key_cache),
                static_cast<size_t>(context_length) * kHeadSize * sizeof(float), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.selected), params.capacity * sizeof(int32_t), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.float_workspace),
                GetQsaWorkspaceFloatCount(params) * sizeof(float), state) ||
      !Allocate(reinterpret_cast<void**>(&buffers.int_workspace),
                GetQsaWorkspaceIntCount(params) * sizeof(int32_t), state)) {
    return;
  }

  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    state.SkipWithError("cudaStreamCreate failed");
    return;
  }
  cudaMemsetAsync(buffers.query, 0, kNumHeads * kHeadSize * sizeof(float), stream);
  cudaMemsetAsync(buffers.key, 0, kHeadSize * sizeof(float), stream);
  cudaMemsetAsync(buffers.weight, 0, kHeadSize * sizeof(float), stream);
  cudaMemsetAsync(buffers.cosine, 0, static_cast<size_t>(context_length) * kHeadSize * sizeof(float), stream);
  cudaMemsetAsync(buffers.sine, 0, static_cast<size_t>(context_length) * kHeadSize * sizeof(float), stream);
  cudaMemsetAsync(buffers.mask, 1, context_length * sizeof(bool), stream);
  cudaMemsetAsync(buffers.key_cache, 0, static_cast<size_t>(context_length) * kHeadSize * sizeof(float), stream);
  cudaStreamSynchronize(stream);

  const auto launch = [&]() {
    return LaunchQsaSparseAttentionIndexer<float>(
        stream, params, buffers.query, buffers.key, buffers.weight, buffers.cosine, buffers.sine,
        buffers.mask, buffers.key_cache, buffers.selected, buffers.key_cache,
        buffers.float_workspace, buffers.int_workspace);
  };
  for (int i = 0; i < 5; ++i) {
    if (!launch().IsOK()) {
      state.SkipWithError("SparseAttentionIndexer warmup failed");
      cudaStreamDestroy(stream);
      return;
    }
  }
  cudaStreamSynchronize(stream);

  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    if (!launch().IsOK() || cudaStreamSynchronize(stream) != cudaSuccess) {
      state.SkipWithError("SparseAttentionIndexer execution failed");
      break;
    }
    const auto end = std::chrono::steady_clock::now();
    state.SetIterationTime(std::chrono::duration<double>(end - start).count());
  }

  state.counters["context"] = context_length;
  state.counters["selected"] = kTokenBudget;
  cudaStreamDestroy(stream);
}

BENCHMARK(BM_SparseAttentionIndexerQsaDecode)
    ->ArgName("context")
    ->Arg(8192)
    ->Arg(32768)
    ->Arg(65536)
    ->Arg(131072)
    ->Arg(262144)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

}  // namespace
}  // namespace test
}  // namespace onnxruntime
