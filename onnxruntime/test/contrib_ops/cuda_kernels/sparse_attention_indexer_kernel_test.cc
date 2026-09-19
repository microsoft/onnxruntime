// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime_api.h>

#include <algorithm>
#include <string>
#include <vector>

#include "contrib_ops/cuda/sparse/sparse_attention_indexer_impl.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

TEST(SparseAttentionIndexerCudaKernelTest, QsaWorkspaceIncludesPreparedQuery) {
  SparseAttentionIndexerParams params;
  params.batch_size = 2;
  params.sequence_length = 3;
  params.num_heads = 4;
  params.head_size = 128;
  params.max_block_count = 2048;

  constexpr size_t kRows = 6;
  constexpr size_t kPreparedQueryElements = kRows * 4 * 128;
  constexpr size_t kBlockScoreElements = kRows * 2048;
  EXPECT_EQ(GetQsaWorkspaceFloatCount(params), kPreparedQueryElements + kBlockScoreElements);
}

TEST(SparseAttentionIndexerCudaKernelTest, QsaLongContextPerformanceRegression) {
  constexpr int kContextLength = 8192;
  constexpr int kHeadSize = 128;
  constexpr int kNumHeads = 4;
  constexpr int kTokenBudget = 2048;
  constexpr int kCompressRatio = 4;
  constexpr int kThreads = 128;
  constexpr int kWarmupIterations = 10;
  constexpr int kTimedIterations = 50;

  SparseAttentionIndexerParams params;
  params.batch_size = 1;
  params.sequence_length = 1;
  params.num_heads = kNumHeads;
  params.head_size = kHeadSize;
  params.rotary_width = kHeadSize;
  params.max_rotary_length = kContextLength;
  params.compress_ratio = kCompressRatio;
  params.capacity = kTokenBudget + kCompressRatio - 1;
  params.scale = 0.5f;
  params.past_sequence_length = kContextLength - 1;
  params.total_sequence_length = kContextLength;
  params.past_key_capacity = kContextLength;
  params.key_cache_capacity = kContextLength;
  params.max_block_count = kContextLength / kCompressRatio;
  params.block_topk = kTokenBudget / kCompressRatio;

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

  float* query = nullptr;
  float* key = nullptr;
  float* weight = nullptr;
  float* cosine = nullptr;
  float* sine = nullptr;
  int64_t* mask = nullptr;
  float* key_cache = nullptr;
  int32_t* selected = nullptr;
  float* float_workspace = nullptr;
  int32_t* int_workspace = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&query), kNumHeads * kHeadSize * sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&key), kHeadSize * sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&weight), kHeadSize * sizeof(float)));
  ASSERT_EQ(cudaSuccess,
            cudaMalloc(reinterpret_cast<void**>(&cosine), kContextLength * kHeadSize * sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&sine), kContextLength * kHeadSize * sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&mask), kContextLength * sizeof(int64_t)));
  ASSERT_EQ(cudaSuccess,
            cudaMalloc(reinterpret_cast<void**>(&key_cache), kContextLength * kHeadSize * sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&selected), params.capacity * sizeof(int32_t)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&float_workspace),
                                    GetQsaWorkspaceFloatCount(params) * sizeof(float)));
  ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&int_workspace),
                                    GetQsaWorkspaceIntCount(params) * sizeof(int32_t)));

  std::vector<float> ones(kContextLength * kHeadSize, 1.0f);
  std::vector<float> zeros(kContextLength * kHeadSize, 0.0f);
  std::vector<int64_t> visible(kContextLength, 1);
  ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(query, ones.data(), kNumHeads * kHeadSize * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(key, ones.data(), kHeadSize * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(weight, ones.data(), kHeadSize * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(cosine, ones.data(), ones.size() * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(sine, zeros.data(), zeros.size() * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(mask, visible.data(), kContextLength * sizeof(int64_t),
                                         cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(key_cache, ones.data(), ones.size() * sizeof(float),
                                         cudaMemcpyHostToDevice, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  const auto launch = [&]() {
    return LaunchQsaSparseAttentionIndexer<float>(
        stream, params, query, key, weight, weight, cosine, sine, mask, key_cache, selected, key_cache,
        float_workspace, int_workspace);
  };
  ASSERT_TRUE(launch().IsOK());
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  ASSERT_TRUE(launch().IsOK());
  ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));

  size_t node_count = 0;
  ASSERT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  ASSERT_EQ(cudaSuccess, cudaGraphGetNodes(graph, nodes.data(), &node_count));

  int query_prepare_nodes = 0;
  int block_score_nodes = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType node_type;
    ASSERT_EQ(cudaSuccess, cudaGraphNodeGetType(node, &node_type));
    if (node_type != cudaGraphNodeTypeKernel) {
      continue;
    }

    cudaKernelNodeParams kernel_params{};
    ASSERT_EQ(cudaSuccess, cudaGraphKernelNodeGetParams(node, &kernel_params));
    if (kernel_params.blockDim.x != kThreads) {
      continue;
    }

    const size_t query_prepare_shared_bytes = (kHeadSize + kThreads) * sizeof(float);
    const size_t block_score_shared_bytes = (2 * kHeadSize + kThreads) * sizeof(float);
    if (kernel_params.gridDim.x == kNumHeads && kernel_params.sharedMemBytes == query_prepare_shared_bytes) {
      ++query_prepare_nodes;
    }
    if (kernel_params.gridDim.x == params.max_block_count &&
        kernel_params.sharedMemBytes == block_score_shared_bytes) {
      ++block_score_nodes;
    }
  }
  EXPECT_EQ(query_prepare_nodes, 1) << "QSA query RMSNorm and RoPE must run once per query head";
  EXPECT_EQ(block_score_nodes, 1) << "QSA scoring must consume prepared queries without extra shared state";

  ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  for (int i = 0; i < kWarmupIterations; ++i) {
    ASSERT_EQ(cudaSuccess, cudaGraphLaunch(graph_exec, stream));
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  ASSERT_EQ(cudaSuccess, cudaEventCreate(&start));
  ASSERT_EQ(cudaSuccess, cudaEventCreate(&stop));
  ASSERT_EQ(cudaSuccess, cudaEventRecord(start, stream));
  for (int i = 0; i < kTimedIterations; ++i) {
    ASSERT_EQ(cudaSuccess, cudaGraphLaunch(graph_exec, stream));
  }
  ASSERT_EQ(cudaSuccess, cudaEventRecord(stop, stream));
  ASSERT_EQ(cudaSuccess, cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  ASSERT_EQ(cudaSuccess, cudaEventElapsedTime(&elapsed_ms, start, stop));
  RecordProperty("qsa_8k_decode_us", std::to_string(elapsed_ms * 1000.0f / kTimedIterations));

  std::vector<int32_t> actual(params.capacity);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), selected, actual.size() * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost));
  for (int i = 0; i < kTokenBudget; ++i) {
    EXPECT_EQ(actual[i], i);
  }
  for (int i = kTokenBudget; i < params.capacity; ++i) {
    EXPECT_EQ(actual[i], -1);
  }

  ASSERT_EQ(cudaSuccess, cudaEventDestroy(stop));
  ASSERT_EQ(cudaSuccess, cudaEventDestroy(start));
  ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(graph_exec));
  ASSERT_EQ(cudaSuccess, cudaGraphDestroy(graph));
  ASSERT_EQ(cudaSuccess, cudaFree(int_workspace));
  ASSERT_EQ(cudaSuccess, cudaFree(float_workspace));
  ASSERT_EQ(cudaSuccess, cudaFree(selected));
  ASSERT_EQ(cudaSuccess, cudaFree(key_cache));
  ASSERT_EQ(cudaSuccess, cudaFree(mask));
  ASSERT_EQ(cudaSuccess, cudaFree(sine));
  ASSERT_EQ(cudaSuccess, cudaFree(cosine));
  ASSERT_EQ(cudaSuccess, cudaFree(weight));
  ASSERT_EQ(cudaSuccess, cudaFree(key));
  ASSERT_EQ(cudaSuccess, cudaFree(query));
  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

}  // namespace
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
