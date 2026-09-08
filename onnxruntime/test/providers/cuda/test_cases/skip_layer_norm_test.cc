// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include "gtest/gtest.h"

#include "contrib_ops/cuda/bert/skip_layer_norm_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace test {

namespace {

struct CudaDeleter {
  void operator()(void* ptr) const {
    cudaFree(ptr);
  }
};

template <typename T>
std::unique_ptr<T, CudaDeleter> AllocateCudaBuffer(size_t count) {
  T* ptr = nullptr;
  EXPECT_EQ(cudaSuccess, cudaMalloc(&ptr, count * sizeof(T)));
  return std::unique_ptr<T, CudaDeleter>(ptr);
}

}  // namespace

TEST(SkipLayerNormCudaKernelTest, StatisticsGraphCaptureReplay) {
  constexpr int hidden_size = 8;
  constexpr int row_count = 1;
  constexpr float epsilon = 1e-5f;
  const std::vector<float> input{
      1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
  const std::vector<float> skip(hidden_size, 0.5f);
  const std::vector<float> gamma(hidden_size, 1.0f);
  const std::vector<float> beta(hidden_size, 0.0f);

  auto output = AllocateCudaBuffer<float>(hidden_size);
  auto sum = AllocateCudaBuffer<float>(hidden_size);
  auto mean = AllocateCudaBuffer<float>(1);
  auto inv_std = AllocateCudaBuffer<float>(1);
  auto input_device = AllocateCudaBuffer<float>(hidden_size);
  auto skip_device = AllocateCudaBuffer<float>(hidden_size);
  auto gamma_device = AllocateCudaBuffer<float>(hidden_size);
  auto beta_device = AllocateCudaBuffer<float>(hidden_size);
  ASSERT_NE(output, nullptr);
  ASSERT_NE(sum, nullptr);
  ASSERT_NE(mean, nullptr);
  ASSERT_NE(inv_std, nullptr);
  ASSERT_NE(input_device, nullptr);
  ASSERT_NE(skip_device, nullptr);
  ASSERT_NE(gamma_device, nullptr);
  ASSERT_NE(beta_device, nullptr);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(input_device.get(), input.data(), input.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(skip_device.get(), skip.data(), skip.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(gamma_device.get(), gamma.data(), gamma.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(beta_device.get(), beta.data(), beta.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));

  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  LaunchSkipLayerNormKernel<float, false>(
      stream, output.get(), sum.get(), mean.get(), inv_std.get(), input_device.get(), skip_device.get(),
      nullptr, gamma_device.get(), beta_device.get(), epsilon, hidden_size, row_count, hidden_size);
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  LaunchSkipLayerNormKernel<float, false>(
      stream, output.get(), sum.get(), mean.get(), inv_std.get(), input_device.get(), skip_device.get(),
      nullptr, gamma_device.get(), beta_device.get(), epsilon, hidden_size, row_count, hidden_size);
  ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));
  ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

  ASSERT_EQ(cudaSuccess, cudaMemsetAsync(output.get(), 0, hidden_size * sizeof(float), stream));
  ASSERT_EQ(cudaSuccess, cudaMemsetAsync(mean.get(), 0, sizeof(float), stream));
  ASSERT_EQ(cudaSuccess, cudaMemsetAsync(inv_std.get(), 0, sizeof(float), stream));
  ASSERT_EQ(cudaSuccess, cudaGraphLaunch(graph_exec, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  float actual_mean = 0.0f;
  float actual_inv_std = 0.0f;
  ASSERT_EQ(cudaSuccess, cudaMemcpy(&actual_mean, mean.get(), sizeof(float), cudaMemcpyDeviceToHost));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(&actual_inv_std, inv_std.get(), sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> added(hidden_size);
  for (int i = 0; i < hidden_size; ++i) {
    added[static_cast<size_t>(i)] = input[static_cast<size_t>(i)] + skip[static_cast<size_t>(i)];
  }
  const float expected_mean =
      std::accumulate(added.begin(), added.end(), 0.0f) / hidden_size;
  float variance = 0.0f;
  for (float value : added) {
    const float deviation = value - expected_mean;
    variance += deviation * deviation;
  }
  const float expected_inv_std = 1.0f / std::sqrt(variance / hidden_size + epsilon);
  EXPECT_NEAR(actual_mean, expected_mean, 2e-5f);
  EXPECT_NEAR(actual_inv_std, expected_inv_std, 2e-5f);

  ASSERT_EQ(cudaSuccess, cudaGraphExecDestroy(graph_exec));
  ASSERT_EQ(cudaSuccess, cudaGraphDestroy(graph));
  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

}  // namespace test
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
