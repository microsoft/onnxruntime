// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/providers/cuda/math/tinygemm2.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace cuda {
namespace test {
namespace {

struct CudaDeviceMemoryDeleter {
  void operator()(void* p) const { cudaFree(p); }
};

struct CudaGraphResources {
  CudaGraphResources() { CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~CudaGraphResources() {
    cudaStreamSynchronize(stream);
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaGraphResources);

  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
};

template <typename T>
std::unique_ptr<T, CudaDeviceMemoryDeleter> AllocateDeviceMemory(size_t count) {
  void* buffer{};
  CUDA_CALL_THROW(cudaMalloc(&buffer, count * sizeof(T)));
  return std::unique_ptr<T, CudaDeviceMemoryDeleter>(static_cast<T*>(buffer));
}

bool TinyGemm2Available() {
  int device = 0;
  CUDA_CALL_THROW(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device));
  return IsTinyGemm2Supported(prop);
}

template <typename HostT>
void RunTinyGemm2Case(int m, int n, int k, bool b_is_constant) {
  using DeviceT = std::conditional_t<std::is_same_v<HostT, MLFloat16>, half, nv_bfloat16>;
  SCOPED_TRACE(std::string(std::is_same_v<HostT, MLFloat16> ? "fp16" : "bf16") + " m=" + std::to_string(m) +
               ", n=" + std::to_string(n) + ", k=" + std::to_string(k) +
               ", constant_b=" + std::to_string(b_is_constant));
  std::vector<HostT> a(static_cast<size_t>(m) * k);
  std::vector<HostT> b(static_cast<size_t>(k) * n);
  for (size_t index = 0; index < a.size(); ++index) {
    a[index] = HostT(static_cast<float>((index * 17 + 3) % 29) / 16.0f - 0.875f);
  }
  for (size_t index = 0; index < b.size(); ++index) {
    b[index] = HostT(static_cast<float>((index * 13 + 5) % 31) / 16.0f - 0.9375f);
  }

  auto device_a = AllocateDeviceMemory<HostT>(a.size());
  auto device_b = AllocateDeviceMemory<HostT>(b.size());
  auto device_c = AllocateDeviceMemory<HostT>(static_cast<size_t>(m) * n);
  CUDA_CALL_THROW(cudaMemcpy(device_a.get(), a.data(), a.size() * sizeof(HostT), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_b.get(), b.data(), b.size() * sizeof(HostT), cudaMemcpyHostToDevice));
  // NaN bytes catch any output the kernel fails to write.
  CUDA_CALL_THROW(cudaMemset(device_c.get(), 0xff, static_cast<size_t>(m) * n * sizeof(HostT)));

  ASSERT_STATUS_OK(LaunchTinyGemm2(nullptr, reinterpret_cast<const DeviceT*>(device_a.get()),
                                   reinterpret_cast<const DeviceT*>(device_b.get()),
                                   reinterpret_cast<DeviceT*>(device_c.get()), m, n, k, b_is_constant));
  CUDA_CALL_THROW(cudaDeviceSynchronize());

  std::vector<HostT> output(static_cast<size_t>(m) * n);
  CUDA_CALL_THROW(cudaMemcpy(output.data(), device_c.get(), output.size() * sizeof(HostT), cudaMemcpyDeviceToHost));
  // Accumulation is fp32, so the error is dominated by rounding the output to T.
  const float relative_tolerance = std::is_same_v<HostT, MLFloat16> ? 1.0f / 1024.0f : 1.0f / 128.0f;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float expected = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        expected += a[static_cast<size_t>(row) * k + kk].ToFloat() * b[static_cast<size_t>(kk) * n + col].ToFloat();
      }
      ASSERT_NEAR(output[static_cast<size_t>(row) * n + col].ToFloat(), expected,
                  0.01f + relative_tolerance * std::fabs(expected))
          << "row=" << row << " col=" << col;
    }
  }
}

void RunTinyGemm2CaseAllTypes(int m, int n, int k) {
  for (const bool b_is_constant : {false, true}) {
    RunTinyGemm2Case<MLFloat16>(m, n, k, b_is_constant);
    RunTinyGemm2Case<BFloat16>(m, n, k, b_is_constant);
  }
}

template <typename HostT>
void RunProducedBChain(bool capture) {
  using DeviceT = std::conditional_t<std::is_same_v<HostT, MLFloat16>, half, nv_bfloat16>;
  SCOPED_TRACE((std::is_same_v<HostT, MLFloat16> ? "fp16" : "bf16"));
  constexpr int m = 8;
  constexpr int n = 1032;
  constexpr int k = 64;
  constexpr int producer_k = 5120;
  const std::vector<HostT> producer_a(k * producer_k, HostT(1.0f));
  const std::vector<HostT> producer_b(producer_k * n, HostT(1.0f));
  const std::vector<HostT> a(m * k, HostT(1.0f / 1024.0f));
  auto device_producer_a = AllocateDeviceMemory<HostT>(producer_a.size());
  auto device_producer_b = AllocateDeviceMemory<HostT>(producer_b.size());
  auto device_a = AllocateDeviceMemory<HostT>(a.size());
  auto device_b = AllocateDeviceMemory<HostT>(k * n);
  auto device_c = AllocateDeviceMemory<HostT>(m * n);
  CUDA_CALL_THROW(cudaMemcpy(device_producer_a.get(), producer_a.data(), producer_a.size() * sizeof(HostT),
                             cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_producer_b.get(), producer_b.data(), producer_b.size() * sizeof(HostT),
                             cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_a.get(), a.data(), a.size() * sizeof(HostT), cudaMemcpyHostToDevice));

  CudaGraphResources resources;
  auto launch_chain = [&]() -> Status {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(device_b.get(), 0, k * n * sizeof(HostT), resources.stream));
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(device_c.get(), 0xff, m * n * sizeof(HostT), resources.stream));
    ORT_RETURN_IF_ERROR(LaunchTinyGemm2(resources.stream,
                                        reinterpret_cast<const DeviceT*>(device_producer_a.get()),
                                        reinterpret_cast<const DeviceT*>(device_producer_b.get()),
                                        reinterpret_cast<DeviceT*>(device_b.get()), k, n, producer_k, true));
    return LaunchTinyGemm2(resources.stream, reinterpret_cast<const DeviceT*>(device_a.get()),
                           reinterpret_cast<const DeviceT*>(device_b.get()),
                           reinterpret_cast<DeviceT*>(device_c.get()), m, n, k);
  };
  if (capture) {
    CUDA_CALL_THROW(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal));
    const Status status = launch_chain();
    CUDA_CALL_THROW(cudaStreamEndCapture(resources.stream, &resources.graph));
    ASSERT_STATUS_OK(status);
    CUDA_CALL_THROW(cudaGraphInstantiate(&resources.graph_exec, resources.graph, nullptr, nullptr, 0));
  }

  std::vector<HostT> output(m * n);
  for (int replay = 0; replay < 3; ++replay) {
    SCOPED_TRACE(replay);
    if (capture) {
      CUDA_CALL_THROW(cudaGraphLaunch(resources.graph_exec, resources.stream));
    } else {
      ASSERT_STATUS_OK(launch_chain());
    }
    CUDA_CALL_THROW(cudaStreamSynchronize(resources.stream));
    CUDA_CALL_THROW(cudaMemcpy(output.data(), device_c.get(), output.size() * sizeof(HostT), cudaMemcpyDeviceToHost));
    for (const HostT value : output) {
      ASSERT_EQ(value.ToFloat(), static_cast<float>(producer_k * k) / 1024.0f);
    }
  }
}

TEST(TinyGemm2Test, Eligibility) {
  const void* aligned = reinterpret_cast<const void*>(uintptr_t{256});
  EXPECT_TRUE(CanUseTinyGemm2(1, 8, 8, aligned, aligned));
  EXPECT_TRUE(CanUseTinyGemm2(64, 4096, 8192, aligned, aligned));
  EXPECT_FALSE(CanUseTinyGemm2(65, 48, 5120, aligned, aligned));
  EXPECT_FALSE(CanUseTinyGemm2(8, 4096, 8200, aligned, aligned));  // N * K above 32M
  EXPECT_FALSE(CanUseTinyGemm2(8, std::numeric_limits<int64_t>::max() - 7, 8, aligned, aligned));
  EXPECT_FALSE(CanUseTinyGemm2(8, 8, std::numeric_limits<int64_t>::max() - 7, aligned, aligned));
  EXPECT_FALSE(CanUseTinyGemm2(8, 44, 5120, aligned, aligned));
  EXPECT_FALSE(CanUseTinyGemm2(8, 48, 5124, aligned, aligned));
  EXPECT_FALSE(CanUseTinyGemm2(8, 48, 5120, reinterpret_cast<const void*>(uintptr_t{264}), aligned));
  EXPECT_FALSE(CanUseTinyGemm2(8, 48, 5120, aligned, reinterpret_cast<const void*>(uintptr_t{264})));
}

TEST(TinyGemm2Test, MatchesReference) {
  if (!TinyGemm2Available()) {
    GTEST_SKIP() << "tinygemm2 needs an SM 9.0+ device with SM 9.0+ code in this build.";
  }
  // Partial column tiles (N % 16 != 0), partial row tiles (M % 8 != 0), K below one tile, K not a
  // multiple of the 1024-element K loop, and multiple CTAs along both grid axes.
  for (const int m : {1, 3, 8, 9, 64}) {
    for (const int n : {8, 24, 136}) {
      for (const int k : {8, 136, 1032}) {
        RunTinyGemm2CaseAllTypes(m, n, k);
      }
    }
  }
  RunTinyGemm2CaseAllTypes(4, 2880, 720);
  RunTinyGemm2CaseAllTypes(17, 48, 5120);
}

TEST(TinyGemm2Test, ProducedBChain) {
  if (!TinyGemm2Available()) {
    GTEST_SKIP() << "tinygemm2 needs an SM 9.0+ device with SM 9.0+ code in this build.";
  }
  RunProducedBChain<MLFloat16>(false);
  RunProducedBChain<BFloat16>(false);
}

TEST(TinyGemm2Test, ProducedBChainCudaGraph) {
  if (!TinyGemm2Available()) {
    GTEST_SKIP() << "tinygemm2 needs an SM 9.0+ device with SM 9.0+ code in this build.";
  }
  RunProducedBChain<MLFloat16>(true);
  RunProducedBChain<BFloat16>(true);
}

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
