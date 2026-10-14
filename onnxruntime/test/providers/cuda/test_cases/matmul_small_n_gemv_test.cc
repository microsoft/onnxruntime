// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "core/common/float16.h"
#include "core/providers/cuda/math/matmul_small_n_gemv.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace cuda {
namespace test {
namespace {

struct CudaDeviceMemoryDeleter {
  template <typename T>
  void operator()(T* p) const {
    cudaFree(p);
  }
};

template <typename T>
std::unique_ptr<T, CudaDeviceMemoryDeleter> AllocateDeviceMemory(size_t count) {
  T* buffer{};
  CUDA_CALL_THROW(cudaMalloc(&buffer, count * sizeof(T)));
  return std::unique_ptr<T, CudaDeviceMemoryDeleter>(buffer);
}

template <typename HostT>
struct DeviceType;
template <>
struct DeviceType<MLFloat16> {
  using type = half;
};
template <>
struct DeviceType<BFloat16> {
  using type = nv_bfloat16;
};

template <typename HostT>
void RunSmallNGemvCase(int m, int n, int k, bool capture_graph = false) {
  using DeviceT = typename DeviceType<HostT>::type;
  SCOPED_TRACE(std::string(std::is_same_v<HostT, MLFloat16> ? "fp16" : "bf16") + " m=" + std::to_string(m) +
               ", n=" + std::to_string(n) + ", k=" + std::to_string(k));
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
  auto workspace = AllocateDeviceMemory<float>(SmallNGemvWorkspaceElements(m, n, k));
  auto counter = AllocateDeviceMemory<unsigned int>(SmallNGemvCounterElements(n));
  CUDA_CALL_THROW(cudaMemcpy(device_a.get(), a.data(), a.size() * sizeof(HostT), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_b.get(), b.data(), b.size() * sizeof(HostT), cudaMemcpyHostToDevice));
  // The launcher must clear stale counters itself.
  CUDA_CALL_THROW(cudaMemset(counter.get(), 0xff, SmallNGemvCounterElements(n) * sizeof(unsigned int)));

  // Accumulation is fp32, so the error is dominated by rounding the output to T.
  const float relative_tolerance = std::is_same_v<HostT, MLFloat16> ? 0.0f : 1.0f / 128.0f;
  for (int iteration = 0; iteration < 2; ++iteration) {
    cudaStream_t stream = nullptr;
    if (capture_graph) {
      CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      CUDA_CALL_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    }
    ASSERT_STATUS_OK(LaunchSmallNGemv(stream,
                                      reinterpret_cast<const DeviceT*>(device_a.get()),
                                      reinterpret_cast<const DeviceT*>(device_b.get()),
                                      reinterpret_cast<DeviceT*>(device_c.get()),
                                      m, n, k, workspace.get(), counter.get()));
    if (capture_graph) {
      cudaGraph_t graph;
      cudaGraphExec_t graph_exec;
      CUDA_CALL_THROW(cudaStreamEndCapture(stream, &graph));
      CUDA_CALL_THROW(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
      for (int replay = 0; replay < 3; ++replay) {
        CUDA_CALL_THROW(cudaGraphLaunch(graph_exec, stream));
      }
      CUDA_CALL_THROW(cudaStreamSynchronize(stream));
      CUDA_CALL_THROW(cudaGraphExecDestroy(graph_exec));
      CUDA_CALL_THROW(cudaGraphDestroy(graph));
      CUDA_CALL_THROW(cudaStreamDestroy(stream));
    }
    CUDA_CALL_THROW(cudaDeviceSynchronize());

    std::vector<HostT> output(static_cast<size_t>(m) * n);
    CUDA_CALL_THROW(cudaMemcpy(output.data(), device_c.get(), output.size() * sizeof(HostT),
                               cudaMemcpyDeviceToHost));
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        float expected = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          expected += a[static_cast<size_t>(row) * k + kk].ToFloat() *
                      b[static_cast<size_t>(kk) * n + col].ToFloat();
        }
        EXPECT_NEAR(output[static_cast<size_t>(row) * n + col].ToFloat(), expected,
                    0.05f + relative_tolerance * std::fabs(expected));
      }
    }
  }
}

void RunSmallNGemvCaseAllTypes(int m, int n, int k, bool capture_graph = false) {
  RunSmallNGemvCase<MLFloat16>(m, n, k, capture_graph);
  RunSmallNGemvCase<BFloat16>(m, n, k, capture_graph);
}

TEST(MatMulSmallNGemvTest, HandlesAllMVariants) {
  for (int m = 1; m <= 8; ++m) {
    RunSmallNGemvCaseAllTypes(m, 37, 133);
  }
}

// Even N, K % 8 == 0 and aligned buffers select the vectorized kernel, including a partial
// 64-column tile, the fewest K splits (two) and multiple 8-row chunks.
TEST(MatMulSmallNGemvTest, HandlesVectorizedVariants) {
  for (int m = 1; m <= 8; ++m) {
    RunSmallNGemvCaseAllTypes(m, 48, 5120);
  }
  RunSmallNGemvCaseAllTypes(8, 96, 2048);
  RunSmallNGemvCaseAllTypes(3, 1024, 128);
  RunSmallNGemvCaseAllTypes(17, 2, 1032);
}

TEST(MatMulSmallNGemvTest, HandlesColumnTileBoundaries) {
  for (const int n : {1, 32, 1024}) {
    RunSmallNGemvCaseAllTypes(1, n, 128);
  }
}

TEST(MatMulSmallNGemvTest, HandlesHigherSplitCounts) {
  for (int m = 1; m <= 3; ++m) {
    RunSmallNGemvCaseAllTypes(m, 324, 10240);
    RunSmallNGemvCaseAllTypes(m, 33, 1031);
  }
  RunSmallNGemvCaseAllTypes(17, 2, 4104);
  RunSmallNGemvCaseAllTypes(9, 1, 5120);
}

TEST(MatMulSmallNGemvTest, SizesWorkspaceForHigherSplitCounts) {
  EXPECT_EQ(SmallNGemvSplitK(1, 5120), 64);
  EXPECT_EQ(SmallNGemvWorkspaceElements(3, 324, 10240), size_t{64 * 3 * 324});
  EXPECT_EQ(SmallNGemvWorkspaceElements(17, 2, 4104), size_t{64 * 8 * 2});
  EXPECT_EQ(SmallNGemvWorkspaceElements(3, 1024, 128), size_t{4 * 3 * 1024});
}

TEST(MatMulSmallNGemvTest, ReplaysHigherSplitCountsInCudaGraph) {
  for (int m = 1; m <= 3; ++m) {
    RunSmallNGemvCaseAllTypes(m, 324, 10240, true);
    RunSmallNGemvCaseAllTypes(m, 33, 1031, true);
  }
  RunSmallNGemvCaseAllTypes(17, 2, 4104, true);
}

// The smallest eligible K still splits in two, so the vectorized kernel never needs a single-slice path.
TEST(MatMulSmallNGemvTest, SelectsVectorizedKernelForAlignedEvenShapes) {
  const void* aligned = reinterpret_cast<const void*>(uintptr_t{256});
  EXPECT_TRUE(SmallNGemvUsesVectorizedKernel(1024, 128, aligned, aligned));
  EXPECT_TRUE(SmallNGemvUsesVectorizedKernel(2, 5120, aligned, aligned));
  EXPECT_FALSE(SmallNGemvUsesVectorizedKernel(1024, 120, aligned, aligned));
  EXPECT_FALSE(SmallNGemvUsesVectorizedKernel(1023, 128, aligned, aligned));
  EXPECT_FALSE(SmallNGemvUsesVectorizedKernel(1024, 128, reinterpret_cast<const void*>(uintptr_t{8}), aligned));
}

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime