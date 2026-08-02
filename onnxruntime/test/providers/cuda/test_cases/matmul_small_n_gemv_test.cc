// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/framework/float16.h"
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

TEST(MatMulSmallNGemvTest, HandlesUnevenSplitsAndResetsCounters) {
  constexpr int m = 3;
  constexpr int n = 37;
  constexpr int k = 133;

  std::vector<MLFloat16> a(static_cast<size_t>(m) * k);
  std::vector<MLFloat16> b(static_cast<size_t>(k) * n);
  for (int row = 0; row < m; ++row) {
    std::fill_n(a.begin() + static_cast<size_t>(row) * k, k, MLFloat16(static_cast<float>(row + 1)));
  }
  for (int col = 0; col < n; ++col) {
    const MLFloat16 value(col % 2 == 0 ? 1.0f : 0.5f);
    for (int row = 0; row < k; ++row) {
      b[static_cast<size_t>(row) * n + col] = value;
    }
  }

  auto device_a = AllocateDeviceMemory<MLFloat16>(a.size());
  auto device_b = AllocateDeviceMemory<MLFloat16>(b.size());
  auto device_c = AllocateDeviceMemory<MLFloat16>(static_cast<size_t>(m) * n);
  auto workspace = AllocateDeviceMemory<float>(SmallNGemvWorkspaceElements(m, n, k));
  auto counter = AllocateDeviceMemory<unsigned int>(SmallNGemvCounterElements(n));
  CUDA_CALL_THROW(cudaMemcpy(device_a.get(), a.data(), a.size() * sizeof(MLFloat16), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_b.get(), b.data(), b.size() * sizeof(MLFloat16), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemset(counter.get(), 0, SmallNGemvCounterElements(n) * sizeof(unsigned int)));

  for (int iteration = 0; iteration < 2; ++iteration) {
    ASSERT_STATUS_OK(LaunchSmallNGemv(nullptr,
                                      reinterpret_cast<const half*>(device_a.get()),
                                      reinterpret_cast<const half*>(device_b.get()),
                                      reinterpret_cast<half*>(device_c.get()),
                                      m, n, k, workspace.get(), counter.get()));
    CUDA_CALL_THROW(cudaDeviceSynchronize());

    std::vector<MLFloat16> output(static_cast<size_t>(m) * n);
    CUDA_CALL_THROW(cudaMemcpy(output.data(), device_c.get(), output.size() * sizeof(MLFloat16),
                               cudaMemcpyDeviceToHost));
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        const float scale = col % 2 == 0 ? 1.0f : 0.5f;
        EXPECT_EQ(output[static_cast<size_t>(row) * n + col].ToFloat(), k * (row + 1) * scale);
      }
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime