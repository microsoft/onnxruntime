// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <memory>
#include <string>
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

void RunSmallNGemvCase(int m, int n, int k) {
  SCOPED_TRACE("m=" + std::to_string(m) + ", n=" + std::to_string(n) + ", k=" + std::to_string(k));
  std::vector<MLFloat16> a(static_cast<size_t>(m) * k);
  std::vector<MLFloat16> b(static_cast<size_t>(k) * n);
  for (size_t index = 0; index < a.size(); ++index) {
    a[index] = MLFloat16(static_cast<float>((index * 17 + 3) % 29) / 16.0f - 0.875f);
  }
  for (size_t index = 0; index < b.size(); ++index) {
    b[index] = MLFloat16(static_cast<float>((index * 13 + 5) % 31) / 16.0f - 0.9375f);
  }

  auto device_a = AllocateDeviceMemory<MLFloat16>(a.size());
  auto device_b = AllocateDeviceMemory<MLFloat16>(b.size());
  auto device_c = AllocateDeviceMemory<MLFloat16>(static_cast<size_t>(m) * n);
  auto workspace = AllocateDeviceMemory<float>(SmallNGemvWorkspaceElements(m, n, k));
  auto counter = AllocateDeviceMemory<unsigned int>(SmallNGemvCounterElements(n));
  CUDA_CALL_THROW(cudaMemcpy(device_a.get(), a.data(), a.size() * sizeof(MLFloat16), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_b.get(), b.data(), b.size() * sizeof(MLFloat16), cudaMemcpyHostToDevice));

  for (int iteration = 0; iteration < 2; ++iteration) {
    CUDA_CALL_THROW(cudaMemset(counter.get(), 0, SmallNGemvCounterElements(n) * sizeof(unsigned int)));
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
        float expected = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          expected += a[static_cast<size_t>(row) * k + kk].ToFloat() *
                      b[static_cast<size_t>(kk) * n + col].ToFloat();
        }
        EXPECT_NEAR(output[static_cast<size_t>(row) * n + col].ToFloat(), expected, 0.05f);
      }
    }
  }
}

TEST(MatMulSmallNGemvTest, HandlesAllMVariants) {
  for (int m = 1; m <= 8; ++m) {
    RunSmallNGemvCase(m, 37, 133);
  }
}

TEST(MatMulSmallNGemvTest, HandlesColumnTileBoundaries) {
  for (const int n : {1, 32, 1024}) {
    RunSmallNGemvCase(1, n, 128);
  }
}

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime