// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

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
void RunTinyGemm2Case(int m, int n, int k) {
  using DeviceT = std::conditional_t<std::is_same_v<HostT, MLFloat16>, half, nv_bfloat16>;
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
  CUDA_CALL_THROW(cudaMemcpy(device_a.get(), a.data(), a.size() * sizeof(HostT), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_b.get(), b.data(), b.size() * sizeof(HostT), cudaMemcpyHostToDevice));
  // NaN bytes catch any output the kernel fails to write.
  CUDA_CALL_THROW(cudaMemset(device_c.get(), 0xff, static_cast<size_t>(m) * n * sizeof(HostT)));

  ASSERT_STATUS_OK(LaunchTinyGemm2(nullptr, reinterpret_cast<const DeviceT*>(device_a.get()),
                                   reinterpret_cast<const DeviceT*>(device_b.get()),
                                   reinterpret_cast<DeviceT*>(device_c.get()), m, n, k));
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
  RunTinyGemm2Case<MLFloat16>(m, n, k);
  RunTinyGemm2Case<BFloat16>(m, n, k);
}

TEST(TinyGemm2Test, Eligibility) {
  const void* aligned = reinterpret_cast<const void*>(uintptr_t{256});
  EXPECT_TRUE(CanUseTinyGemm2(1, 8, 8, aligned, aligned));
  EXPECT_TRUE(CanUseTinyGemm2(64, 4096, 8192, aligned, aligned));
  EXPECT_FALSE(CanUseTinyGemm2(65, 48, 5120, aligned, aligned));
  EXPECT_FALSE(CanUseTinyGemm2(8, 4096, 8200, aligned, aligned));  // N * K above 32M
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

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
