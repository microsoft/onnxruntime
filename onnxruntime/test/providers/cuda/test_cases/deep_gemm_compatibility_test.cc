// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_DEEP_GEMM

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include <cuda_runtime.h>
#include "gtest/gtest.h"

namespace onnxruntime::test {

cudaError_t LaunchDeepGemmScales(void* storage, int count);
cudaError_t LaunchDeepGemmPackedStore(void* storage);

TEST(CUDA_EP_Unittest, DeepGemmScaleRoundingAtPowerOfTwoBoundaries) {
  constexpr int count = 21;
  std::array<float2, 3 * count> storage{};
  std::array<float2, count> expected_scale{};
  std::array<float2, count> expected_inverse{};
  constexpr float factor = static_cast<float>(1.0 / 448.0);
  int index = 0;
  for (int exponent = -90; exponent <= 90; exponent += 30) {
    const float exact = std::ldexp(448.0f, exponent);
    const std::array<float, 3> values{
        std::nextafter(exact, 0.0f), exact,
        std::nextafter(exact, std::numeric_limits<float>::infinity())};
    for (int offset = 0; offset < 3; ++offset, ++index) {
      storage[index] = make_float2(values[offset], values[2 - offset]);
      auto reference = [&](float value) {
        // Round the multiplication to float before finding the next power of two.
        volatile float product = value * factor;
        int power = 0;
        const float fraction = std::frexp(product, &power);
        power -= fraction == 0.5f ? 1 : 0;
        return make_float2(std::ldexp(1.0f, power), std::ldexp(1.0f, -power));
      };
      const auto x = reference(storage[index].x);
      const auto y = reference(storage[index].y);
      expected_scale[index] = make_float2(x.x, y.x);
      expected_inverse[index] = make_float2(x.y, y.y);
    }
  }
  ASSERT_EQ(index, count);
  void* raw = nullptr;
  ASSERT_EQ(cudaMalloc(&raw, sizeof(storage)), cudaSuccess);
  auto free_buffer = [](void* buffer) { EXPECT_EQ(cudaFree(buffer), cudaSuccess); };
  std::unique_ptr<void, decltype(free_buffer)> device(raw, free_buffer);
  ASSERT_EQ(cudaMemcpy(device.get(), storage.data(), sizeof(storage), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(LaunchDeepGemmScales(device.get(), count), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(storage.data(), device.get(), sizeof(storage), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(std::memcmp(&storage[count + i], &expected_scale[i], sizeof(float2)), 0) << i;
    EXPECT_EQ(std::memcmp(&storage[2 * count + i], &expected_inverse[i], sizeof(float2)), 0) << i;
  }
}

TEST(CUDA_EP_Unittest, DeepGemmPackedSharedStorePreservesEveryBit) {
  constexpr int count = 32;
  std::array<uint4, 2 * count> storage{};
  for (uint32_t i = 0; i < count; ++i) {
    storage[i] = make_uint4(0x80000000u + i, 0xffffffffu - i * 0x01010101u, 0x7f800001u | (i << 17), 0x9e3779b9u * i);
  }
  storage[0] = make_uint4(0, 0, 0, 0);
  storage[1] = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  storage[2] = make_uint4(0xaaaaaaaau, 0x55555555u, 0xaaaaaaaau, 0x55555555u);
  void* raw = nullptr;
  ASSERT_EQ(cudaMalloc(&raw, sizeof(storage)), cudaSuccess);
  auto free_buffer = [](void* buffer) { EXPECT_EQ(cudaFree(buffer), cudaSuccess); };
  std::unique_ptr<void, decltype(free_buffer)> device(raw, free_buffer);
  ASSERT_EQ(cudaMemcpy(device.get(), storage.data(), sizeof(storage), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(LaunchDeepGemmPackedStore(device.get()), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(storage.data(), device.get(), sizeof(storage), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(std::memcmp(&storage[count + i], &storage[(i + 1) % count], sizeof(uint4)), 0) << i;
  }
}

}  // namespace onnxruntime::test

#endif  // USE_DEEP_GEMM
