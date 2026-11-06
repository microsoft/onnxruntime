// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/matmul_packed_projection.cuh"
#include "core/common/inlined_containers.h"
#include "core/common/float16.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <memory>

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
namespace onnxruntime::test {
namespace {
using DeviceBuffer = std::unique_ptr<void, decltype(&cudaFree)>;

DeviceBuffer Allocate(size_t bytes) {
  void* ptr = nullptr;
  EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
  return DeviceBuffer(ptr, cudaFree);
}

TEST(PackedProjectionKernelTest, Nvfp4MmaAlignedAndOffsetActivations) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count == 0) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr float global_scale = 0.75f;
  constexpr uint8_t local_bits[] = {0x30, 0x38, 0x3c, 0x40};
  constexpr float local_values[] = {0.5f, 1.f, 1.5f, 2.f};
  for (int rows : {3, 8}) {
    for (int k : {4096, 8192}) {
      for (int n : {19, 8193, 18560}) {
        SCOPED_TRACE(testing::Message() << "rows=" << rows << " K=" << k << " N=" << n);
        InlinedVector<uint16_t> a(rows * k), actual(rows * n + 2), expected(rows * n);
        InlinedVector<uint8_t> b(n * k / 2), scales(n * k / 16);
        for (int r = 0; r < rows; ++r)
          std::fill_n(a.begin() + r * k, k, BFloat16((r + 1) / 4096.f).val);
        for (int j = 0; j < n; ++j) {
          // E2M1 nibbles encode +1 or -1; local E4M3 scales are exact BF16 values.
          const float sign = j % 2 == 0 ? 1.f : -1.f;
          std::fill_n(b.begin() + j * k / 2, k / 2, j % 2 == 0 ? 0x22 : 0xaa);
          std::fill_n(scales.begin() + j * k / 16, k / 16, local_bits[j % 4]);
          for (int r = 0; r < rows; ++r)
            expected[r * n + j] =
                BFloat16((r + 1) / 4096.f * k * sign * local_values[j % 4] * global_scale).val;
        }
        auto da = Allocate((a.size() + 1) * sizeof(uint16_t)), db = Allocate(b.size());
        auto ds = Allocate(scales.size()), dg = Allocate(sizeof(float));
        auto dy = Allocate(actual.size() * sizeof(uint16_t));
        for (const auto* ptr : {da.get(), db.get(), ds.get(), dg.get(), dy.get()}) ASSERT_NE(ptr, nullptr);
        ASSERT_EQ(cudaMemcpy(db.get(), b.data(), b.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(ds.get(), scales.data(), scales.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dg.get(), &global_scale, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        for (int offset : {0, 1}) {
          SCOPED_TRACE(testing::Message() << "activation_offset=" << offset);
          auto* input = static_cast<uint16_t*>(da.get()) + offset;
          auto* output = static_cast<uint16_t*>(dy.get()) + 1;
          ASSERT_EQ(cudaMemcpy(input, a.data(), a.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
          ASSERT_EQ(cudaMemset(dy.get(), 0x55, actual.size() * sizeof(uint16_t)), cudaSuccess);
          ASSERT_EQ(contrib::cuda::LaunchPackedProjection(
                        contrib::cuda::PackedProjectionFormat::NVFP4, input,
                        static_cast<const uint8_t*>(db.get()), ds.get(), static_cast<const float*>(dg.get()),
                        false, rows, n, k, output, nullptr),
                    cudaSuccess);
          ASSERT_EQ(cudaMemcpy(actual.data(), dy.get(), actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_EQ(actual.front(), 0x5555);
          EXPECT_EQ(actual.back(), 0x5555);
          for (int i = 0; i < rows * n; ++i) {
            ASSERT_EQ(actual[i + 1], expected[i]) << i;
          }
        }
      }
    }
  }
}

TEST(PackedProjectionKernelTest, Q8MmaReductionOrderCancellation) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count == 0) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int rows = 3, k = 1088;
  InlinedVector<uint16_t> a(rows * k, BFloat16(0.f).val);
  a[0] = BFloat16(0x1p24f).val;
  a[64] = BFloat16(1.f).val;
  a[256] = BFloat16(-0x1p24f).val;
  a[k + 64] = BFloat16(2.f).val;
  a[2 * k + 256] = BFloat16(-2.f).val;
  for (int n : {19, 8192}) {
    for (bool fp16_scale : {false, true}) {
      SCOPED_TRACE(testing::Message() << "N=" << n << " fp16_scale=" << fp16_scale);
      InlinedVector<uint8_t> b(n * k, 129);
      const uint16_t one = fp16_scale ? MLFloat16(1.f).val : BFloat16(1.f).val;
      InlinedVector<uint16_t> scales(n * k / 32, one), actual(rows * n);
      auto da = Allocate(a.size() * sizeof(uint16_t)), db = Allocate(b.size());
      auto ds = Allocate(scales.size() * sizeof(uint16_t)), dy = Allocate(actual.size() * sizeof(uint16_t));
      for (const auto* ptr : {da.get(), db.get(), ds.get(), dy.get()}) ASSERT_NE(ptr, nullptr);
      ASSERT_EQ(cudaMemcpy(da.get(), a.data(), a.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(db.get(), b.data(), b.size(), cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(ds.get(), scales.data(), scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
                cudaSuccess);
      ASSERT_EQ(contrib::cuda::LaunchPackedProjection(
                    contrib::cuda::PackedProjectionFormat::Q8, da.get(), static_cast<const uint8_t*>(db.get()),
                    ds.get(), nullptr, fp16_scale, rows, n, k, dy.get(), nullptr),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(actual.data(), dy.get(), actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                cudaSuccess);
      for (int j = 0; j < n; ++j) {
        EXPECT_EQ(actual[j], BFloat16(n == 19 ? 0.f : 1.f).val) << j;
        EXPECT_EQ(actual[n + j], BFloat16(2.f).val) << j;
        EXPECT_EQ(actual[2 * n + j], BFloat16(-2.f).val) << j;
      }
    }
  }
}
}  // namespace
}  // namespace onnxruntime::test
#endif
