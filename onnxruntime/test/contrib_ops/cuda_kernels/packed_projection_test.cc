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
