// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/packed_moe_impl.h"
#include "core/common/inlined_containers.h"
#include "core/common/float16.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
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

TEST(PackedMoEKernelTest, MultiWarpGroupingAndOffsetBindings) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count == 0) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int rows = 8, topk = 16, pairs = rows * topk, experts = 64, intermediate = 256;
  InlinedVector<uint16_t> x(rows * 1024), scale1(experts * intermediate * 4), scale2(experts * 1024);
  // Grid zero with zero signs/scale nibble decodes to the raw FP16 super-scale.
  InlinedVector<uint8_t> fc1(experts * intermediate * 256, 0), fc2(experts * 1024 * 64, 0);
  InlinedVector<int32_t> indices(pairs), metadata(1 + 9 * pairs, -1), unique;
  InlinedVector<float> weights(pairs), expected_part(pairs * 1024);
  InlinedVector<uint16_t> expected_hidden(pairs * intermediate), expected_y(rows * 1024);
  for (int e = 0; e < experts; ++e) {
    std::fill_n(scale1.begin() + e * intermediate * 4, intermediate * 4, MLFloat16((e + 1) / 256.f).val);
    std::fill_n(scale2.begin() + e * 1024, 1024, MLFloat16((e % 7 + 1) / 16.f).val);
  }
  for (int r = 0; r < rows; ++r) {
    const float activation = BFloat16((r + 1) / 1024.f).ToFloat();
    std::fill_n(x.begin() + r * 1024, 1024, BFloat16(activation).val);
    for (int k = 0; k < topk; ++k) {
      const int p = r * topk + k;
      // New experts first appear at pairs 0, 32, 64 and 96, exercising all four warp prefixes.
      const int e = indices[p] = (r / 2) * topk + k;
      weights[p] = (k + 1) / 32.f;
      auto it = std::find(unique.begin(), unique.end(), e);
      if (it == unique.end()) {
        unique.push_back(e);
        it = unique.end() - 1;
      }
      metadata[1 + pairs + static_cast<int>(it - unique.begin()) * 8 + r] = p;
      float dot = 0.f;
      const float w1 = BFloat16(MLFloat16::FromBits(scale1[e * intermediate * 4]).ToFloat()).ToFloat();
      for (int i = 0; i < 1024; ++i) dot = std::fma(activation, w1, dot);
      const float relu = std::max(BFloat16(dot).ToFloat(), 0.f);
      const BFloat16 hidden(relu * relu);
      std::fill_n(expected_hidden.begin() + p * intermediate, intermediate, hidden.val);
      dot = 0.f;
      const float w2 = BFloat16(MLFloat16::FromBits(scale2[e * 1024]).ToFloat()).ToFloat();
      for (int i = 0; i < intermediate; ++i) dot = std::fma(hidden.ToFloat(), w2, dot);
      std::fill_n(expected_part.begin() + p * 1024, 1024, BFloat16(dot).ToFloat());
    }
    float sum = 0.f;
    for (int k = 0; k < topk; ++k)
      sum = std::fma(weights[r * topk + k], expected_part[(r * topk + k) * 1024], sum);
    std::fill_n(expected_y.begin() + r * 1024, 1024, BFloat16(sum).val);
  }
  metadata[0] = static_cast<int32_t>(unique.size());
  ASSERT_EQ(unique.size(), static_cast<size_t>(experts));
  std::copy(unique.begin(), unique.end(), metadata.begin() + 1);
  for (int offset_mask : {0, 1, 2, 4, 8, 16, 32, 63}) {
    SCOPED_TRACE(testing::Message() << "offset_mask=" << offset_mask);
    const size_t ox = offset_mask & 1 ? sizeof(uint16_t) : 0;
    const size_t ofc1 = offset_mask & 2 ? 1 : 0, ofc2 = offset_mask & 4 ? 1 : 0;
    const size_t oh = offset_mask & 8 ? sizeof(uint16_t) : 0;
    const size_t op = offset_mask & 16 ? sizeof(float) : 0;
    const size_t oy = offset_mask & 32 ? sizeof(uint16_t) : 0;
    auto dx = Allocate(x.size() * sizeof(uint16_t) + ox);
    auto d1 = Allocate(fc1.size() + ofc1), d2 = Allocate(fc2.size() + ofc2);
    auto ds1 = Allocate(scale1.size() * sizeof(uint16_t)), ds2 = Allocate(scale2.size() * sizeof(uint16_t));
    auto di = Allocate(indices.size() * sizeof(int32_t)), dw = Allocate(weights.size() * sizeof(float));
    auto dh = Allocate(expected_hidden.size() * sizeof(uint16_t) + oh);
    auto dp = Allocate(expected_part.size() * sizeof(float) + op);
    auto dy = Allocate(expected_y.size() * sizeof(uint16_t) + oy);
    auto dm = Allocate(metadata.size() * sizeof(int32_t));
    for (const auto* ptr : {dx.get(), d1.get(), d2.get(), ds1.get(), ds2.get(), di.get(), dw.get(),
                            dh.get(), dp.get(), dy.get(), dm.get()})
      ASSERT_NE(ptr, nullptr);
    auto* px = static_cast<uint8_t*>(dx.get()) + ox;
    auto* p1 = static_cast<uint8_t*>(d1.get()) + ofc1;
    auto* p2 = static_cast<uint8_t*>(d2.get()) + ofc2;
    auto* ph = static_cast<uint8_t*>(dh.get()) + oh;
    auto* pp = reinterpret_cast<float*>(static_cast<uint8_t*>(dp.get()) + op);
    auto* py = static_cast<uint8_t*>(dy.get()) + oy;
    ASSERT_EQ(cudaMemcpy(px, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(p1, fc1.data(), fc1.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(p2, fc2.data(), fc2.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ds1.get(), scale1.data(), scale1.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ds2.get(), scale2.data(), scale2.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(di.get(), indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dw.get(), weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(contrib::cuda::LaunchMoEPackedIQ2XXS(
                  nullptr, rows, topk, experts, intermediate, px, static_cast<const int32_t*>(di.get()),
                  static_cast<const float*>(dw.get()), p1, ds1.get(), p2, ds2.get(), py, ph, pp,
                  static_cast<int32_t*>(dm.get())),
              cudaSuccess);
    InlinedVector<uint16_t> actual_hidden(expected_hidden.size()), actual_y(expected_y.size());
    InlinedVector<float> actual_part(expected_part.size());
    InlinedVector<int32_t> actual_metadata(metadata.size());
    ASSERT_EQ(cudaMemcpy(actual_hidden.data(), ph, actual_hidden.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_part.data(), pp, actual_part.size() * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_y.data(), py, actual_y.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_metadata.data(), dm.get(), actual_metadata.size() * sizeof(int32_t),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(actual_metadata, metadata);
    EXPECT_EQ(actual_hidden, expected_hidden);
    EXPECT_EQ(actual_part, expected_part);
    EXPECT_EQ(actual_y, expected_y);
  }
}

TEST(PackedMoEKernelTest, CooperativeGroupingExcludesInvalidRows) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count == 0) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int rows = 8, topk = 9, pairs = rows * topk, experts = 16, intermediate = 256;
  InlinedVector<uint16_t> x(rows * 1024, 0), scale1(experts * intermediate * 4, 0), scale2(experts * 1024, 0);
  InlinedVector<uint8_t> fc1(experts * intermediate * 256, 0), fc2(experts * 1024 * 64, 0);
  InlinedVector<float> weights(pairs, 1.f);
  auto dx = Allocate(x.size() * sizeof(uint16_t));
  auto d1 = Allocate(fc1.size()), d2 = Allocate(fc2.size());
  auto ds1 = Allocate(scale1.size() * sizeof(uint16_t)), ds2 = Allocate(scale2.size() * sizeof(uint16_t));
  auto dw = Allocate(weights.size() * sizeof(float));
  auto di = Allocate((pairs + 1) * sizeof(int32_t));
  auto dm = Allocate((1 + 9 * pairs + 2) * sizeof(int32_t));
  auto dh = Allocate(pairs * intermediate * sizeof(uint16_t)), dp = Allocate(pairs * 1024 * sizeof(float));
  auto dy = Allocate(rows * 1024 * sizeof(uint16_t));
  for (const auto* ptr : {dx.get(), d1.get(), d2.get(), ds1.get(), ds2.get(), dw.get(),
                          di.get(), dm.get(), dh.get(), dp.get(), dy.get()})
    ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(cudaMemcpy(dx.get(), x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d1.get(), fc1.data(), fc1.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d2.get(), fc2.data(), fc2.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(ds1.get(), scale1.data(), scale1.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(ds2.get(), scale2.data(), scale2.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dw.get(), weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
  for (int kind = 0; kind < 6; ++kind) {
    SCOPED_TRACE(testing::Message() << "kind=" << kind);
    InlinedVector<int32_t> indices(pairs), metadata(1 + 9 * pairs, -1), unique;
    bool invalid[rows]{};
    for (int r = 0; r < rows; ++r)
      for (int k = 0; k < topk; ++k) indices[r * topk + k] = (r * 3 + k) % experts;
    if (kind == 1) {
      indices[1] = indices[0];
      invalid[0] = true;
    } else if (kind == 2) {
      // Row three spans a warp boundary; both sides of the duplicate must invalidate it.
      indices[32] = indices[31];
      invalid[3] = true;
    } else if (kind == 3 || kind == 4) {
      indices[4 * topk] = kind == 3 ? -1 : experts;
      invalid[4] = true;
    } else if (kind == 5) {
      for (int r = 0; r < rows; ++r) {
        indices[r * topk] = -1;
        invalid[r] = true;
      }
    }
    for (int r = 0; r < rows; ++r) {
      if (invalid[r]) continue;
      for (int k = 0; k < topk; ++k) {
        const int p = r * topk + k, e = indices[p];
        auto it = std::find(unique.begin(), unique.end(), e);
        if (it == unique.end()) {
          unique.push_back(e);
          it = unique.end() - 1;
        }
        metadata[1 + pairs + static_cast<int>(it - unique.begin()) * 8 + r] = p;
      }
    }
    metadata[0] = static_cast<int32_t>(unique.size());
    std::copy(unique.begin(), unique.end(), metadata.begin() + 1);
    for (int offset : {0, 1}) {
      SCOPED_TRACE(testing::Message() << "offset=" << offset);
      auto* pi = static_cast<int32_t*>(di.get()) + offset;
      auto* pm = static_cast<int32_t*>(dm.get()) + 1;
      ASSERT_EQ(cudaMemcpy(pi, indices.data(), pairs * sizeof(int32_t), cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(cudaMemset(dm.get(), 0x55, (metadata.size() + 2) * sizeof(int32_t)), cudaSuccess);
      ASSERT_EQ(contrib::cuda::LaunchMoEPackedIQ2XXS(
                    nullptr, rows, topk, experts, intermediate, dx.get(), pi, static_cast<const float*>(dw.get()),
                    static_cast<const uint8_t*>(d1.get()), ds1.get(), static_cast<const uint8_t*>(d2.get()),
                    ds2.get(), dy.get(), dh.get(), static_cast<float*>(dp.get()), pm),
                cudaSuccess);
      InlinedVector<int32_t> actual_metadata(metadata.size() + 2);
      InlinedVector<uint16_t> hidden(pairs * intermediate), y(rows * 1024);
      InlinedVector<float> part(pairs * 1024);
      ASSERT_EQ(cudaMemcpy(actual_metadata.data(), dm.get(), actual_metadata.size() * sizeof(int32_t),
                           cudaMemcpyDeviceToHost),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(hidden.data(), dh.get(), hidden.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(part.data(), dp.get(), part.size() * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(y.data(), dy.get(), y.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(actual_metadata.front(), 0x55555555);
      EXPECT_EQ(actual_metadata.back(), 0x55555555);
      EXPECT_TRUE(std::equal(metadata.begin(), metadata.end(), actual_metadata.begin() + 1));
      for (int r = 0; r < rows; ++r) {
        SCOPED_TRACE(testing::Message() << "row=" << r);
        for (int k = 0; k < topk; ++k) {
          const int p = r * topk + k;
          for (int j = 0; j < intermediate; ++j) {
            const float value = BFloat16::FromBits(hidden[p * intermediate + j]).ToFloat();
            ASSERT_TRUE(invalid[r] ? std::isnan(value) : value == 0.f);
          }
          for (int j = 0; j < 1024; ++j) {
            const float value = part[p * 1024 + j];
            ASSERT_TRUE(invalid[r] ? std::isnan(value) : value == 0.f);
          }
        }
        for (int j = 0; j < 1024; ++j) {
          const float value = BFloat16::FromBits(y[r * 1024 + j]).ToFloat();
          ASSERT_TRUE(invalid[r] ? std::isnan(value) : value == 0.f);
        }
      }
    }
  }
}
}  // namespace
}  // namespace onnxruntime::test
#endif
