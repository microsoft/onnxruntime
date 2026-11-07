// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>

#include "contrib_ops/cuda/quantization/moe_packed_impl.cuh"
#include "core/common/float16.h"
#include "core/common/inlined_containers.h"
#include "gtest/gtest.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
namespace onnxruntime::test {
namespace {
using DeviceBuffer = std::unique_ptr<void, decltype(&cudaFree)>;

DeviceBuffer Allocate(size_t bytes) {
  void* ptr = nullptr;
  EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
  return DeviceBuffer(ptr, cudaFree);
}

void CheckGroupingAndOffsets(int intermediate) {
  constexpr int rows = 8, topk = 16, pairs = rows * topk, experts = 64;
  InlinedVector<uint16_t> x(rows * 1024), hidden(pairs * intermediate), y(rows * 1024);
  InlinedVector<uint8_t> fc1(static_cast<size_t>(experts) * intermediate * 264);
  InlinedVector<uint8_t> fc2(static_cast<size_t>(experts) * 1024 * (intermediate / 256) * 66);
  InlinedVector<int32_t> indices(pairs), metadata(1 + 9 * pairs, -1), unique;
  InlinedVector<float> weights(pairs), partial(pairs * 1024);
  for (int e = 0; e < experts; ++e) {
    const auto s1 = MLFloat16((e + 1) / 256.f).val, s2 = MLFloat16((e % 7 + 1) / 16.f).val;
    for (int block = 0; block < intermediate * 4; ++block) {
      const size_t offset = (static_cast<size_t>(e) * intermediate * 4 + block) * 66;
      fc1[offset] = static_cast<uint8_t>(s1);
      fc1[offset + 1] = static_cast<uint8_t>(s1 >> 8);
    }
    for (int block = 0; block < 1024 * (intermediate / 256); ++block) {
      const size_t offset = (static_cast<size_t>(e) * 1024 * (intermediate / 256) + block) * 66;
      fc2[offset] = static_cast<uint8_t>(s2);
      fc2[offset + 1] = static_cast<uint8_t>(s2 >> 8);
    }
  }
  for (int r = 0; r < rows; ++r) {
    const float activation = BFloat16((r + 1) / 1024.f).ToFloat();
    std::fill_n(x.begin() + r * 1024, 1024, BFloat16(activation).val);
    for (int k = 0; k < topk; ++k) {
      const int p = r * topk + k, e = indices[p] = (r / 2) * topk + k;
      weights[p] = (k + 1) / 32.f;
      auto it = std::find(unique.begin(), unique.end(), e);
      if (it == unique.end()) {
        unique.push_back(e);
        it = unique.end() - 1;
      }
      metadata[1 + pairs + static_cast<int>(it - unique.begin()) * 8 + r] = p;
      const float w1 = BFloat16(MLFloat16((e + 1) / 256.f).ToFloat()).ToFloat();
      float dot = 0.f;
      for (int i = 0; i < 1024; ++i) dot = std::fma(activation, w1, dot);
      const float relu = std::max(BFloat16(dot).ToFloat(), 0.f);
      const BFloat16 h(relu * relu);
      std::fill_n(hidden.begin() + p * intermediate, intermediate, h.val);
      const float w2 = BFloat16(MLFloat16((e % 7 + 1) / 16.f).ToFloat()).ToFloat();
      dot = 0.f;
      for (int i = 0; i < intermediate; ++i) dot = std::fma(h.ToFloat(), w2, dot);
      std::fill_n(partial.begin() + p * 1024, 1024, BFloat16(dot).ToFloat());
    }
    float sum = 0.f;
    for (int k = 0; k < topk; ++k)
      sum = std::fma(weights[r * topk + k], partial[(r * topk + k) * 1024], sum);
    std::fill_n(y.begin() + r * 1024, 1024, BFloat16(sum).val);
  }
  metadata[0] = static_cast<int32_t>(unique.size());
  ASSERT_EQ(unique.size(), static_cast<size_t>(experts));
  std::copy(unique.begin(), unique.end(), metadata.begin() + 1);
  for (int mask : {0, 1, 2, 4, 8, 16, 32, 63}) {
    SCOPED_TRACE(testing::Message() << "intermediate=" << intermediate << ", mask=" << mask);
    const size_t ox = mask & 1 ? 2 : 0, o1 = mask & 2 ? 1 : 0, o2 = mask & 4 ? 1 : 0;
    const size_t oh = mask & 8 ? 2 : 0, op = mask & 16 ? 4 : 0, oy = mask & 32 ? 2 : 0;
    auto dx = Allocate(x.size() * 2 + ox), d1 = Allocate(fc1.size() + o1), d2 = Allocate(fc2.size() + o2);
    auto dh = Allocate(hidden.size() * 2 + oh), dy = Allocate(y.size() * 2 + oy);
    auto dp = Allocate(partial.size() * 4 + op), dm = Allocate(metadata.size() * 4);
    auto di = Allocate(indices.size() * 4), dw = Allocate(weights.size() * 4);
    auto* px = static_cast<uint8_t*>(dx.get()) + ox;
    auto* p1 = static_cast<uint8_t*>(d1.get()) + o1;
    auto* p2 = static_cast<uint8_t*>(d2.get()) + o2;
    auto* ph = static_cast<uint8_t*>(dh.get()) + oh;
    auto* py = static_cast<uint8_t*>(dy.get()) + oy;
    auto* pp = reinterpret_cast<float*>(static_cast<uint8_t*>(dp.get()) + op);
    ASSERT_EQ(cudaMemcpy(px, x.data(), x.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(p1, fc1.data(), fc1.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(p2, fc2.data(), fc2.size(), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(di.get(), indices.data(), indices.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dw.get(), weights.data(), weights.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(contrib::cuda::LaunchMoEPacked(
                  nullptr, rows, topk, experts, intermediate, px, static_cast<int32_t*>(di.get()),
                  static_cast<float*>(dw.get()), p1, p2, py, ph, pp, static_cast<int32_t*>(dm.get())),
              cudaSuccess);
    InlinedVector<uint16_t> actual_hidden(hidden.size()), actual_y(y.size());
    InlinedVector<float> actual_partial(partial.size());
    InlinedVector<int32_t> actual_metadata(metadata.size());
    ASSERT_EQ(cudaMemcpy(actual_hidden.data(), ph, actual_hidden.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_y.data(), py, actual_y.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_partial.data(), pp, actual_partial.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_metadata.data(), dm.get(), actual_metadata.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual_metadata, metadata);
    EXPECT_EQ(actual_hidden, hidden);
    EXPECT_EQ(actual_partial, partial);
    EXPECT_EQ(actual_y, y);
  }
}
}  // namespace

TEST(NativeMoEPackedKernelTest, MultiWarpGroupingNativeBlocksAndOffsets) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (!count) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int intermediate : {256, 1536, 2304})
    ASSERT_NO_FATAL_FAILURE(CheckGroupingAndOffsets(intermediate));
}

TEST(NativeMoEPackedKernelTest, RouterExceptionalRanksAndStableTies) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (!count) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int rows = 1, width = 256, experts = 4, topk = 4;
  std::array<uint16_t, width> x{}, w{};
  auto dx = Allocate(sizeof(x)), dw = Allocate(sizeof(w) * experts), db = Allocate(4 * sizeof(float));
  auto di = Allocate(4 * sizeof(int32_t)), dweights = Allocate(4 * sizeof(float)), ds = Allocate(4 * sizeof(float));
  ASSERT_EQ(cudaMemcpy(dx.get(), x.data(), sizeof(x), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(dw.get(), 0, sizeof(w) * experts), cudaSuccess);
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  for (const auto& biases : {std::array<float, 4>{0.f, 0.f, 0.f, 0.f},
                             std::array<float, 4>{nan, -inf, inf, -0.5f},
                             std::array<float, 4>{nan, nan, nan, nan}}) {
    SCOPED_TRACE(testing::Message() << "first bias=" << biases[0]);
    ASSERT_EQ(cudaMemcpy(db.get(), biases.data(), sizeof(biases), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(contrib::cuda::LaunchMoEPackedRouter(
                  nullptr, rows, width, experts, topk, 2.5f, dx.get(), dw.get(), static_cast<float*>(db.get()),
                  static_cast<int32_t*>(di.get()), static_cast<float*>(dweights.get()), static_cast<float*>(ds.get())),
              cudaSuccess);
    std::array<int32_t, 4> indices{};
    std::array<float, 4> weights{}, scores{};
    ASSERT_EQ(cudaMemcpy(indices.data(), di.get(), sizeof(indices), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(weights.data(), dweights.get(), sizeof(weights), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(scores.data(), ds.get(), sizeof(scores), cudaMemcpyDeviceToHost), cudaSuccess);
    const std::array<int32_t, 4> expected = std::isinf(biases[2]) ? std::array<int32_t, 4>{2, 3, 1, 0}
                                                                  : std::array<int32_t, 4>{0, 1, 2, 3};
    EXPECT_EQ(indices, expected);
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(weights[i], 0.625f);
      EXPECT_EQ(scores[i], 0.5f);
    }
  }
}
}  // namespace onnxruntime::test
#endif
