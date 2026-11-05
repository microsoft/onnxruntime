// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sequence/fused_state_space_impl.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
namespace onnxruntime::test {
namespace {
using DeviceBuffer = std::unique_ptr<void, decltype(&cudaFree)>;

DeviceBuffer Allocate(size_t bytes) {
  void* ptr = nullptr;
  EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
  return DeviceBuffer(ptr, cudaFree);
}

uint16_t ToBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

float FromBf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

class FusedStateSpaceKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    if (count == 0) GTEST_SKIP() << "Requires CUDA";
    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
    if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
};

TEST_F(FusedStateSpaceKernelTest, MambaNorm65536Groups) {
  // Isolate the production normalization launch without allocating full recurrent states.
  constexpr int rows = 1, width = 65536, groups = width, stride = width * 5;
  constexpr float epsilon = 1e-5f;
  std::vector<uint16_t> projection(stride), output(width);
  std::vector<float> readout(width), gamma(width), expected(width);
  for (int j = 0; j < width; ++j) {
    const float gate = (j % 4 + 1) * 0.25f;
    projection[j] = ToBf16(gate);
    readout[j] = (j % 11 - 5) * 0.125f;
    gamma[j] = (j % 9 + 1) * 0.125f;
    const float v = readout[j] * (gate / (1.f + std::exp(-gate)));
    expected[j] = v / std::sqrt(v * v + epsilon) * gamma[j];
  }
  auto device_proj = Allocate(projection.size() * sizeof(uint16_t));
  auto device_readout = Allocate(readout.size() * sizeof(float));
  auto device_gamma = Allocate(gamma.size() * sizeof(float));
  auto device_output = Allocate(output.size() * sizeof(uint16_t));
  ASSERT_NE(device_proj, nullptr);
  ASSERT_NE(device_readout, nullptr);
  ASSERT_NE(device_gamma, nullptr);
  ASSERT_NE(device_output, nullptr);
  ASSERT_EQ(cudaMemcpy(device_proj.get(), projection.data(), projection.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_readout.get(), readout.data(), readout.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_gamma.get(), gamma.data(), gamma.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output.get(), 0xff, output.size() * sizeof(uint16_t)), cudaSuccess);
  // Restoring dim3(rows, groups) exceeds CUDA's grid-y limit.
  ASSERT_EQ(contrib::cuda::LaunchMamba2VerifyNormalization(
                nullptr, rows, width, groups, stride, epsilon, device_proj.get(),
                static_cast<const float*>(device_readout.get()), static_cast<const float*>(device_gamma.get()),
                device_output.get()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int j = 0; j < width; ++j) {
    const float actual = FromBf16(output[j]);
    ASSERT_TRUE(std::isfinite(actual)) << j;
    EXPECT_NEAR(actual, expected[j], 0.004f * std::abs(expected[j]) + 1e-6f) << j;
  }
}

TEST_F(FusedStateSpaceKernelTest, MambaOffsetConvolutionWeights) {
  constexpr int heads = 4, groups = 2, width = heads * 64, channels = width + groups * 256;
  constexpr int stride = width + channels + heads;
  for (int rows : {1, 8}) {
    const int slots = rows + 3;
    std::vector<uint16_t> projection(rows * stride), conv(slots * channels * 3);
    std::vector<uint16_t> weight(channels * 4), bias(channels);
    std::vector<float> ssm(static_cast<size_t>(slots) * heads * 128 * 64);
    for (size_t i = 0; i < projection.size(); ++i) projection[i] = ToBf16((static_cast<int>(i % 23) - 11) / 64.f);
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = ToBf16((static_cast<int>(i % 17) - 8) / 128.f);
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = ToBf16((static_cast<int>(i % 7) - 3) / 16.f);
    for (size_t i = 0; i < ssm.size(); ++i) ssm[i] = (static_cast<int>(i % 19) - 9) / 1024.f;
    auto dp = Allocate(projection.size() * 2), dc = Allocate(conv.size() * 2);
    auto ds = Allocate(ssm.size() * 4), dw = Allocate(weight.size() * 2 + 2);
    auto db = Allocate(bias.size() * 2), parameters = Allocate((heads * 3 + width) * sizeof(float));
    auto selector = Allocate(sizeof(int32_t));
    auto y = Allocate(static_cast<size_t>(rows) * width * 2), co = Allocate(conv.size() * 2);
    auto so = Allocate(ssm.size() * 4), xbc = Allocate(static_cast<size_t>(rows) * channels * 2);
    auto readout = Allocate(static_cast<size_t>(rows) * width * 4);
    for (const auto* ptr : {dp.get(), dc.get(), ds.get(), dw.get(), db.get(), parameters.get(),
                            selector.get(), y.get(), co.get(), so.get(), xbc.get(), readout.get()})
      ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(cudaMemcpy(dp.get(), projection.data(), projection.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dc.get(), conv.data(), conv.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ds.get(), ssm.data(), ssm.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(db.get(), bias.data(), bias.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<float> params(heads * 3 + width, 1.f);
    std::fill_n(params.begin() + heads, heads, -0.125f);
    ASSERT_EQ(cudaMemcpy(parameters.get(), params.data(), params.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    const int32_t selected = 3;
    ASSERT_EQ(cudaMemcpy(selector.get(), &selected, sizeof(selected), cudaMemcpyHostToDevice), cudaSuccess);
    std::vector<uint8_t> expected;
    for (size_t offset : {size_t{0}, size_t{2}}) {
      auto* wp = static_cast<uint8_t*>(dw.get()) + offset;
      ASSERT_EQ(cudaMemcpy(wp, weight.data(), weight.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(cudaMemset(co.get(), 0xff, conv.size() * 2), cudaSuccess);
      ASSERT_EQ(cudaMemset(so.get(), 0xff, ssm.size() * 4), cudaSuccess);
      const size_t conv_offset = static_cast<size_t>(rows) * channels * 3 * 2;
      const size_t ssm_offset = static_cast<size_t>(rows) * heads * 128 * 64 * 4;
      ASSERT_EQ(cudaMemcpy(static_cast<uint8_t*>(co.get()) + conv_offset,
                           static_cast<const uint8_t*>(dc.get()) + conv_offset,
                           conv.size() * 2 - conv_offset, cudaMemcpyDeviceToDevice),
                cudaSuccess);
      ASSERT_EQ(cudaMemcpy(static_cast<uint8_t*>(so.get()) + ssm_offset,
                           static_cast<const uint8_t*>(ds.get()) + ssm_offset,
                           ssm.size() * 4 - ssm_offset, cudaMemcpyDeviceToDevice),
                cudaSuccess);
      const auto* p = static_cast<const float*>(parameters.get());
      ASSERT_EQ(contrib::cuda::LaunchMamba2Verify(
                    nullptr, rows, slots, heads, groups, 1e-5f, 0.001f, dp.get(), dc.get(),
                    static_cast<const float*>(ds.get()), wp, db.get(), p, p + heads, p + heads * 2,
                    p + heads * 3, static_cast<const int32_t*>(selector.get()), y.get(), co.get(),
                    static_cast<float*>(so.get()), xbc.get(), static_cast<float*>(readout.get())),
                cudaSuccess);
      std::vector<uint8_t> actual;
      for (const auto& entry : {std::pair{y.get(), static_cast<size_t>(rows) * width * 2},
                                std::pair{co.get(), conv.size() * 2},
                                std::pair{so.get(), ssm.size() * 4},
                                std::pair{xbc.get(), static_cast<size_t>(rows) * channels * 2},
                                std::pair{readout.get(), static_cast<size_t>(rows) * width * 4}}) {
        const auto begin = actual.size();
        actual.resize(begin + entry.second);
        ASSERT_EQ(cudaMemcpy(actual.data() + begin, entry.first, entry.second, cudaMemcpyDeviceToHost), cudaSuccess);
      }
      if (offset == 0)
        expected = std::move(actual);
      else
        EXPECT_EQ(actual, expected) << "rows=" << rows;
    }
  }
}

TEST_F(FusedStateSpaceKernelTest, RouterOffsetBindingsAndIndependentReference) {
  constexpr int rows = 3, width = 768, topk = 32;
  constexpr float scale = 2.5f;
  for (int experts : {511, 512}) {
    std::vector<uint16_t> x(rows * width), weight(experts * width);
    std::vector<float> bias(experts), scores(rows * experts);
    for (int r = 0; r < rows; ++r)
      for (int k = 0; k < width; ++k) x[r * width + k] = ToBf16(((k * 3 + r * 5) % 23 - 11) / 32.f);
    for (int e = 0; e < experts; ++e) {
      bias[e] = (e % 31 - 15) / 128.f;
      for (int k = 0; k < width; ++k)
        weight[e * width + k] = ToBf16(((e * 7 + k * 5 + k / 256 * 11) % 29 - 14) / 64.f);
    }
    std::vector<int32_t> indices(rows * topk);
    std::vector<float> weights(rows * topk);
    for (int r = 0; r < rows; ++r) {
      for (int e = 0; e < experts; ++e) {
        float dot = 0.f;
        for (int k = 0; k < width; ++k)
          dot += FromBf16(x[r * width + k]) * FromBf16(weight[e * width + k]);
        scores[r * experts + e] = 1.f / (1.f + std::exp(-dot));
      }
      std::vector<int32_t> order(experts);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return scores[r * experts + a] + bias[a] > scores[r * experts + b] + bias[b];
      });
      float sum = 0.f;
      for (int j = 0; j < topk; ++j) sum += scores[r * experts + order[j]];
      for (int j = 0; j < topk; ++j) {
        indices[r * topk + j] = order[j];
        weights[r * topk + j] = scores[r * experts + order[j]] / (sum + 1e-20f) * scale;
      }
    }
    auto dx = Allocate(x.size() * 2 + 2), dw = Allocate(weight.size() * 2 + 2);
    auto db = Allocate(bias.size() * 4), di = Allocate(indices.size() * 4);
    auto dout = Allocate(weights.size() * 4), ds = Allocate(scores.size() * 4);
    for (const auto* ptr : {dx.get(), dw.get(), db.get(), di.get(), dout.get(), ds.get()}) ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(cudaMemcpy(db.get(), bias.data(), bias.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    // Cover independently offset X/W and both offset, not just the naturally aligned allocator case.
    for (int mask = 0; mask < 4; ++mask) {
      auto* xp = static_cast<uint8_t*>(dx.get()) + ((mask & 1) ? 2 : 0);
      auto* wp = static_cast<uint8_t*>(dw.get()) + ((mask & 2) ? 2 : 0);
      ASSERT_EQ(cudaMemcpy(xp, x.data(), x.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(wp, weight.data(), weight.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(contrib::cuda::LaunchBiasCorrectedMoERoute(
                    nullptr, rows, width, experts, topk, scale, xp, wp, static_cast<const float*>(db.get()),
                    static_cast<int32_t*>(di.get()), static_cast<float*>(dout.get()), static_cast<float*>(ds.get())),
                cudaSuccess);
      std::vector<int32_t> actual_indices(indices.size());
      std::vector<float> actual_weights(weights.size()), actual_scores(scores.size());
      ASSERT_EQ(cudaMemcpy(actual_indices.data(), di.get(), indices.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(actual_weights.data(), dout.get(), weights.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(actual_scores.data(), ds.get(), scores.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(actual_indices, indices) << "experts=" << experts << ", binding mask=" << mask;
      for (size_t i = 0; i < weights.size(); ++i) EXPECT_NEAR(actual_weights[i], weights[i], 2e-6f) << i;
      for (size_t i = 0; i < scores.size(); ++i) EXPECT_NEAR(actual_scores[i], scores[i], 2e-6f) << i;
    }
  }
}

TEST_F(FusedStateSpaceKernelTest, RouterNaNsSortLastWithStableExpertTies) {
  constexpr int width = 256;
  for (int experts : {2, 40}) {
    const int topk = std::min(experts, 32);
    std::vector<uint16_t> x(width), weight(experts * width);
    std::vector<float> bias(experts, std::numeric_limits<float>::quiet_NaN());
    const int numeric_expert = experts == 2 ? 1 : 33;
    bias[numeric_expert] = 0.f;
    auto dx = Allocate(x.size() * 2), dw = Allocate(weight.size() * 2);
    auto db = Allocate(bias.size() * 4), di = Allocate(topk * sizeof(int32_t));
    auto dout = Allocate(topk * sizeof(float)), ds = Allocate(experts * sizeof(float));
    for (const auto* ptr : {dx.get(), dw.get(), db.get(), di.get(), dout.get(), ds.get()}) ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(cudaMemcpy(dx.get(), x.data(), x.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(dw.get(), weight.data(), weight.size() * 2, cudaMemcpyHostToDevice), cudaSuccess);
    for (bool all_nan : {false, true}) {
      if (all_nan) bias[numeric_expert] = std::numeric_limits<float>::quiet_NaN();
      ASSERT_EQ(cudaMemcpy(db.get(), bias.data(), bias.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
      ASSERT_EQ(contrib::cuda::LaunchBiasCorrectedMoERoute(
                    nullptr, 1, width, experts, topk, 1.f, dx.get(), dw.get(), static_cast<const float*>(db.get()),
                    static_cast<int32_t*>(di.get()), static_cast<float*>(dout.get()), static_cast<float*>(ds.get())),
                cudaSuccess);
      std::vector<int32_t> actual(topk), expected;
      if (!all_nan) expected.push_back(numeric_expert);
      for (int e = 0; expected.size() < static_cast<size_t>(topk); ++e)
        if (all_nan || e != numeric_expert) expected.push_back(e);
      std::vector<float> weights(topk);
      ASSERT_EQ(cudaMemcpy(actual.data(), di.get(), topk * sizeof(int32_t), cudaMemcpyDeviceToHost), cudaSuccess);
      ASSERT_EQ(cudaMemcpy(weights.data(), dout.get(), topk * sizeof(float), cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(actual, expected);
      for (float value : weights) EXPECT_FLOAT_EQ(value, 1.f / topk);
    }
  }
}
}  // namespace
}  // namespace onnxruntime::test
#endif
