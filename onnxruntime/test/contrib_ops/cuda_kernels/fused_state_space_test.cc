// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/sequence/fused_state_space_impl.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <tuple>
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

TEST_F(FusedStateSpaceKernelTest, MambaNormRowsBeyondGridYLimit) {
  // Prefill chunks can exceed CUDA's 65535 grid-y limit, so every row slice of the launch must be normalized.
  constexpr int rows = 65537, width = 64, groups = 2, group_size = width / groups, stride = width + 3;
  constexpr float epsilon = 1e-5f;
  std::vector<uint16_t> projection(static_cast<size_t>(rows) * stride), output(static_cast<size_t>(rows) * width);
  std::vector<float> readout(output.size()), gamma(width), expected(output.size());
  for (int j = 0; j < width; ++j) gamma[j] = (j % 9 + 1) * 0.125f;
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < groups; ++g) {
      float v[group_size];
      float ss = 0.f;
      for (int i = 0; i < group_size; ++i) {
        const int j = g * group_size + i;
        const size_t index = static_cast<size_t>(r) * width + j;
        const float gate = ((r + j) % 4 + 1) * 0.25f;
        projection[static_cast<size_t>(r) * stride + j] = ToBf16(gate);
        readout[index] = ((r * 7 + j) % 11 - 5) * 0.125f;
        v[i] = readout[index] * (gate / (1.f + std::exp(-gate)));
        ss += v[i] * v[i];
      }
      const float rs = 1.f / std::sqrt(ss / group_size + epsilon);
      for (int i = 0; i < group_size; ++i) {
        expected[static_cast<size_t>(r) * width + g * group_size + i] = v[i] * rs * gamma[g * group_size + i];
      }
    }
  }
  auto device_proj = Allocate(projection.size() * sizeof(uint16_t));
  auto device_readout = Allocate(readout.size() * sizeof(float));
  auto device_gamma = Allocate(gamma.size() * sizeof(float));
  auto device_output = Allocate(output.size() * sizeof(uint16_t));
  for (const auto* ptr : {device_proj.get(), device_readout.get(), device_gamma.get(), device_output.get()})
    ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(cudaMemcpy(device_proj.get(), projection.data(), projection.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_readout.get(), readout.data(), readout.size() * sizeof(float),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(device_gamma.get(), gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(device_output.get(), 0xff, output.size() * sizeof(uint16_t)), cudaSuccess);
  ASSERT_EQ(contrib::cuda::LaunchMamba2VerifyNormalization(
                nullptr, rows, width, groups, stride, epsilon, device_proj.get(),
                static_cast<const float*>(device_readout.get()), static_cast<const float*>(device_gamma.get()),
                device_output.get()),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(output.data(), device_output.get(), output.size() * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (size_t index = 0; index < output.size(); ++index) {
    const float actual = FromBf16(output[index]);
    if (!std::isfinite(actual) || std::abs(actual - expected[index]) > 0.004f * std::abs(expected[index]) + 1e-6f) {
      FAIL() << "row " << index / width << " col " << index % width << ": " << actual << " vs " << expected[index];
    }
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

TEST_F(FusedStateSpaceKernelTest, MambaSingleRowReplayPreservesMultiRowReadout) {
  constexpr int rows = 8, heads = 4, groups = 2, width = heads * 64;
  constexpr int channels = width + groups * 256, stride = width + channels + heads;
  constexpr size_t state_row_size = heads * 128 * 64;
  std::vector<uint16_t> projection(rows * stride), conv(rows * channels * 3);
  std::vector<uint16_t> weight(channels * 4), bias(channels);
  std::vector<float> ssm(rows * state_row_size), params(heads * 3 + width, 1.f);
  for (size_t i = 0; i < projection.size(); ++i)
    projection[i] = ToBf16((static_cast<int>(i % 23) - 11) / 64.f);
  for (size_t i = 0; i < conv.size(); ++i)
    conv[i] = ToBf16((static_cast<int>(i % 17) - 8) / 128.f);
  for (size_t i = 0; i < weight.size(); ++i)
    weight[i] = ToBf16((static_cast<int>(i % 7) - 3) / 16.f);
  for (size_t i = 0; i < ssm.size(); ++i)
    ssm[i] = (static_cast<int>(i % 19) - 9) / 1024.f;
  std::fill_n(params.begin() + heads, heads, -0.125f);
  auto dp = Allocate(projection.size() * 2), dc = Allocate(conv.size() * 2);
  auto ds = Allocate(ssm.size() * 4), dw = Allocate(weight.size() * 2);
  auto db = Allocate(bias.size() * 2), parameters = Allocate(params.size() * 4);
  auto selector = Allocate(sizeof(int32_t));
  auto y = Allocate(rows * width * 2), co = Allocate(conv.size() * 2);
  auto so = Allocate(ssm.size() * 4), xbc = Allocate(rows * channels * 2);
  auto readout = Allocate(rows * width * 4);
  for (const auto* ptr : {dp.get(), dc.get(), ds.get(), dw.get(), db.get(), parameters.get(),
                          selector.get(), y.get(), co.get(), so.get(), xbc.get(), readout.get()})
    ASSERT_NE(ptr, nullptr);
  for (const auto& entry : {std::pair{dp.get(), std::pair{projection.data(), projection.size() * 2}},
                            std::pair{dc.get(), std::pair{conv.data(), conv.size() * 2}},
                            std::pair{dw.get(), std::pair{weight.data(), weight.size() * 2}},
                            std::pair{db.get(), std::pair{bias.data(), bias.size() * 2}}})
    ASSERT_EQ(cudaMemcpy(entry.first, entry.second.first, entry.second.second, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(ds.get(), ssm.data(), ssm.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(parameters.get(), params.data(), params.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(selector.get(), 0, sizeof(int32_t)), cudaSuccess);
  const auto* p = static_cast<const float*>(parameters.get());
  const auto launch = [&](int count, const void* input, void* conv_output, float* ssm_output) {
    return contrib::cuda::LaunchMamba2Verify(
        nullptr, count, rows, heads, groups, 1e-5f, 0.001f, input, dc.get(),
        static_cast<const float*>(ds.get()), dw.get(), db.get(), p, p + heads, p + heads * 2,
        p + heads * 3, static_cast<const int32_t*>(selector.get()), y.get(), conv_output,
        ssm_output, xbc.get(), static_cast<float*>(readout.get()));
  };
  ASSERT_EQ(launch(rows, dp.get(), co.get(), static_cast<float*>(so.get())), cudaSuccess);
  std::vector<uint16_t> expected_y(rows * width), expected_conv(conv.size());
  std::vector<float> expected_readout(rows * width), expected_ssm(ssm.size());
  ASSERT_EQ(cudaMemcpy(expected_y.data(), y.get(), expected_y.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(expected_conv.data(), co.get(), expected_conv.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(expected_readout.data(), readout.get(), expected_readout.size() * 4,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(expected_ssm.data(), so.get(), expected_ssm.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
  for (int row = 0; row < rows; ++row) {
    const auto* input = static_cast<const uint16_t*>(dp.get()) + row * stride;
    ASSERT_EQ(launch(1, input, dc.get(), static_cast<float*>(ds.get())), cudaSuccess);
    std::vector<uint16_t> actual_y(width), actual_conv(channels * 3);
    std::vector<float> actual_readout(width), actual_ssm(state_row_size);
    ASSERT_EQ(cudaMemcpy(actual_y.data(), y.get(), actual_y.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_conv.data(), dc.get(), actual_conv.size() * 2, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_readout.data(), readout.get(), actual_readout.size() * 4,
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(actual_ssm.data(), ds.get(), actual_ssm.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(std::memcmp(actual_y.data(), expected_y.data() + row * width, width * 2), 0) << row;
    EXPECT_EQ(std::memcmp(actual_conv.data(), expected_conv.data() + row * channels * 3,
                          actual_conv.size() * 2),
              0)
        << row;
    EXPECT_EQ(std::memcmp(actual_readout.data(), expected_readout.data() + row * width, width * 4), 0) << row;
    EXPECT_EQ(std::memcmp(actual_ssm.data(), expected_ssm.data() + row * state_row_size,
                          state_row_size * 4),
              0)
        << row;
  }
}

TEST_F(FusedStateSpaceKernelTest, MambaSingleSlotRetainsTwoPartitionReadout) {
  constexpr int heads = 1, groups = 1, width = 64, channels = width + 256;
  constexpr int stride = width + channels + heads;
  std::vector<uint16_t> projection(stride), conv(channels * 3), weight(channels * 4), bias(channels);
  // x=B=0, C=BF16(SiLU(1)); raw dt=32 and A=0 keep the state bitwise unchanged.
  std::fill(bias.begin() + width + 128, bias.end(), ToBf16(1.f));
  projection.back() = ToBf16(32.f);
  std::vector<float> ssm(128 * width);
  for (int col = 0; col < width; ++col) {
    ssm[col] = 0x1p25f;
    ssm[32 * width + col] = 1.f;
    ssm[64 * width + col] = -0x1p25f;
    ssm[96 * width + col] = 1.f;
  }
  std::vector<float> params(heads * 3 + width, 0.f);
  std::fill(params.begin() + heads * 3, params.end(), 1.f);
  auto dp = Allocate(projection.size() * 2), dc = Allocate(conv.size() * 2);
  auto ds = Allocate(ssm.size() * 4), dw = Allocate(weight.size() * 2), db = Allocate(bias.size() * 2);
  auto parameters = Allocate(params.size() * 4), selector = Allocate(sizeof(int32_t));
  auto y = Allocate(width * 2), co = Allocate(conv.size() * 2), so = Allocate(ssm.size() * 4);
  auto xbc = Allocate(channels * 2), readout = Allocate(width * 4);
  for (const auto* ptr : {dp.get(), dc.get(), ds.get(), dw.get(), db.get(), parameters.get(),
                          selector.get(), y.get(), co.get(), so.get(), xbc.get(), readout.get()})
    ASSERT_NE(ptr, nullptr);
  for (const auto& entry : {std::pair{dp.get(), std::pair{projection.data(), projection.size() * 2}},
                            std::pair{dc.get(), std::pair{conv.data(), conv.size() * 2}},
                            std::pair{dw.get(), std::pair{weight.data(), weight.size() * 2}},
                            std::pair{db.get(), std::pair{bias.data(), bias.size() * 2}}})
    ASSERT_EQ(cudaMemcpy(entry.first, entry.second.first, entry.second.second, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(ds.get(), ssm.data(), ssm.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(parameters.get(), params.data(), params.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemset(selector.get(), 0, sizeof(int32_t)), cudaSuccess);
  const auto* p = static_cast<const float*>(parameters.get());
  ASSERT_EQ(contrib::cuda::LaunchMamba2Verify(
                nullptr, 1, 1, heads, groups, 1e-5f, 0.001f, dp.get(), dc.get(),
                static_cast<const float*>(ds.get()), dw.get(), db.get(), p, p + heads, p + heads * 2,
                p + heads * 3, static_cast<const int32_t*>(selector.get()), y.get(), co.get(),
                static_cast<float*>(so.get()), xbc.get(), static_cast<float*>(readout.get())),
            cudaSuccess);
  std::vector<float> actual_readout(width), actual_ssm(ssm.size());
  ASSERT_EQ(cudaMemcpy(actual_readout.data(), readout.get(), width * 4, cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(actual_ssm.data(), so.get(), ssm.size() * 4, cudaMemcpyDeviceToHost), cudaSuccess);
  // Two partitions lose both small addends before cancellation; four retain the last C.
  const std::vector<float> expected_readout(width, 0.f);
  EXPECT_EQ(std::memcmp(actual_readout.data(), expected_readout.data(), width * 4), 0);
  EXPECT_EQ(std::memcmp(actual_ssm.data(), ssm.data(), ssm.size() * 4), 0);
}

TEST_F(FusedStateSpaceKernelTest, MambaPrefillInPlaceStatesMatchSeparateOutputs) {
  // The op allows input states to alias their snapshot outputs. With state_row=0 and R<=slots, snapshot row 0
  // overwrites the starting convolution state before later snapshot rows have read it.
  constexpr int heads = 4, groups = 2, width = heads * 64, channels = width + groups * 256;
  constexpr int stride = width + channels + heads;
  constexpr size_t conv_row = channels * 3, ssm_row = heads * 128 * 64;
  for (const auto& config : {std::tuple{20, 24, 0}, std::tuple{37, 8, 0}, std::tuple{37, 8, 5}}) {
    const int rows = std::get<0>(config), slots = std::get<1>(config);
    const int32_t selected = std::get<2>(config);
    SCOPED_TRACE(testing::Message() << "rows=" << rows << " slots=" << slots << " state_row=" << selected);
    const size_t snapshots = static_cast<size_t>(std::min(rows, slots));
    std::vector<uint16_t> projection(static_cast<size_t>(rows) * stride), conv(slots * conv_row);
    std::vector<uint16_t> weight(channels * 4), bias(channels);
    std::vector<float> ssm(slots * ssm_row), params(heads * 3 + width, 1.f);
    for (size_t i = 0; i < projection.size(); ++i) projection[i] = ToBf16((static_cast<int>(i % 23) - 11) / 64.f);
    for (size_t i = 0; i < conv.size(); ++i) conv[i] = ToBf16((static_cast<int>(i % 17) - 8) / 128.f);
    for (size_t i = 0; i < weight.size(); ++i) weight[i] = ToBf16((static_cast<int>(i % 7) - 3) / 16.f);
    for (size_t i = 0; i < ssm.size(); ++i) ssm[i] = (static_cast<int>(i % 19) - 9) / 1024.f;
    std::fill_n(params.begin() + heads, heads, -0.125f);
    auto dp = Allocate(projection.size() * 2), dw = Allocate(weight.size() * 2), db = Allocate(bias.size() * 2);
    auto dc = Allocate(conv.size() * 2), ds = Allocate(ssm.size() * 4);
    auto parameters = Allocate(params.size() * 4), selector = Allocate(sizeof(int32_t));
    auto separate_conv = Allocate(conv.size() * 2), separate_ssm = Allocate(ssm.size() * 4);
    auto inplace_conv = Allocate(conv.size() * 2), inplace_ssm = Allocate(ssm.size() * 4);
    auto y = Allocate(static_cast<size_t>(rows) * width * 2), xbc = Allocate(static_cast<size_t>(rows) * channels * 2);
    auto readout = Allocate(static_cast<size_t>(rows) * width * 4);
    for (const auto* ptr : {dp.get(), dw.get(), db.get(), dc.get(), ds.get(), parameters.get(), selector.get(),
                            separate_conv.get(), separate_ssm.get(), inplace_conv.get(), inplace_ssm.get(), y.get(),
                            xbc.get(), readout.get()})
      ASSERT_NE(ptr, nullptr);
    for (const auto& entry : {std::pair{dp.get(), std::pair{projection.data(), projection.size() * 2}},
                              std::pair{dc.get(), std::pair{conv.data(), conv.size() * 2}},
                              std::pair{inplace_conv.get(), std::pair{conv.data(), conv.size() * 2}},
                              std::pair{dw.get(), std::pair{weight.data(), weight.size() * 2}},
                              std::pair{db.get(), std::pair{bias.data(), bias.size() * 2}}})
      ASSERT_EQ(cudaMemcpy(entry.first, entry.second.first, entry.second.second, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(ds.get(), ssm.data(), ssm.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(inplace_ssm.get(), ssm.data(), ssm.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(parameters.get(), params.data(), params.size() * 4, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(selector.get(), &selected, sizeof(selected), cudaMemcpyHostToDevice), cudaSuccess);
    const auto* p = static_cast<const float*>(parameters.get());
    const auto launch = [&](const void* conv_in, const float* ssm_in, void* conv_out, float* ssm_out) {
      return contrib::cuda::LaunchMamba2Verify(
          nullptr, rows, slots, heads, groups, 1e-5f, 0.001f, dp.get(), conv_in, ssm_in, dw.get(), db.get(), p,
          p + heads, p + heads * 2, p + heads * 3, static_cast<const int32_t*>(selector.get()), y.get(), conv_out,
          ssm_out, xbc.get(), static_cast<float*>(readout.get()));
    };
    const auto read = [](const void* device, size_t bytes) {
      std::vector<uint8_t> host(bytes);
      EXPECT_EQ(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
      return host;
    };
    const size_t y_bytes = static_cast<size_t>(rows) * width * 2, xbc_bytes = static_cast<size_t>(rows) * channels * 2;
    const size_t readout_bytes = static_cast<size_t>(rows) * width * 4;
    ASSERT_EQ(launch(dc.get(), static_cast<const float*>(ds.get()), separate_conv.get(),
                     static_cast<float*>(separate_ssm.get())),
              cudaSuccess);
    const auto expected_y = read(y.get(), y_bytes), expected_xbc = read(xbc.get(), xbc_bytes);
    const auto expected_readout = read(readout.get(), readout_bytes);
    const auto expected_conv = read(separate_conv.get(), snapshots * conv_row * 2);
    const auto expected_ssm = read(separate_ssm.get(), snapshots * ssm_row * 4);
    ASSERT_EQ(launch(inplace_conv.get(), static_cast<const float*>(inplace_ssm.get()), inplace_conv.get(),
                     static_cast<float*>(inplace_ssm.get())),
              cudaSuccess);
    EXPECT_EQ(read(y.get(), y_bytes), expected_y);
    EXPECT_EQ(read(xbc.get(), xbc_bytes), expected_xbc);
    EXPECT_EQ(read(readout.get(), readout_bytes), expected_readout);
    const auto actual_conv = read(inplace_conv.get(), conv.size() * 2);
    const auto actual_ssm = read(inplace_ssm.get(), ssm.size() * 4);
    const auto* conv_bytes = reinterpret_cast<const uint8_t*>(conv.data());
    const auto* ssm_bytes = reinterpret_cast<const uint8_t*>(ssm.data());
    EXPECT_TRUE(std::equal(expected_conv.begin(), expected_conv.end(), actual_conv.begin()));
    EXPECT_TRUE(std::equal(expected_ssm.begin(), expected_ssm.end(), actual_ssm.begin()));
    // Slots without a snapshot keep their input state.
    EXPECT_TRUE(std::equal(actual_conv.begin() + expected_conv.size(), actual_conv.end(),
                           conv_bytes + expected_conv.size()));
    EXPECT_TRUE(std::equal(actual_ssm.begin() + expected_ssm.size(), actual_ssm.end(),
                           ssm_bytes + expected_ssm.size()));
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
