// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace test {

static void CalculateGlobalAvgPoolU8(
    const uint8_t* x, int64_t batch, int64_t hw, int64_t channel, bool is_nchw, uint8_t* y,
    int32_t x_zero_point, float x_scale, int32_t y_zero_point, float y_scale) {
  int32_t bias = -x_zero_point * gsl::narrow_cast<int32_t>(hw);
  int64_t stride_image = is_nchw ? 1 : channel;
  int64_t stride_channel = is_nchw ? hw : 1;

  for (int64_t b = 0; b < batch; ++b) {
    const uint8_t* bx = x + b * hw * channel;
    uint8_t* by = y + b * channel;
    for (int64_t c = 0; c < channel; ++c) {
      const uint8_t* ix = bx + c * stride_channel;
      int32_t sum = 0;
      for (int64_t i = 0; i < hw; ++i) {
        sum += static_cast<int32_t>(*ix);
        ix += stride_image;
      }
      sum += bias;
      int32_t r = static_cast<int32_t>(std::nearbyintf(x_scale * sum / static_cast<float>(hw) / y_scale));
      r += y_zero_point;
      r = std::min(255, r);
      r = std::max(0, r);
      by[c] = static_cast<uint8_t>(r);
    }
  }
}

void RunQLinearGlobalAveragePoolU8(
    bool is_nchw, int64_t batch, int64_t channel, int64_t h, int64_t w,
    uint8_t x_zero_point, float x_scale, uint8_t y_zero_point, float y_scale, int32_t seed) {
  std::vector<int64_t> x_dims = is_nchw ? std::vector<int64_t>{batch, channel, h, w} : std::vector<int64_t>{batch, h, w, channel};
  std::vector<int64_t> y_dims = is_nchw ? std::vector<int64_t>{batch, channel, 1, 1} : std::vector<int64_t>{batch, 1, 1, channel};
  int64_t x_size = batch * channel * h * w;
  int64_t y_size = batch * channel;
  std::vector<uint8_t> x_data((size_t)x_size);
  std::vector<uint8_t> y_data((size_t)y_size);

  RandomValueGenerator random{ seed ? optional<RandomValueGenerator::RandomSeedType>{seed} : optional<RandomValueGenerator::RandomSeedType>{}};
  std::vector<int> tmp_x_data = random.Uniform<int32_t>(x_dims, 0, 255);
  std::transform(tmp_x_data.begin(), tmp_x_data.end(), x_data.data(), [](int32_t v) -> uint8_t {
    return static_cast<uint8_t>(v);
  });

  CalculateGlobalAvgPoolU8(x_data.data(), batch, h * w, channel, is_nchw, y_data.data(),
                           x_zero_point, x_scale, y_zero_point, y_scale);

  OpTester test("QLinearGlobalAveragePool", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("nchw", is_nchw ? 1LL : 0LL);
  test.AddInput<uint8_t>("X", x_dims, x_data);
  test.AddInput<float>("x_scale", {}, {x_scale});
  test.AddInput<uint8_t>("x_zero_point", {}, {x_zero_point});
  test.AddInput<float>("y_scale", {}, {y_scale});
  test.AddInput<uint8_t>("y_zero_point", {}, {y_zero_point});
  test.AddOutput<uint8_t>("Y", y_dims, y_data);

  test.Run();
}

static int32_t seed = 1234569;

TEST(QLinearGlobalAveragePool, Nchw_1x1x32x32) {
  RunQLinearGlobalAveragePoolU8(true, 1, 1, 32, 32, 128, 1.0, 64, 2.0, seed);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x32x32x1) {
  RunQLinearGlobalAveragePoolU8(false, 1, 1, 32, 32, 128, 1.0, 64, 2.0, seed);
}

TEST(QLinearGlobalAveragePool, Nchw_1x256x8x8) {
  RunQLinearGlobalAveragePoolU8(true, 1, 256, 8, 8, 128, 1.0, 64, 3.0, 89);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x8x8x256) {
  RunQLinearGlobalAveragePoolU8(false, 1, 256, 8, 8, 128, 1.0, 64, 3.0, seed);
}

TEST(QLinearGlobalAveragePool, Nchw_1x255x7x7) {
  RunQLinearGlobalAveragePoolU8(true, 1, 255, 7, 7, 128, 7.0, 128, 21.0, seed);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x7x7x255) {
  RunQLinearGlobalAveragePoolU8(false, 1, 255, 7, 7, 128, 7.0, 128, 21.0, seed);
}

TEST(QLinearGlobalAveragePool, Nchw_1x255x8x8) {
  RunQLinearGlobalAveragePoolU8(true, 1, 255, 8, 8, 128, 1.0, 128, 2.0, 89);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x8x8x255) {
  RunQLinearGlobalAveragePoolU8(false, 1, 255, 8, 8, 128, 1.0, 128, 2.0, seed);
}

TEST(QLinearGlobalAveragePool, Nchw_1x256x7x7) {
  RunQLinearGlobalAveragePoolU8(true, 1, 256, 7, 7, 128, 7.0, 128, 21.0, seed);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x7x7x256) {
  RunQLinearGlobalAveragePoolU8(false, 1, 256, 7, 7, 128, 7.0, 128, 21.0, seed);
}

// tests for BatchSize > 1
TEST(QLinearGlobalAveragePool, Nchw_3x256x8x8) {
  RunQLinearGlobalAveragePoolU8(true, 3, 256, 8, 8, 128, 1.0, 64, 3.0, 99);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x8x8x256) {
  RunQLinearGlobalAveragePoolU8(false, 3, 256, 8, 8, 128, 1.0, 64, 3.0, seed);
}

TEST(QLinearGlobalAveragePool, Nchw_3x255x7x7) {
  RunQLinearGlobalAveragePoolU8(true, 3, 255, 7, 7, 128, 7.0, 128, 21.0, seed);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x7x7x255) {
  RunQLinearGlobalAveragePoolU8(false, 3, 255, 7, 7, 128, 7.0, 128, 21.0, seed);
}

TEST(QLinearGlobalAveragePool, Nchw_3x255x8x8) {
  RunQLinearGlobalAveragePoolU8(true, 3, 255, 8, 8, 128, 1.0, 128, 2.0, 89);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x8x8x255) {
  RunQLinearGlobalAveragePoolU8(false, 3, 255, 8, 8, 128, 1.0, 128, 2.0, 686);
}

TEST(QLinearGlobalAveragePool, Nchw_3x256x7x7) {
  RunQLinearGlobalAveragePoolU8(true, 3, 256, 7, 7, 128, 7.0, 128, 21.0, seed);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x7x7x256) {
  RunQLinearGlobalAveragePoolU8(false, 3, 256, 7, 7, 128, 7.0, 128, 21.0, seed);
}

}
}
