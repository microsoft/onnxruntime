// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace test {

static void CalculateGlobalAvgPoolU8(
    const uint8_t* x, int64_t batch, int64_t hw, int64_t channel, bool channels_last, uint8_t* y,
    int32_t x_zero_point, float x_scale, int32_t y_zero_point, float y_scale) {
  int32_t bias = -x_zero_point * gsl::narrow_cast<int32_t>(hw);
  int64_t stride_image = channels_last ? channel : 1;
  int64_t stride_channel = channels_last ? 1 : hw;

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
    bool channels_last, int64_t batch, int64_t channel, int64_t h, int64_t w,
    uint8_t x_zero_point, float x_scale, uint8_t y_zero_point, float y_scale, int32_t seed = 0) {
  std::vector<int64_t> x_dims = channels_last ? std::vector<int64_t>{batch, h, w, channel} : std::vector<int64_t>{batch, channel, h, w};
  std::vector<int64_t> y_dims = channels_last ? std::vector<int64_t>{batch, 1, 1, channel} : std::vector<int64_t>{batch, channel, 1, 1};
  int64_t x_size = batch * channel * h * w;
  int64_t y_size = batch * channel;
  std::vector<uint8_t> x_data((size_t)x_size);
  std::vector<uint8_t> y_data((size_t)y_size);

  RandomValueGenerator random{seed ? optional<RandomValueGenerator::RandomSeedType>{seed} : optional<RandomValueGenerator::RandomSeedType>{}};
  std::vector<int> tmp_x_data = random.Uniform<int32_t>(x_dims, 0, 255);
  std::transform(tmp_x_data.begin(), tmp_x_data.end(), x_data.data(), [](int32_t v) -> uint8_t {
    return static_cast<uint8_t>(v);
  });

  CalculateGlobalAvgPoolU8(x_data.data(), batch, h * w, channel, channels_last, y_data.data(),
                           x_zero_point, x_scale, y_zero_point, y_scale);

  OpTester test("QLinearGlobalAveragePool", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("channels_last", channels_last ? 1LL : 0LL);
  test.AddInput<uint8_t>("X", x_dims, x_data);
  test.AddInput<float>("x_scale", {}, {x_scale});
  test.AddInput<uint8_t>("x_zero_point", {}, {x_zero_point});
  test.AddInput<float>("y_scale", {}, {y_scale});
  test.AddInput<uint8_t>("y_zero_point", {}, {y_zero_point});
  test.AddOutput<uint8_t>("Y", y_dims, y_data);

  auto q8checker = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    const OrtValue& ort_value = fetches[0];
    if (ort_value.Fence()) {
      ort_value.Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, 0);
    }

    auto y_shape = TensorShape(y_dims);
    const Tensor& output_tensor = ort_value.Get<Tensor>();
    ORT_ENFORCE(y_shape == output_tensor.Shape(),
                "Expected output shape [" + y_shape.ToString() + "] did not match run output shape [" +
                    output_tensor.Shape().ToString() + "] for Y @" + provider_type);
    auto* output = output_tensor.Data<uint8_t>();
    auto size = static_cast<int>(output_tensor.Shape().Size());
    for (int i = 0; i < size; ++i) {
      int diff = abs(y_data[i] - output[i]);
      EXPECT_LE(diff, 1) << "i:" << i << " expected:" << y_data[i] << " " << (int)y_data[i]
                         << ", got:" << output[i] << " " << (int)output[i] << ", provider_type: " << provider_type;
    }
  };
  test.SetCustomOutputVerifier(q8checker);

  test.Run();
}

TEST(QLinearGlobalAveragePool, Nhwc_1x1x32x32) {
  RunQLinearGlobalAveragePoolU8(true, 1, 1, 32, 32, 128, 1.0, 64, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x32x32x1) {
  RunQLinearGlobalAveragePoolU8(false, 1, 1, 32, 32, 128, 1.0, 64, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x256x8x8) {
  RunQLinearGlobalAveragePoolU8(true, 1, 256, 8, 8, 128, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x8x8x256) {
  RunQLinearGlobalAveragePoolU8(false, 1, 256, 8, 8, 128, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x255x7x7) {
  RunQLinearGlobalAveragePoolU8(true, 1, 255, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x7x7x255) {
  RunQLinearGlobalAveragePoolU8(false, 1, 255, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x255x8x8) {
  RunQLinearGlobalAveragePoolU8(true, 1, 255, 8, 8, 128, 1.0, 128, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x8x8x255) {
  RunQLinearGlobalAveragePoolU8(false, 1, 255, 8, 8, 128, 1.0, 128, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x256x7x7) {
  RunQLinearGlobalAveragePoolU8(true, 1, 256, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x7x7x256) {
  RunQLinearGlobalAveragePoolU8(false, 1, 256, 7, 7, 128, 7.0, 128, 21.0);
}

// tests for BatchSize > 1
TEST(QLinearGlobalAveragePool, Nhwc_3x256x8x8) {
  RunQLinearGlobalAveragePoolU8(true, 3, 256, 8, 8, 128, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x8x8x256) {
  RunQLinearGlobalAveragePoolU8(false, 3, 256, 8, 8, 128, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x255x7x7) {
  RunQLinearGlobalAveragePoolU8(true, 3, 255, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x7x7x255) {
  RunQLinearGlobalAveragePoolU8(false, 3, 255, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x255x8x8) {
  RunQLinearGlobalAveragePoolU8(true, 3, 255, 8, 8, 128, 1.0, 128, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x8x8x255) {
  RunQLinearGlobalAveragePoolU8(false, 3, 255, 8, 8, 128, 1.0, 128, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x256x7x7) {
  RunQLinearGlobalAveragePoolU8(true, 3, 256, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x7x7x256) {
  RunQLinearGlobalAveragePoolU8(false, 3, 256, 7, 7, 128, 7.0, 128, 21.0);
}

}  // namespace test
}  // namespace onnxruntime
