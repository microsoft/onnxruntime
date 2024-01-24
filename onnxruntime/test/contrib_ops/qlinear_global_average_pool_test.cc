// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace test {

template <typename T8Bits>
static void CalculateGlobalAvgPool(
    const T8Bits* x, int64_t batch, int64_t hw, int64_t channel, bool channels_last, T8Bits* y,
    int32_t x_zero_point, float x_scale, int32_t y_zero_point, float y_scale) {
  int32_t bias = -x_zero_point * gsl::narrow_cast<int32_t>(hw);
  int64_t stride_image = channels_last ? channel : 1;
  int64_t stride_channel = channels_last ? 1 : hw;

  for (int64_t b = 0; b < batch; ++b) {
    const T8Bits* bx = x + b * hw * channel;
    T8Bits* by = y + b * channel;
    for (int64_t c = 0; c < channel; ++c) {
      const T8Bits* ix = bx + c * stride_channel;
      int32_t sum = 0;
      for (int64_t i = 0; i < hw; ++i) {
        sum += static_cast<int32_t>(*ix);
        ix += stride_image;
      }
      sum += bias;
      int32_t r = static_cast<int32_t>(std::nearbyintf(x_scale * sum / static_cast<float>(hw) / y_scale));
      r += y_zero_point;
      r = std::min((int32_t)(std::numeric_limits<T8Bits>::max()), r);
      r = std::max((int32_t)(std::numeric_limits<T8Bits>::lowest()), r);
      by[c] = static_cast<T8Bits>(r);
    }
  }
}

template <typename T8Bits = uint8_t>
void RunQLinearGlobalAveragePool(
    bool channels_last, int64_t batch, int64_t channel, int64_t h, int64_t w,
    T8Bits x_zero_point, float x_scale, T8Bits y_zero_point, float y_scale, int32_t seed = 0) {
  std::vector<int64_t> x_dims = channels_last ? std::vector<int64_t>{batch, h, w, channel} : std::vector<int64_t>{batch, channel, h, w};
  std::vector<int64_t> y_dims = channels_last ? std::vector<int64_t>{batch, 1, 1, channel} : std::vector<int64_t>{batch, channel, 1, 1};
  int64_t x_size = batch * channel * h * w;
  int64_t y_size = batch * channel;
  std::vector<T8Bits> x_data((size_t)x_size);
  std::vector<T8Bits> y_data((size_t)y_size);

  RandomValueGenerator random{seed ? optional<RandomValueGenerator::RandomSeedType>{seed} : optional<RandomValueGenerator::RandomSeedType>{}};
  std::vector<int> tmp_x_data = random.Uniform<int32_t>(x_dims, std::numeric_limits<T8Bits>::lowest(), std::numeric_limits<T8Bits>::max());
  std::transform(tmp_x_data.begin(), tmp_x_data.end(), x_data.data(), [](int32_t v) -> T8Bits {
    return static_cast<T8Bits>(v);
  });

  CalculateGlobalAvgPool(x_data.data(), batch, h * w, channel, channels_last, y_data.data(),
                         x_zero_point, x_scale, y_zero_point, y_scale);

  OpTester test("QLinearGlobalAveragePool", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("channels_last", channels_last ? 1LL : 0LL);
  test.AddInput<T8Bits>("X", x_dims, x_data);
  test.AddInput<float>("x_scale", {}, {x_scale});
  test.AddInput<T8Bits>("x_zero_point", {}, {x_zero_point});
  test.AddInput<float>("y_scale", {}, {y_scale});
  test.AddInput<T8Bits>("y_zero_point", {}, {y_zero_point});
  test.AddOutput<T8Bits>("Y", y_dims, y_data);
  if (channels_last) {
    test.AddAttribute("channels_last", (int64_t)1LL);
  }

  auto q8checker = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    const OrtValue& ort_value = fetches[0];
    auto y_shape = TensorShape(y_dims);
    const Tensor& output_tensor = ort_value.Get<Tensor>();
    ORT_ENFORCE(y_shape == output_tensor.Shape(),
                "Expected output shape [" + y_shape.ToString() + "] did not match run output shape [" +
                    output_tensor.Shape().ToString() + "] for Y @" + provider_type);
    auto* output = output_tensor.Data<T8Bits>();
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
  RunQLinearGlobalAveragePool<uint8_t>(true, 1, 1, 32, 32, 128, 1.0, 64, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x32x32x1) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 1, 1, 32, 32, 128, 1.0, 64, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x256x8x8) {
  RunQLinearGlobalAveragePool<uint8_t>(true, 1, 256, 8, 8, 128, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x8x8x256) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 1, 256, 8, 8, 128, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x255x7x7) {
  RunQLinearGlobalAveragePool<uint8_t>(true, 1, 255, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x7x7x255) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 1, 255, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x255x8x8) {
  RunQLinearGlobalAveragePool<uint8_t>(true, 1, 255, 8, 8, 128, 1.0, 128, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x8x8x255) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 1, 255, 8, 8, 128, 1.0, 128, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x256x7x7) {
  RunQLinearGlobalAveragePool<uint8_t>(true, 1, 256, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x7x7x256) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 1, 256, 7, 7, 128, 7.0, 128, 21.0);
}

// tests for BatchSize > 1
TEST(QLinearGlobalAveragePool, Nhwc_3x256x8x8) {
  RunQLinearGlobalAveragePool<uint8_t>(true, 3, 256, 8, 8, 128, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x8x8x256) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 3, 256, 8, 8, 128, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x255x7x7) {
  RunQLinearGlobalAveragePool<uint8_t>(true, 3, 255, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x7x7x255) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 3, 255, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x255x8x8) {
  RunQLinearGlobalAveragePool<uint8_t>(true, 3, 255, 8, 8, 128, 1.0, 128, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x8x8x255) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 3, 255, 8, 8, 128, 1.0, 128, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x256x7x7) {
  RunQLinearGlobalAveragePool<uint8_t>(true, 3, 256, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x7x7x256) {
  RunQLinearGlobalAveragePool<uint8_t>(false, 3, 256, 7, 7, 128, 7.0, 128, 21.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x1x32x32_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 1, 1, 32, 32, 1, 1.0, -64, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x32x32x1_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 1, 1, 32, 32, 1, 1.0, 64, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x256x8x8_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 1, 256, 8, 8, -1, 1.0, -64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x8x8x256_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 1, 256, 8, 8, -1, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x255x7x7_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 1, 255, 7, 7, 64, 7.0, 1, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x7x7x255_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 1, 255, 7, 7, 64, 7.0, -1, 21.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x255x8x8_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 1, 255, 8, 8, -64, 1.0, 1, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x8x8x255_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 1, 255, 8, 8, -64, 1.0, -1, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_1x256x7x7_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 1, 256, 7, 7, -64, 7.0, 64, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_1x7x7x256_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 1, 256, 7, 7, 64, 7.0, -64, 21.0);
}

// tests for BatchSize > 1
TEST(QLinearGlobalAveragePool, Nhwc_3x256x8x8_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 3, 256, 8, 8, 1, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x8x8x256_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 3, 256, 8, 8, 1, 1.0, 64, 3.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x255x7x7_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 3, 255, 7, 7, 1, 7.0, -1, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x7x7x255_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 3, 255, 7, 7, 1, 7.0, -1, 21.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x255x8x8_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 3, 255, 8, 8, 1, 1.0, -1, 2.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x8x8x255_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 3, 255, 8, 8, -1, 1.0, 1, 2.0);
}

TEST(QLinearGlobalAveragePool, Nhwc_3x256x7x7_S8) {
  RunQLinearGlobalAveragePool<int8_t>(true, 3, 256, 7, 7, -1, 7.0, 1, 21.0);
}

TEST(QLinearGlobalAveragePool, Nchw_3x7x7x256_S8) {
  RunQLinearGlobalAveragePool<int8_t>(false, 3, 256, 7, 7, -1, 7.0, 1, 21.0);
}

}  // namespace test
}  // namespace onnxruntime
