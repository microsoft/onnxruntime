// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/quantization_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace test {

struct DimIterator {
  DimIterator(const std::vector<int64_t>& dims) : dims_(dims) {
    size_ = std::accumulate(dims_.begin(), dims_.end(), 1LL, std::multiplies<int64_t>());
    restart();
  }

  void restart() {
    pos_.resize(dims_.size(), 0LL);
    index_ = 0LL;
  }

  bool has_next() { return index_ < size_; }

  // if has more data return current data ptr and iterator to next pos_
  // otherwise return -1
  int64_t next() {
    if (has_next()) {
      for (size_t i = dims_.size(); i > 0;) {
        i--;
        ++pos_[i];
        if (pos_[i] < dims_[i]) {
          break;
        }
        pos_[i] = 0;
      }
      return index_++;
    }
    return -1L;
  }

  const std::vector<int64_t> dims_;
  std::vector<int64_t> pos_;
  int64_t size_;
  int64_t index_;
};

template <typename T8Bits>
static void
CalculateAvgPoolNchw(
    T8Bits* x,
    const std::vector<int64_t> x_dims,
    const quantization::Params<T8Bits>& x_params,
    T8Bits* y,
    const std::vector<int64_t> y_dims,
    const quantization::Params<T8Bits>& y_params,
    const std::vector<int64_t> kernel_shape,
    const std::vector<int64_t> strides,
    const std::vector<int64_t> pads,
    const int64_t count_include_pad) {
  int64_t batch = y_dims[0];
  int64_t channel = y_dims[1];

  std::vector<int64_t> y_img_dims(y_dims.begin() + 2, y_dims.end());
  std::vector<int64_t> x_img_dims(x_dims.begin() + 2, x_dims.end());
  std::vector<int64_t> x_img_strides(x_img_dims.size(), 1LL);
  for (size_t i = x_img_dims.size() - 1; i > 0;) {
    i--;
    x_img_strides[i] = x_img_strides[i + 1] * x_img_dims[i + 1];
  }

  int64_t y_step = std::accumulate(y_img_dims.begin(), y_img_dims.end(), 1LL, std::multiplies<int64_t>());
  int64_t x_step = std::accumulate(x_img_dims.begin(), x_img_dims.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t c = 0; c < channel; ++c) {
      T8Bits* ybc = y + (b * channel + c) * y_step;
      T8Bits* xbc = x + (b * channel + c) * x_step;

      DimIterator yit(y_img_dims);
      while (yit.has_next()) {
        std::vector<int64_t> kernel_topleft(y_img_dims.size(), 0);
        for (size_t i = 0; i < y_img_dims.size(); ++i) {
          kernel_topleft[i] = yit.pos_[i] * strides[i];
        }

        float y_value_sum = 0.0f;
        int count = 0;
        for (DimIterator kit(kernel_shape); kit.has_next(); kit.next()) {
          int64_t kernel_offset = 0;
          for (size_t i = 0; kernel_offset >= 0 && i < kernel_shape.size(); ++i) {
            int64_t x_real_dim = kernel_topleft[i] + kit.pos_[i] - pads[i];
            if (x_real_dim >= 0 && x_real_dim < x_img_dims[i]) {
              kernel_offset += x_real_dim * x_img_strides[i];
            } else {
              kernel_offset = -1LL;  // padding element
            }
          }
          if (kernel_offset >= 0) {
            y_value_sum += quantization::Dequantize(xbc[kernel_offset], x_params);
            ++count;
          } else {
            count += count_include_pad ? 1 : 0;
          }
        }
        auto y_offset = yit.next();
        auto y_u8 = QuantizeTestValue<T8Bits>(y_value_sum / count, y_params);
        ybc[y_offset] = y_u8;
      }
    }
  }
}

template <typename T8Bits = uint8_t>
void RunQLinearAveragePoolNchw(
    const std::vector<int64_t> x_dims,
    const std::vector<int64_t> y_dims,
    const std::vector<int64_t> kernel_shape,
    const std::vector<int64_t> strides,
    const std::vector<int64_t> pads,
    const int64_t count_include_pad = 0) {
  auto run_test = [&](bool only_x_not_initializer, bool x_y_same_zero_point) {
    float x_scale = 1.0f / 255.0f;
    T8Bits x_zero_point = (std::numeric_limits<T8Bits>::lowest() + std::numeric_limits<T8Bits>::max() - 5) / 2;
    quantization::Params<T8Bits> x_params(x_scale, x_zero_point);
    RandomValueGenerator random{};
    std::vector<float> x_data_fp32 = random.Uniform<float>(x_dims, -0.5f, 0.5f);
    std::vector<T8Bits> x_data = QuantizeTestVector<T8Bits>(x_data_fp32, x_params);

    float y_scale = 1.0f / 255.0f;
    T8Bits y_zero_point_not_same = (std::numeric_limits<T8Bits>::lowest() + std::numeric_limits<T8Bits>::max() + 10) / 2;
    T8Bits y_zero_point = x_y_same_zero_point ? x_params.zero_point : y_zero_point_not_same;
    const quantization::Params<T8Bits> y_params(y_scale, y_zero_point);
    int64_t y_size = std::accumulate(y_dims.begin(), y_dims.end(), 1LL, std::multiplies<int64_t>());
    std::vector<T8Bits> y_data(y_size);
    CalculateAvgPoolNchw(
        x_data.data(), x_dims, x_params,
        y_data.data(), y_dims, y_params,
        kernel_shape, strides, pads, count_include_pad);

    OpTester test("QLinearAveragePool", 1, onnxruntime::kMSDomain);

    test.AddAttribute("auto_pad", "");
    test.AddAttribute("strides", strides);
    test.AddAttribute("pads", pads);
    test.AddAttribute("kernel_shape", kernel_shape);
    test.AddAttribute("count_include_pad", count_include_pad);

    test.AddInput<T8Bits>("X", x_dims, x_data);
    test.AddInput<float>("x_scale", {}, {x_scale}, only_x_not_initializer);
    test.AddInput<T8Bits>("x_zero_point", {}, {x_params.zero_point}, only_x_not_initializer);
    test.AddInput<float>("y_scale", {}, {y_scale}, only_x_not_initializer);
    test.AddInput<T8Bits>("y_zero_point", {}, {y_params.zero_point}, only_x_not_initializer);
    test.AddOutput<T8Bits>("Y", y_dims, y_data);

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
  };

  run_test(false /* only_x_not_initializer */, false /* x_y_same_zero_point */);

  // NNAPI will require all inputs except X to be initializers
  // Also NNAPI average pool will require output has the same scale and zero point as input
  run_test(true /* only_x_not_initializer */, false /* x_y_same_zero_point */);
  run_test(true /* only_x_not_initializer */, true /* x_y_same_zero_point */);
}

static std::vector<int64_t> dims_to_nhwc(const std::vector<int64_t>& nchw) {
  std::vector<int64_t> nhwc(nchw);
  nhwc.erase(nhwc.begin() + 1);
  nhwc.push_back(nchw[1]);
  return nhwc;
}

template <typename T8Bits>
static std::vector<T8Bits> transpose_to_nhwc(const std::vector<T8Bits>& nchw_data, const std::vector<int64_t>& nchw_dims) {
  std::vector<T8Bits> nhwc_data(nchw_data.size());

  auto batch_count = nchw_dims[0];
  auto channels = nchw_dims[1];
  int64_t image_size = std::accumulate(nchw_dims.begin() + 2, nchw_dims.end(), 1LL, std::multiplies<int64_t>());
  for (int64_t b = 0; b < batch_count; b++) {
    const T8Bits* nchw_image = nchw_data.data() + (b * channels * image_size);
    T8Bits* nhwc_image = nhwc_data.data() + (b * channels * image_size);
    for (int64_t img_index = 0; img_index < image_size; ++img_index) {
      for (int64_t c = 0; c < channels; c++) {
        *nhwc_image++ = nchw_image[c * image_size + img_index];
      }
    }
  }

  return nhwc_data;
}

template <typename T8Bits = uint8_t>
void RunQLinearAveragePoolNhwc(
    const std::vector<int64_t> x_dims,
    const std::vector<int64_t> y_dims,
    const std::vector<int64_t> kernel_shape,
    const std::vector<int64_t> strides,
    const std::vector<int64_t> pads,
    const int64_t count_include_pad = 0) {
  float x_scale = 1.0f / 255.0f;
  T8Bits x_zero_point = (std::numeric_limits<T8Bits>::lowest() + std::numeric_limits<T8Bits>::max() - 5) / 2;
  const quantization::Params<T8Bits> x_params(x_scale, x_zero_point);
  RandomValueGenerator random{};
  std::vector<float> x_data_fp32 = random.Uniform<float>(x_dims, -0.5f, 0.5f);
  std::vector<T8Bits> x_data = QuantizeTestVector<T8Bits>(x_data_fp32, x_params);

  float y_scale = 1.0f / 255.0f;
  T8Bits y_zero_point = (std::numeric_limits<T8Bits>::lowest() + std::numeric_limits<T8Bits>::max() + 10) / 2;
  const quantization::Params<T8Bits> y_params(y_scale, y_zero_point);
  int64_t y_size = std::accumulate(y_dims.begin(), y_dims.end(), 1LL, std::multiplies<int64_t>());
  std::vector<T8Bits> y_data(y_size);
  CalculateAvgPoolNchw(
      x_data.data(), x_dims, x_params,
      y_data.data(), y_dims, y_params,
      kernel_shape, strides, pads, count_include_pad);

  // transpose the result
  std::vector<T8Bits> y_data_nhwc = transpose_to_nhwc(y_data, y_dims);
  std::vector<T8Bits> x_data_nhwc = transpose_to_nhwc(x_data, x_dims);
  auto x_dims_nhwc = dims_to_nhwc(x_dims);
  auto y_dims_nhwc = dims_to_nhwc(y_dims);

  OpTester test("QLinearAveragePool", 1, onnxruntime::kMSDomain);

  test.AddAttribute("auto_pad", "");
  test.AddAttribute("strides", strides);
  test.AddAttribute("pads", pads);
  test.AddAttribute("kernel_shape", kernel_shape);
  test.AddAttribute("count_include_pad", count_include_pad);
  test.AddAttribute("channels_last", (int64_t)1LL);

  test.AddInput<T8Bits>("X", x_dims_nhwc, x_data_nhwc);
  test.AddInput<float>("x_scale", {}, {x_scale});
  test.AddInput<T8Bits>("x_zero_point", {}, {x_params.zero_point});
  test.AddInput<float>("y_scale", {}, {y_scale});
  test.AddInput<T8Bits>("y_zero_point", {}, {y_params.zero_point});
  test.AddOutput<T8Bits>("Y", y_dims_nhwc, y_data_nhwc);

  auto q8checker = [&](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    const OrtValue& ort_value = fetches[0];

    auto y_shape = TensorShape(y_dims_nhwc);
    const Tensor& output_tensor = ort_value.Get<Tensor>();
    ORT_ENFORCE(y_shape == output_tensor.Shape(),
                "Expected output shape [" + y_shape.ToString() + "] did not match run output shape [" +
                    output_tensor.Shape().ToString() + "] for Y @" + provider_type);
    auto* output = output_tensor.Data<T8Bits>();
    auto size = static_cast<int>(output_tensor.Shape().Size());
    for (int i = 0; i < size; ++i) {
      int diff = abs(y_data_nhwc[i] - output[i]);
      EXPECT_LE(diff, 1) << "i:" << i << " expected:" << y_data_nhwc[i] << " " << (int)y_data_nhwc[i]
                         << ", got:" << output[i] << " " << (int)output[i] << ", provider_type: " << provider_type;
    }
  };
  test.SetCustomOutputVerifier(q8checker);

  static std::unordered_set<std::string> excluded_providers = {kNnapiExecutionProvider};

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
}

TEST(QLinearPoolTest, AveragePool1D_ExcludePadPixel) {
  RunQLinearAveragePoolNchw(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      0);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool1D_IncludePadPixel) {
  RunQLinearAveragePoolNchw(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      1);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_ExcludePadPixel) {
  RunQLinearAveragePoolNchw(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      0);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_IncludePadPixel) {
  RunQLinearAveragePoolNchw(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_MultiChannel) {
  RunQLinearAveragePoolNchw(
      {1, 3, 5, 7},  // x shape
      {1, 3, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_ExcludePadPixel) {
  RunQLinearAveragePoolNchw(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      0);                  // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_IncludePadPixel) {
  RunQLinearAveragePoolNchw(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      1);                  // count_include_pad
}

/*************************************************
 * Channels last test
 **************************************************/
TEST(QLinearPoolTest, AveragePool1D_ExcludePadPixel_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      0);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool1D_IncludePadPixel_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      1);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_ExcludePadPixel_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      0);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_IncludePadPixel_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_MultiChannel_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 3, 5, 7},  // x shape
      {1, 3, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_ExcludePadPixel_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      0);                  // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_IncludePadPixel_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      1);                  // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_BigImage) {
  RunQLinearAveragePoolNchw(
      {1, 1, 32, 64},  // x shape
      {1, 1, 32, 64},  // expected y shape
      {3, 3},          // kernel shape
      {1, 1},          // strides
      {1, 1, 1, 1},    // pads
      1);              // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_BigImage_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 1, 32, 64},  // x shape
      {1, 1, 32, 64},  // expected y shape
      {3, 3},          // kernel shape
      {1, 1},          // strides
      {1, 1, 1, 1},    // pads
      1);              // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_Global) {
  RunQLinearAveragePoolNchw(
      {1, 2, 32, 16},  // x shape
      {1, 2, 1, 1},    // expected y shape
      {32, 16},        // kernel shape
      {1, 1},          // strides
      {0, 0, 0, 0},    // pads
      1);              // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_Global_nhwc) {
  RunQLinearAveragePoolNhwc(
      {1, 2, 32, 16},  // x shape
      {1, 2, 1, 1},    // expected y shape
      {32, 16},        // kernel shape
      {1, 1},          // strides
      {0, 0, 0, 0},    // pads
      1);              // count_include_pad
}

TEST(QLinearPoolTest, AveragePool1D_ExcludePadPixel_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      0);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool1D_IncludePadPixel_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      1);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_ExcludePadPixel_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      0);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_IncludePadPixel_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_MultiChannel_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 3, 5, 7},  // x shape
      {1, 3, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_ExcludePadPixel_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      0);                  // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_IncludePadPixel_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      1);                  // count_include_pad
}

/*************************************************
 * Channels last test
 **************************************************/
TEST(QLinearPoolTest, AveragePool1D_ExcludePadPixel_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      0);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool1D_IncludePadPixel_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      1);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_ExcludePadPixel_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      0);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_IncludePadPixel_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_MultiChannel_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 3, 5, 7},  // x shape
      {1, 3, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_ExcludePadPixel_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      0);                  // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_IncludePadPixel_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      1);                  // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_BigImage_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 1, 32, 64},  // x shape
      {1, 1, 32, 64},  // expected y shape
      {3, 3},          // kernel shape
      {1, 1},          // strides
      {1, 1, 1, 1},    // pads
      1);              // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_BigImage_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 1, 32, 64},  // x shape
      {1, 1, 32, 64},  // expected y shape
      {3, 3},          // kernel shape
      {1, 1},          // strides
      {1, 1, 1, 1},    // pads
      1);              // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_Global_S8) {
  RunQLinearAveragePoolNchw<int8_t>(
      {1, 2, 32, 16},  // x shape
      {1, 2, 1, 1},    // expected y shape
      {32, 16},        // kernel shape
      {1, 1},          // strides
      {0, 0, 0, 0},    // pads
      1);              // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_Global_nhwc_S8) {
  RunQLinearAveragePoolNhwc<int8_t>(
      {1, 2, 32, 16},  // x shape
      {1, 2, 1, 1},    // expected y shape
      {32, 16},        // kernel shape
      {1, 1},          // strides
      {0, 0, 0, 0},    // pads
      1);              // count_include_pad
}

}  // namespace test
}  // namespace onnxruntime
