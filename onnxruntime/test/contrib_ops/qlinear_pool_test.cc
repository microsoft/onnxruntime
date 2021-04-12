// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace test {

static inline float dequantize_u8(uint8_t x, float x_scale, uint8_t x_zero_point) {
  return x_scale * (static_cast<int>(x) - x_zero_point);
}

static inline uint8_t quantize_u8(float y, float y_scale, uint8_t y_zero_point) {
  return static_cast<uint8_t>(std::max(0.0f, std::min(std::nearbyintf(y / y_scale + y_zero_point), 255.0f)));
}

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

static void
CalculateAvgPoolNchwU8(
    uint8_t* x,
    const std::vector<int64_t> x_dims,
    float x_scale,
    int x_zero_point,
    uint8_t* y,
    const std::vector<int64_t> y_dims,
    float y_scale,
    int y_zero_point,
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
      uint8_t* ybc = y + (b * channel + c) * y_step;
      uint8_t* xbc = x + (b * channel + c) * x_step;

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
            y_value_sum += dequantize_u8(xbc[kernel_offset], x_scale, static_cast<uint8_t>(x_zero_point));
            ++count;
          } else {
            count += count_include_pad ? 1 : 0;
          }
        }
        auto y_offset = yit.next();
        auto y_u8 = quantize_u8(y_value_sum / count, y_scale, static_cast<uint8_t>(y_zero_point));
        ybc[y_offset] = y_u8;
      }
    }
  }
}

void RunQLinearAveragePoolNchwU8(
    const std::vector<int64_t> x_dims,
    const std::vector<int64_t> y_dims,
    const std::vector<int64_t> kernel_shape,
    const std::vector<int64_t> strides,
    const std::vector<int64_t> pads,
    const int64_t count_include_pad = 0) {
  auto run_test = [&](bool only_x_not_initializer, bool x_y_same_zero_point) {
    float x_scale = 1.0f / 255.0f;
    uint8_t x_zero_point = 128;
    RandomValueGenerator random{};
    std::vector<float> x_data_fp32 = random.Uniform<float>(x_dims, -0.5f, 0.5f);
    std::vector<uint8_t> x_data(x_data_fp32.size());
    for (size_t i = 0; i < x_data.size(); ++i) {
      x_data[i] = quantize_u8(x_data_fp32[i], x_scale, x_zero_point);
    }

    float y_scale = 1.0f / 255.0f;
    uint8_t y_zero_point = x_y_same_zero_point ? x_zero_point : 100;
    int64_t y_size = std::accumulate(y_dims.begin(), y_dims.end(), 1LL, std::multiplies<int64_t>());
    std::vector<uint8_t> y_data(y_size);
    CalculateAvgPoolNchwU8(
        x_data.data(), x_dims, x_scale, x_zero_point,
        y_data.data(), y_dims, y_scale, y_zero_point,
        kernel_shape, strides, pads, count_include_pad);

    OpTester test("QLinearAveragePool", 1, onnxruntime::kMSDomain);

    test.AddAttribute("auto_pad", "");
    test.AddAttribute("strides", strides);
    test.AddAttribute("pads", pads);
    test.AddAttribute("kernel_shape", kernel_shape);
    test.AddAttribute("count_include_pad", count_include_pad);

    test.AddInput<uint8_t>("X", x_dims, x_data);
    test.AddInput<float>("x_scale", {}, {x_scale}, only_x_not_initializer);
    test.AddInput<uint8_t>("x_zero_point", {}, {x_zero_point}, only_x_not_initializer);
    test.AddInput<float>("y_scale", {}, {y_scale}, only_x_not_initializer);
    test.AddInput<uint8_t>("y_zero_point", {}, {y_zero_point}, only_x_not_initializer);
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
  };

  run_test(false /* only_x_not_initializer */, false /* x_y_same_zero_point */);

  // NNAPI will require all inputs except X to be initializers
  // Also NNAPI average pool will require output has the same scale and zero point as input
  run_test(true /* only_x_not_initializer */, false /* x_y_same_zero_point */);
  run_test(true /* only_x_not_initializer */, true /* x_y_same_zero_point */);
}

TEST(QLinearPoolTest, AveragePool1D_ExcludePadPixel) {
  RunQLinearAveragePoolNchwU8(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      0);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool1D_IncludePadPixel) {
  RunQLinearAveragePoolNchwU8(
      {1, 1, 5},  // x shape
      {1, 1, 6},  // expected y shape
      {3},        // kernel shape
      {1},        // strides
      {1, 2},     // pads
      1);         // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_ExcludePadPixel) {
  RunQLinearAveragePoolNchwU8(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      0);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool2D_IncludePadPixel) {
  RunQLinearAveragePoolNchwU8(
      {1, 1, 5, 7},  // x shape
      {1, 1, 6, 4},  // expected y shape
      {3, 4},        // kernel shape
      {1, 2},        // strides
      {1, 3, 2, 1},  // pads
      1);            // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_ExcludePadPixel) {
  RunQLinearAveragePoolNchwU8(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      0);                  // count_include_pad
}

TEST(QLinearPoolTest, AveragePool3D_IncludePadPixel) {
  RunQLinearAveragePoolNchwU8(
      {1, 1, 5, 7, 9},     // x shape
      {1, 1, 6, 4, 3},     // expected y shape
      {3, 4, 5},           // kernel shape
      {1, 2, 3},           // strides
      {1, 3, 2, 2, 1, 2},  // pads
      1);                  // count_include_pad
}

}  // namespace test
}  // namespace onnxruntime
