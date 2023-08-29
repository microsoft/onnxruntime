// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "nhwc_cuda_helper.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct ConvOp {
  const std::vector<int64_t> input_dims;
  const std::vector<int64_t> kernel_shape;
  int64_t channels;
  int64_t group = 1;
  bool bias = false;
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> padding = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1};

  std::unique_ptr<CompareOpTester> get_test() {
    RandomValueGenerator random{};

    auto test = std::make_unique<CompareOpTester>("Conv", 7);
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 0.3f);

    std::vector<int64_t> weight_dims{channels, input_dims[1] / group, kernel_shape[0], kernel_shape[1]};
    std::vector<T> weight_data = random.Uniform<T>(weight_dims, 0.0f, 0.3f);

    test->AddInput<T>("X", input_dims, input_data);
    test->AddInput<T>("W", weight_dims, weight_data, true);
    if (bias) {
      std::vector<int64_t> bias_dims{channels};
      std::vector<T> bias_data = random.Uniform<T>(bias_dims, 0.0f, 0.3f);
      test->AddInput<T>("B", bias_dims, bias_data, true);
    }
    test->AddAttribute("group", group);
    test->AddAttribute("kernel_shape", kernel_shape);
    test->AddAttribute("strides", strides);
    test->AddAttribute("dilations", dilations);
    test->AddAttribute("pads", padding);

    std::vector<int64_t> output_dims = {
        input_dims[0],
        channels,
        ComputeOutputShape(input_dims[2], strides[0], kernel_shape[0], dilations[0], padding[0], padding[1]),
        ComputeOutputShape(input_dims[3], strides[1], kernel_shape[1], dilations[1], padding[2], padding[3])};
    std::vector<T> output_data = FillZeros<T>(output_dims);

    test->AddOutput<T>("Y", output_dims, output_data);
    return test;
  }
};

template <typename T>
struct ConvTransposeOp {
  const std::vector<int64_t> input_dims;
  const std::vector<int64_t> kernel_shape;
  int64_t channels;
  int64_t group = 1;
  bool bias = false;
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> padding = {0, 0, 0, 0};
  std::vector<int64_t> output_padding = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1};

  std::unique_ptr<CompareOpTester> get_test() {
    RandomValueGenerator random{};

    auto test = std::make_unique<CompareOpTester>("ConvTranspose", 14);
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 0.3f);

    std::vector<int64_t> weight_dims{input_dims[1], channels / group, kernel_shape[0], kernel_shape[1]};
    std::vector<T> weight_data = random.Uniform<T>(weight_dims, 0.2f, 0.5f);

    test->AddInput<T>("X", input_dims, input_data);
    test->AddInput<T>("W", weight_dims, weight_data, true);
    if (bias) {
      std::vector<int64_t> bias_dims{channels};
      std::vector<T> bias_data = random.Uniform<T>(bias_dims, 0.0f, 0.4f);
      test->AddInput<T>("B", bias_dims, bias_data, true);
    }
    test->AddAttribute("group", group);
    test->AddAttribute("kernel_shape", kernel_shape);
    test->AddAttribute("strides", strides);
    test->AddAttribute("dilations", dilations);
    test->AddAttribute("pads", padding);
    if (!output_padding.empty()) {
      test->AddAttribute("output_padding", output_padding);
    }

    std::vector<int64_t> output_dims = {
        input_dims[0],
        channels,
        (kernel_shape[1] - 1) * dilations[1] + (input_dims[2] - 1) * strides[1] - (padding[1] + padding[0]) + 1,
        (kernel_shape[0] - 1) * dilations[0] + (input_dims[3] - 1) * strides[0] - (padding[3] + padding[2]) + 1};
    std::vector<T> output_data = FillZeros<T>(output_dims);

    test->AddOutput<T>("Y", output_dims, output_data);
    return test;
  }
};

TEST(CudaNhwcTest, ConvNhwcBias) {
  {
    auto op = ConvOp<float>{
        .input_dims = {1, 16, 64, 64},
        .kernel_shape = {3, 3},
        .channels = 16,
        .bias = true};

    MAKE_PROVIDERS()
  }
  {
    auto op = ConvOp<MLFloat16>{
        .input_dims = {1, 16, 64, 64},
        .kernel_shape = {3, 3},
        .channels = 16,
        .bias = true};

    MAKE_PROVIDERS_EPS(1e-2)
  }
}

TEST(CudaNhwcTest, ConvNhwcGroupNoBias) {
  auto op = ConvOp<float>{
      .input_dims = {1, 16, 64, 64},
      .kernel_shape = {3, 3},
      .channels = 16,
      .group = 4};

  MAKE_PROVIDERS()
}

TEST(CudaNhwcTest, ConvNhwcPadding) {
  auto op = ConvOp<float>{
      .input_dims = {2, 4, 64, 64},
      .kernel_shape = {3, 3},
      .channels = 4,
      .padding = {4, 4, 4, 4}};

  MAKE_PROVIDERS()
}

TEST(CudaNhwcTest, ConvTransposeNhwcGroupNoBias) {
  {
    auto op = ConvTransposeOp<float>{
        .input_dims = {8, 8, 32, 32},
        .kernel_shape = {3, 3},
        .channels = 16,
        .group = 4};

    MAKE_PROVIDERS()
  }
  {
    auto op = ConvTransposeOp<MLFloat16>{
        .input_dims = {8, 8, 32, 32},
        .kernel_shape = {3, 3},
        .channels = 16,
        .group = 4};

    MAKE_PROVIDERS()
  }
}

TEST(CudaNhwcTest, ConvTransposeNhwcBias) {
  {
    auto op = ConvTransposeOp<float>{
        .input_dims = {1, 8, 80, 80},
        .kernel_shape = {5, 5},
        .channels = 16,
        .bias = true};

    MAKE_PROVIDERS()
  }
  {
    auto op = ConvTransposeOp<MLFloat16>{
        .input_dims = {1, 8, 80, 80},
        .kernel_shape = {5, 5},
        .channels = 16,
        .bias = true};

    MAKE_PROVIDERS()
  }
}

TEST(CudaNhwcTest, ConvTransposeNhwcPad) {
  {
    auto op = ConvTransposeOp<float>{
        .input_dims = {1, 16, 8, 8},
        .kernel_shape = {3, 3},
        .channels = 32,
        .padding = {2, 2, 2, 2},
        .output_padding = {}};

    MAKE_PROVIDERS()
  }
  {
    auto op = ConvTransposeOp<MLFloat16>{
        .input_dims = {1, 16, 8, 8},
        .kernel_shape = {3, 3},
        .channels = 32,
        .padding = {2, 2, 2, 2},
        .output_padding = {}};

    MAKE_PROVIDERS()
  }
}

TEST(CudaNhwcTest, ConvTransposeNhwcOutPad) {
  auto op = ConvTransposeOp<float>{
      .input_dims = {1, 32, 8, 8},
      .kernel_shape = {3, 3},
      .channels = 32,
      .strides = {2, 2},
      .output_padding = {1, 1, 1, 1}};

  MAKE_PROVIDERS()
}

}  // namespace test
}  // namespace onnxruntime
