// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "test/providers/cuda/nhwc/nhwc_cuda_helper.h"

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

    auto test = std::make_unique<CompareOpTester>("Conv", 11);  // internal NHWC domain starts at opset 11
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 1.0f);

    std::vector<int64_t> weight_dims{channels, input_dims[1] / group, kernel_shape[0], kernel_shape[1]};
    std::vector<T> weight_data = random.Uniform<T>(weight_dims, -0.4f, 0.4f);

    test->AddInput<T>("X", input_dims, input_data);
    test->AddInput<T>("W", weight_dims, weight_data, true);
    if (bias) {
      std::vector<int64_t> bias_dims{channels};
      std::vector<T> bias_data = random.Uniform<T>(bias_dims, 0.2f, 0.4f);
      test->AddInput<T>("B", bias_dims, bias_data, true);
    }
    test->AddAttribute("group", group);
    test->AddAttribute("kernel_shape", kernel_shape);
    test->AddAttribute("strides", strides);
    test->AddAttribute("dilations", dilations);
    test->AddAttribute("pads", padding);

    std::vector<int64_t> output_dims = {
        input_dims[0], channels,
        ComputeOutputShape(input_dims[2], strides[0], kernel_shape[0], dilations[0], padding[0], padding[1]),
        ComputeOutputShape(input_dims[3], strides[1], kernel_shape[1], dilations[1], padding[2], padding[3])};
    std::vector<T> output_data = FillZeros<T>(output_dims);

    test->AddOutput<T>("Y", output_dims, output_data);
    return test;
  }
};

TYPED_TEST(CudaNhwcTypedTest, ConvNhwcBias) {
  auto op = ConvOp<TypeParam>{.input_dims = {1, 16, 64, 64}, .kernel_shape = {3, 3}, .channels = 16, .bias = true};

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

TYPED_TEST(CudaNhwcTypedTest, ConvNhwcGroupNoBias) {
  auto op = ConvOp<TypeParam>{.input_dims = {1, 16, 64, 64}, .kernel_shape = {3, 3}, .channels = 16, .group = 4};

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

TYPED_TEST(CudaNhwcTypedTest, ConvNhwcPadding) {
  auto op =
      ConvOp<TypeParam>{.input_dims = {2, 4, 64, 64}, .kernel_shape = {3, 3}, .channels = 4, .padding = {4, 4, 4, 4}};

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

}  // namespace test
}  // namespace onnxruntime
