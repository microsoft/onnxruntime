// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "test/providers/cuda/nhwc/nhwc_cuda_helper.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct ConvTransposeOp {
  std::vector<int64_t> input_dims;
  std::vector<int64_t> kernel_shape;
  int64_t channels;
  int64_t group = 1;
  bool bias = false;
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> padding = {0, 0, 0, 0};
  std::vector<int64_t> output_padding = {0, 0, 0, 0};
  std::vector<int64_t> dilations = {1, 1};

  std::unique_ptr<CompareOpTester> get_test() {
    RandomValueGenerator random{123};  // use seed so output is deterministic to aid in debugging failures

    auto test = std::make_unique<CompareOpTester>("ConvTranspose", 14);
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 1.0f);

    // 1D or 2D input is supported
    const bool is_1D = input_dims.size() == 3;
    std::vector<int64_t> weight_dims{input_dims[1], channels / group, kernel_shape[0]};
    if (!is_1D) {
      weight_dims.push_back(kernel_shape[1]);
    }

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
    if (!output_padding.empty()) {
      test->AddAttribute("output_padding", output_padding);
    } else {
      output_padding = {0, 0, 0, 0};
    }

    // the test input is NCHW so calculate output based on that. conversion to/from NHWC is internal to execution.
    std::vector<int64_t> output_dims = {input_dims[0], channels};

    for (size_t i = 0, end = is_1D ? 1 : 2; i < end; ++i) {
      // formula from https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
      const size_t start_pad = i * 2;
      output_dims.push_back(
          strides[i] * (input_dims[i + 2] - 1) + output_padding[i] +
          ((kernel_shape[i] - 1) * dilations[i] + 1) - padding[start_pad] - padding[start_pad + 1]);
    }

    std::vector<T> output_data = FillZeros<T>(output_dims);

    test->AddOutput<T>("Y", output_dims, output_data);
    return test;
  }
};

TYPED_TEST(CudaNhwcTypedTest, ConvTransposeNhwcGroupNoBias) {
  auto op = ConvTransposeOp<TypeParam>{};
  op.input_dims = {8, 8, 32, 32};
  op.kernel_shape = {3, 3};
  op.channels = 16;
  op.group = 4;

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

TYPED_TEST(CudaNhwcTypedTest, ConvTransposeNhwcBias) {
  auto op = ConvTransposeOp<TypeParam>{};
  op.input_dims = {1, 8, 80, 80};
  op.kernel_shape = {5, 5};
  op.channels = 16;
  op.bias = true;

  if (HasCudaEnvironment(800)) {
    MAKE_PROVIDERS_EPS(1e-2)
  } else {
    MAKE_PROVIDERS_EPS_TYPE(TypeParam)
  }
}

TYPED_TEST(CudaNhwcTypedTest, ConvTransposeNhwcBias1D) {
  auto op = ConvTransposeOp<TypeParam>{};
  op.input_dims = {1, 8, 80};
  op.kernel_shape = {5};
  op.channels = 16;
  op.bias = true;
  op.padding = {0, 0};
  op.strides = {1};
  op.dilations = {1};
  op.output_padding = {};

  // test with adding fake W and H dimensions
  if (HasCudaEnvironment(800)) {
    MAKE_PROVIDERS_EPS_EXT(1e-2, true)   // add fake H dimension of 1 to convert to 2D
    MAKE_PROVIDERS_EPS_EXT(1e-2, false)  // add fake W dimension of 1 to convert to 2D
  } else {
    MAKE_PROVIDERS_EPS_TYPE_EXT(TypeParam, true)
    MAKE_PROVIDERS_EPS_TYPE_EXT(TypeParam, false)
  }
}

TYPED_TEST(CudaNhwcTypedTest, ConvTransposeNhwcPad) {
  auto op = ConvTransposeOp<TypeParam>{};
  op.input_dims = {1, 16, 8, 8};
  op.kernel_shape = {3, 3};
  op.channels = 32;
  op.padding = {2, 2, 2, 2};
  op.output_padding = {};

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

TYPED_TEST(CudaNhwcTypedTest, ConvTransposeNhwcOutPad) {
  auto op = ConvTransposeOp<TypeParam>{};
  op.input_dims = {1, 32, 8, 8};
  op.kernel_shape = {3, 3};
  op.channels = 32;
  op.strides = {2, 2};
  op.output_padding = {1, 1, 1, 1};

  MAKE_PROVIDERS_EPS_TYPE(TypeParam)
}

}  // namespace test
}  // namespace onnxruntime
