// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "test/providers/cuda/nhwc/nhwc_cuda_helper.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct PoolOp {
  std::string pooling_type;
  std::vector<int64_t> input_dims;
  std::vector<int64_t> kernel_shape;
  int64_t channels;
  int64_t group = 1;
  std::vector<int64_t> strides = {1, 1};
  std::vector<int64_t> padding = {0, 0, 0, 0};

  std::unique_ptr<CompareOpTester> get_test() {
    RandomValueGenerator random{};

    auto test = std::make_unique<CompareOpTester>(pooling_type.c_str(), 14);
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 0.3f);

    test->AddInput<T>("X", input_dims, input_data);

    test->AddAttribute("kernel_shape", kernel_shape);
    test->AddAttribute("strides", strides);
    test->AddAttribute("pads", padding);

    std::vector<int64_t> output_dims = {
        input_dims[0], channels,
        (input_dims[2] - (kernel_shape[0] - 1) + padding[1] + padding[0] - 1) / strides[0] + 1,
        (input_dims[3] - (kernel_shape[1] - 1) + padding[3] + padding[2] - 1) / strides[1] + 1};
    std::vector<T> output_data = FillZeros<T>(output_dims);

    test->AddOutput<T>("Y", output_dims, output_data);
    return test;
  }
};

TYPED_TEST(CudaNhwcTypedTest, AveragePoolNhwc) {
  auto op = PoolOp<TypeParam>{};
  op.pooling_type = "AveragePool";
  op.input_dims = {1, 16, 64, 64};
  op.kernel_shape = {3, 3};
  op.channels = 16;

  MAKE_PROVIDERS()
}

TYPED_TEST(CudaNhwcTypedTest, MaxPoolNhwc) {
  auto op = PoolOp<TypeParam>{};
  op.pooling_type = "MaxPool";
  op.input_dims = {1, 16, 64, 64};
  op.kernel_shape = {3, 3};
  op.channels = 16;
  MAKE_PROVIDERS()
}

TYPED_TEST(CudaNhwcTypedTest, GlobalMaxPoolNhwc) {
  RandomValueGenerator random{};
  auto test = std::make_unique<CompareOpTester>("GlobalMaxPool", 14);
  const std::vector<int64_t> input_dims = {4, 16, 4, 8};
  std::vector<TypeParam> input_data = random.Uniform<TypeParam>(input_dims, 0.5f, 1.3f);
  test->AddInput<TypeParam>("X", input_dims, input_data);

  std::vector<int64_t> output_dims = {input_dims[0], input_dims[1], 1, 1};
  std::vector<TypeParam> output_data = FillZeros<TypeParam>(output_dims);
  test->AddOutput<TypeParam>("Y", output_dims, output_data);

  std::vector<std::shared_ptr<IExecutionProvider>> execution_providers;
  OrtCUDAProviderOptionsV2 nhwc{};
  nhwc.prefer_nhwc = true;
  execution_providers.push_back(CudaExecutionProviderWithOptions(&nhwc));

  double error_tolerance = 1e-3;
  OrtCUDAProviderOptionsV2 nchw{};
  nchw.prefer_nhwc = false;
  auto source_ep = CudaExecutionProviderWithOptions(&nchw);
  test->CompareEPs(std::move(source_ep), execution_providers, error_tolerance);
}

TYPED_TEST(CudaNhwcTypedTest, AveragePoolNhwcPad) {
  auto op = PoolOp<TypeParam>{};
  op.pooling_type = "AveragePool";
  op.input_dims = {1, 16, 64, 64};
  op.kernel_shape = {3, 3};
  op.channels = 16;
  op.padding = {2, 2, 2, 2};

  MAKE_PROVIDERS()
}

}  // namespace test
}  // namespace onnxruntime
