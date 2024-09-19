// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#include "test/providers/cuda/nhwc/nhwc_cuda_helper.h"

namespace onnxruntime {
namespace test {

template <typename T>
struct BatchNormOp {
  std::vector<int64_t> input_dims;

  std::unique_ptr<CompareOpTester> get_test() {
    // create rand inputs
    RandomValueGenerator random{};

    auto test = std::make_unique<CompareOpTester>("BatchNormalization", 14);
    std::vector<T> input_data = random.Uniform<T>(input_dims, 0.0f, 0.3f);
    auto channels = input_dims[1];
    test->AddInput<T>("X", input_dims, input_data);

    std::vector<int64_t> bias_dims{channels};
    std::vector<T> bias_data = random.Uniform<T>(bias_dims, 0.2f, 1.0f);
    test->AddInput<T>("B", bias_dims, bias_data);
    // we simply gonna reuse the bias data here.
    test->AddInput<T>("scale", bias_dims, bias_data);

    std::vector<int64_t> mean{channels};
    std::vector<T> mean_data = random.Uniform<T>(mean, 0.7f, 0.8f);
    test->AddInput<T>("input_mean", bias_dims, bias_data);
    std::vector<int64_t> var{channels};
    std::vector<T> var_data = random.Uniform<T>(var, 0.0f, 0.1f);
    test->AddInput<T>("input_var", bias_dims, bias_data);

    std::vector<T> output_data = FillZeros<T>(input_dims);
    test->AddOutput<T>("Y", input_dims, output_data);
    return test;
  }
};

TYPED_TEST(CudaNhwcTypedTest, BatchNormNhwc) {
  auto op = BatchNormOp<TypeParam>{};
  op.input_dims = {4, 16, 64, 64};

  MAKE_PROVIDERS()
}

}  // namespace test
}  // namespace onnxruntime
