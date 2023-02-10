// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

#ifdef USE_CUDA
void CompareFakeQuantKernels(const std::vector<int64_t>& tensor_dim,
                             double per_sample_tolerance = 2e-4,
                             double relative_per_sample_tolerance = 2e-4) {
  CompareOpTester test("FakeQuant", 1, onnxruntime::kMSDomain);

  test.AddAttribute<int64_t>("quant_min", 0);
  test.AddAttribute<int64_t>("quant_max", 7500);

  // Create rand inputs for the input tensor, scale and zero point
  RandomValueGenerator random{};
  std::vector<float> input_data = random.Uniform<float>(tensor_dim, -1000.0f, 1000.0f);
  test.AddInput<float>("input_tensor", tensor_dim, input_data);
  std::vector<float> scale = random.Uniform<float>(std::vector<int64_t>({1}), 0.04f, 0.1f);
  test.AddInput<float>("scale", {1}, scale);
  std::vector<float> zero_point = random.Uniform<float>(std::vector<int64_t>({1}), 0.f, 255.0f);
  test.AddInput<float>("zero_point", {1}, std::vector<float>({std::nearbyint(zero_point.front())}));

  // Create output tensors
  std::vector<float> fake_quantized_data = FillZeros<float>(tensor_dim);
  test.AddOutput<float>("fake_quantized_tensor", tensor_dim, fake_quantized_data);
  std::unique_ptr<bool[]> quantization_mask = std::make_unique<bool[]>(detail::SizeFromDims(tensor_dim));
  test.AddOutput<bool>("quantization_mask", tensor_dim, quantization_mask.get(), detail::SizeFromDims(tensor_dim));

  test.CompareWithCPU(kCudaExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

#endif

}  // namespace

TEST(FakeQuantTest, FakeQuantComputation) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());
#ifdef USE_CUDA
  providers.emplace_back(DefaultCudaExecutionProvider());
#endif

  OpTester test("FakeQuant", 1, onnxruntime::kMSDomain);

  test.AddAttribute<int64_t>("quant_min", 0);
  test.AddAttribute<int64_t>("quant_max", 255);

  test.AddInput<float>("input_tensor", {10}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
  test.AddInput<float>("scale", {1}, {0.075f});
  test.AddInput<float>("zero_point", {1}, {128.0f});
  // quantized values = nearby_int(value / scale + zero_point)
  //                  = {13.33+128, 26.66+128, 40.00+128, 53.33+128, 66.66+128, ...}
  //                  = {141.33, 154.66, 168.00, 171.33, 184.66, ...}
  //                  = {141, 155, 168, 181, 195, 208, 221, 235, 248, 261}
  // de-quantized values = (clamp(value) - zero_point) * scale
  //                     = {13*0.075, 27*0.075, 40*0.075, 53*0.075, 67*0.075, ..., 120*0.075, (255-128)*0.075}
  //                     = {0.975, 2.025, 3.0, 3.975, 5.025, 6.0, 6.975, 8.025, 9.0, 9.525}

  test.AddOutput<float>(
      "fake_quantized_tensor", {10}, {0.975f, 2.025f, 3.0f, 3.975f, 5.025f, 6.0f, 6.975f, 8.025f, 9.0f, 9.525f});
  test.AddOutput<bool>("quantization_mask", {10}, {true, true, true, true, true, true, true, true, true, false});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

#ifdef USE_CUDA
TEST(CudaKernelTest, FakeQuant) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    CompareFakeQuantKernels(test_dim, false);
  }
}
#endif

class FakeQuantGradParameterizedTest : public ::testing::TestWithParam<TensorShape> {
};

TEST_P(FakeQuantGradParameterizedTest, FakeQuantGradComputation) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());
#ifdef USE_CUDA
  providers.emplace_back(DefaultCudaExecutionProvider());
#endif
  OpTester test("FakeQuantGrad", 1, kMSDomain, true);

  auto x_shape = GetParam();
  RandomValueGenerator random{};

  // Randomly generate the gradient w.r.t. the output Y.
  std::vector<float> dY_data = random.Uniform<float>(x_shape.GetDims(), -1000.0f, 1000.0f);

  // Randomly generate the mask data
  std::unique_ptr<bool[]> mask_data = [&random, &x_shape]() {
    auto data_int = random.Uniform<int>(x_shape.GetDims(), 0, 2);
    auto data_bool = std::make_unique<bool[]>(detail::SizeFromDims(x_shape.GetDims()));
    for (size_t i = 0; i < data_int.size(); ++i) {
      data_bool.get()[i] = data_int[i] == 0;
    }
    return data_bool;
  }();

  // Calculate the gradient w.r.t. the input X.
  std::vector<float> dX_data = [](const auto& dY_data, const auto& mask_data) {
    auto dX_data = dY_data;
    for (size_t i = 0; i < dY_data.size(); ++i) {
      if (!mask_data.get()[i])
        dX_data[i] = 0.0f;
    }
    return dX_data;
  }(dY_data, mask_data);

  test.AddInput<float>("dY", x_shape.AsShapeVector(), dY_data);
  test.AddInput<bool>("gradient_mask", x_shape.AsShapeVector(), mask_data.get(), x_shape.Size());
  test.AddOutput<float>("dX", x_shape.AsShapeVector(), dX_data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

INSTANTIATE_TEST_SUITE_P(
    FakeQuantGradTests,
    FakeQuantGradParameterizedTest,
    ::testing::Values(
        TensorShape({4}),
        TensorShape({8, 4}),
        TensorShape({4, 7, 13}),
        TensorShape({4, 8, 16, 32}),
        TensorShape({4, 16, 32, 4096})));

#ifdef USE_CUDA

class FakeQuantGradKernelComparisonParameterizedTest : public ::testing::TestWithParam<std::vector<int64_t>> {
};

TEST_P(FakeQuantGradKernelComparisonParameterizedTest, FakeQuantGradKernels) {
  CompareOpTester test("FakeQuantGrad", 1, onnxruntime::kMSDomain);
  auto tensor_dim = GetParam();

  // Randomly generate the gradient w.r.t. the output Y.
  RandomValueGenerator random{};
  std::vector<float> dY_data = random.Uniform<float>(tensor_dim, -1000.0f, 1000.0f);

  // Randomly generate the mask data
  std::unique_ptr<bool[]> mask_data = [&random, &tensor_dim]() {
    auto data_int = random.Uniform<int>(tensor_dim, 0, 2);
    auto data_bool = std::make_unique<bool[]>(detail::SizeFromDims(tensor_dim));
    for (size_t i = 0; i < data_int.size(); ++i) {
      data_bool.get()[i] = data_int[i] == 0;
    }
    return data_bool;
  }();

  // Initialize the gradient w.r.t. the input X with 0s.
  std::vector<float> dX_data = FillZeros<float>(tensor_dim);

  test.AddInput<float>("dY", tensor_dim, dY_data);
  test.AddInput<bool>("gradient_mask", tensor_dim, mask_data.get(), detail::SizeFromDims(tensor_dim));
  test.AddOutput<float>("dX", tensor_dim, dX_data);

  // Compare the outputs from the two kernels
  constexpr double per_sample_tolerance = 2e-4;
  constexpr double relative_per_sample_tolerance = 2e-4;
  test.CompareWithCPU(kCudaExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

INSTANTIATE_TEST_SUITE_P(
    FakeQuantGradTests,
    FakeQuantGradKernelComparisonParameterizedTest,
    ::testing::Values(
        std::vector<int64_t>({4}),
        std::vector<int64_t>({8, 4}),
        std::vector<int64_t>({4, 7, 13}),
        std::vector<int64_t>({4, 8, 16, 32}),
        std::vector<int64_t>({4, 16, 32, 4096})));

#endif

}  // namespace test
}  // namespace onnxruntime
