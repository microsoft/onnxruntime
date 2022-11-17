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
  test.AddInput<float>("zero_scale", {1}, std::vector<float>({std::nearbyint(zero_point.front())}));

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
  test.AddInput<float>("zero_scale", {1}, {128.0f});
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

}  // namespace test
}  // namespace onnxruntime
