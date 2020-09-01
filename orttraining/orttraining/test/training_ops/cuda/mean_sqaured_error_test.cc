// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void TestMeanSquaredError(const std::vector<int64_t>& X_dims,
                        const std::vector<int64_t>& label_dims,
                        const std::vector<int64_t>& weight_dims,
                        const std::vector<int64_t>& Y_dims,
                        const std::string& reduction,
                        double per_sample_tolerance = 1e-4,
                        double relative_per_sample_tolerance = 1e-4) {
                      
  
  CompareOpTester test("MeanSquaredError", 12, kOnnxDomain);
  // try the same test with NegativeLogLikelihood
  test.AddAttribute("reduction", reduction);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(X_dims, -200.0f, 200.0f);
  std::vector<float> weight_data = random.Uniform<float>(weight_dims, 0.0f, 1.0f);
  std::vector<float> label_data = random.OneHot<float>(label_dims, label_dims.back());
  test.AddInput<float>("scores", X_dims, X_data);
  test.AddInput<float>("labels", label_dims, label_data);
  test.AddInput<float>("weights", weight_dims, weight_data);

  std::vector<float> Y_data = FillZeros<float>(Y_dims);
  test.AddOutput<float>("output", Y_dims, Y_data);

  test.CompareWithCPU(kCudaExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, MeanSquaredError_SmallTensor) {
  std::vector<int64_t> X_dims{1};
  std::vector<int64_t> label_dims{1};
  std::vector<int64_t> weight_dims{1};
  std::vector<int64_t> Y_dims{};
  TestMeanSquaredError(X_dims, label_dims, weight_dims, Y_dims, "mean");
  TestMeanSquaredError(X_dims, label_dims, weight_dims, Y_dims, "sum");
}

}  // namespace test
}  // namespace onnxruntime
