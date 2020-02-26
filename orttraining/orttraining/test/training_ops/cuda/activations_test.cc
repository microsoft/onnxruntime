// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"


namespace onnxruntime {
namespace test {

static void TestActivations(const std::vector<int64_t>& tensor_dim,
                            const std::string& operator_name,
                            bool is_grad_op,
                            double per_sample_tolerance = 2e-4,
                            double relative_per_sample_tolerance = 2e-4) {
  CompareOpTester test(operator_name.c_str(), 1, onnxruntime::kMSDomain);

  // create rand inputs
  RandomValueGenerator random{};
  if (is_grad_op) {
    std::vector<float> dY_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
    test.AddInput<float>("dY", tensor_dim, dY_data);
  }
  std::vector<float> X_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
  test.AddInput<float>("X", tensor_dim, X_data);

  // create output tensors
  if (is_grad_op) {
    std::vector<float> dX_data = FillZeros<float>(tensor_dim);
    test.AddOutput<float>("dX", tensor_dim, dX_data);
  } else {
    std::vector<float> Y_data = FillZeros<float>(tensor_dim);
    test.AddOutput<float>("Y", tensor_dim, Y_data);
  }

  test.CompareWithCPU(kCudaExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

// temporary disable this test due to task 615984.
TEST(CudaKernelTest, DISABLED_Gelu_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "Gelu", false /* grad_op */);
  }
}

// temporary disable this test due to task 615984.
TEST(CudaKernelTest, DISABLED_FastGelu_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "FastGelu", false /* grad_op */);
  }
}

TEST(CudaKernelTest, GeluGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "GeluGrad", true /* grad_op */);
  }
}

// temporary disable this test due to task 615984.
TEST(CudaKernelTest, DISABLED_FastGeluGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "FastGeluGrad", true /* grad_op */);
  }
}

}  // namespace test
}  // namespace onnxruntime
