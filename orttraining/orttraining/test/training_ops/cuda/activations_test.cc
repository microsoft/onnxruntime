// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if USE_CUDA
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#elif USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#endif

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

  test.CompareWithCPU(kGpuExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, Gelu_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    // bump up the tolerance due to the nature of gelu op accumulation computation complexity
    TestActivations(test_dim, "Gelu", false /* grad_op */, 1e-3, 1e-3);
  }
}

TEST(CudaKernelTest, FastGelu_basic) {
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

TEST(CudaKernelTest, FastGeluGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "FastGeluGrad", true /* grad_op */);
  }
}

TEST(CudaKernelTest, ReluGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "ReluGrad", true /* grad_op */);
  }
}

TEST(CudaKernelTest, SigmoidGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "SigmoidGrad", true /* grad_op */);
  }
}

TEST(CudaKernelTest, TanhGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "TanhGrad", true /* grad_op */);
  }
}

TEST(CudaKernelTest, LeakyReluGrad_basic) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivations(test_dim, "LeakyReluGrad", true /* grad_op */);
  }
}

static void TestActivationsWithBroadcastBias(
    const std::vector<int64_t>& tensor_dim,
    const std::string& operator_name,
    bool is_grad_op,
    double per_sample_tolerance = 2e-4,
    double relative_per_sample_tolerance = 2e-4) {
  ORT_ENFORCE(tensor_dim.size() >= 1);
  const std::vector<int64_t> bias_dim(tensor_dim.end() - 1, tensor_dim.end());

  CompareOpTester test(operator_name.c_str(), 1, onnxruntime::kMSDomain);

  // create rand inputs
  RandomValueGenerator random{};
  if (is_grad_op) {
    std::vector<float> dY_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
    test.AddInput<float>("dY", tensor_dim, dY_data);
  }
  std::vector<float> X_data = random.Uniform<float>(tensor_dim, -10.0f, 10.0f);
  test.AddInput<float>("X", tensor_dim, X_data);

  std::vector<float> B_data = random.Uniform<float>(bias_dim, -10.0f, 10.0f);
  test.AddInput<float>("B", bias_dim, B_data);

  // create output tensors
  if (is_grad_op) {
    std::vector<float> dX_data = FillZeros<float>(tensor_dim);
    test.AddOutput<float>("dX", tensor_dim, dX_data);
  } else {
    std::vector<float> Y_data = FillZeros<float>(tensor_dim);
    test.AddOutput<float>("Y", tensor_dim, Y_data);
  }

  test.CompareWithCPU(kGpuExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, FastGelu_bias) {
  std::vector<std::vector<int64_t>> test_dims{{4}, {16, 2}, {8, 2, 128, 128}};
  for (const auto& test_dim : test_dims) {
    TestActivationsWithBroadcastBias(test_dim, "FastGelu", false);
  }
}

TEST(CudaKernelTest, BiasGeluGradDx_basic) {
  std::vector<std::vector<int64_t>> test_dims{
      {1},
      {8},
      {64},
      {128},
      {512},
      {16, 1},
      {16, 8},
      {16, 64},
      {16, 128},
      {16, 192},
      {16, 256},
      {16, 258},
      {8, 2, 128, 1},
      {8, 2, 128, 8},
      {8, 2, 128, 64},
      {9, 2, 128, 128},
      {16, 128, 6144},
      {16, 127, 6144},
      {16, 128, 6143},
      {16, 3, 224, 224},
      {15, 3, 223, 223},
      // multiplier of the initial 3 dims > 65535
      {128, 3, 224, 2},
      {128, 3, 224, 3},
  };

  for (const auto& test_dim : test_dims) {
    TestActivationsWithBroadcastBias(test_dim, "BiasGeluGrad_dX", true);
  }
}

TEST(CudaKernelTest, BiasFastGeluGradDx_basic) {
  std::vector<std::vector<int64_t>> test_dims{
      {1},
      {8},
      {64},
      {128},
      {512},
      {16, 1},
      {16, 8},
      {16, 64},
      {16, 128},
      {16, 192},
      {16, 256},
      {16, 258},
      {8, 2, 128, 1},
      {8, 2, 128, 8},
      {8, 2, 128, 64},
      {9, 2, 128, 128},
      {16, 128, 6144},
      {16, 127, 6144},
      {16, 128, 6143},
      {16, 3, 224, 224},
      {15, 3, 223, 223},
      // multiplier of the initial 3 dims > 65535
      {128, 3, 224, 2},
      {128, 3, 224, 3},
  };
  for (const auto& test_dim : test_dims) {
    TestActivationsWithBroadcastBias(test_dim, "BiasFastGeluGrad_dX", true);
  }
}

}  // namespace test
}  // namespace onnxruntime
