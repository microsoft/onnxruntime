// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void TestSoftmax(const std::vector<int64_t>& X_dims,
                        const std::vector<int64_t>& Y_dims,
                        bool is_log_softmax=false,
                        double per_sample_tolerance = 1e-4,
                        double relative_per_sample_tolerance = 1e-4) {
                      
  const char* op = is_log_softmax? "LogSoftmax" : "Softmax";
  CompareOpTester test(op);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(X_dims, -10.0f, 10.0f);
  test.AddInput<float>("X", X_dims, X_data);

  std::vector<float> Y_data = FillZeros<float>(Y_dims);
  test.AddOutput<float>("Y", Y_dims, Y_data);

  test.CompareWithCPU(kCudaExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, Softmax_SmallTensor) {
  std::vector<int64_t> X_dims{8, 2, 128, 128};
  std::vector<int64_t> Y_dims{8, 2, 128, 128};
  TestSoftmax(X_dims, Y_dims, false);
}

TEST(CudaKernelTest, Softmax_LargeTensor) {
  std::vector<int64_t> X_dims{8, 16, 512, 512};
  std::vector<int64_t> Y_dims{8, 16, 512, 512};
  TestSoftmax(X_dims, Y_dims, false);
}

TEST(CudaKernelTest, LogSoftmax_SmallTensor) {
  std::vector<int64_t> X_dims{8, 2, 128, 128};
  std::vector<int64_t> Y_dims{8, 2, 128, 128};
  TestSoftmax(X_dims, Y_dims, true);
}

TEST(CudaKernelTest, LogSoftmax_LargeTensor) {
  std::vector<int64_t> X_dims{8, 16, 512, 512};
  std::vector<int64_t> Y_dims{8, 16, 512, 512};
  TestSoftmax(X_dims, Y_dims, true);
}

static void TestSoftmaxGrad(const std::vector<int64_t>& dY_dims,
                            const std::vector<int64_t>& Y_dims,
                            const std::vector<int64_t>& dX_dims,
                            bool is_log_softmax = false,
                            double per_sample_tolerance = 1e-4,
                            double relative_per_sample_tolerance = 1e-4) {  

  const char* op = is_log_softmax? "LogSoftmaxGrad" : "SoftmaxGrad";
  CompareOpTester test(op, 1, kMSDomain);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, 0.0f, 1.0f);
  // Add 1e-2 for numerical stability to prevent zero probability.
  std::vector<float> Y_data = random.Uniform<float>(Y_dims, 0.02f, 1.02f);

  test.AddInput<float>("dY", dY_dims, dY_data);
  test.AddInput<float>("Y", Y_dims, Y_data);

  std::vector<float> dX_data = FillZeros<float>(dX_dims);
  test.AddOutput<float>("dX", dX_dims, dX_data);

  test.CompareWithCPU(kCudaExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, SoftmaxGrad_SmallTensor) {
  std::vector<int64_t> dY_dims{8, 2, 128, 128};
  std::vector<int64_t> Y_dims{8, 2, 128, 128};
  std::vector<int64_t> dX_dims{8, 2, 128, 128};
  TestSoftmaxGrad(dY_dims, Y_dims, dX_dims);
}

TEST(CudaKernelTest, SoftmaxGrad_LargeTensor) {
  std::vector<int64_t> dY_dims{8, 16, 512, 512};
  std::vector<int64_t> Y_dims{8, 16, 512, 512};
  std::vector<int64_t> dX_dims{8, 16, 512, 512};
  TestSoftmaxGrad(dY_dims, Y_dims, dX_dims);
}

TEST(CudaKernelTest, LogSoftmaxGrad_SmallTensor) {
  std::vector<int64_t> dY_dims{8, 2, 128, 128};
  std::vector<int64_t> Y_dims{8, 2, 128, 128};
  std::vector<int64_t> dX_dims{8, 2, 128, 128};
  TestSoftmaxGrad(dY_dims, Y_dims, dX_dims, true);
}

TEST(CudaKernelTest, LogSoftmaxGrad_LargeTensor) {
  std::vector<int64_t> dY_dims{8, 16, 512, 512};
  std::vector<int64_t> Y_dims{8, 16, 512, 512};
  std::vector<int64_t> dX_dims{8, 16, 512, 512};
  TestSoftmaxGrad(dY_dims, Y_dims, dX_dims, true);
}

}  // namespace test
}  // namespace onnxruntime
