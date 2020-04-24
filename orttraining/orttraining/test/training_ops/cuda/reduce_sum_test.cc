// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void TestReduceSum(const std::vector<int64_t>& X_dims,
                          const std::vector<int64_t>& Y_dims,
                          const std::vector<int64_t>& axes,
                          bool keepdims,
                          double per_sample_tolerance = 2e-4,
                          double relative_per_sample_tolerance = 2e-4) {
  CompareOpTester test("ReduceSum");
  test.AddAttribute("axes", axes);
  test.AddAttribute("keepdims", int64_t(keepdims));

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> X_data = random.Uniform<float>(X_dims, -10.0f, 10.0f);
  test.AddInput<float>("X", X_dims, X_data);

  std::vector<float> Y_data = FillZeros<float>(Y_dims);
  test.AddOutput<float>("Y", Y_dims, Y_data);

  test.CompareWithCPU(kCudaExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

TEST(CudaKernelTest, ReduceSum_Scalar) {
  std::vector<int64_t> X_dims{1};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> axes{0};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_2DtoLastDim) {
  std::vector<int64_t> X_dims{16, 2};
  std::vector<int64_t> Y_dims{2};
  std::vector<int64_t> axes{0};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_SmallTensor) {
  std::vector<int64_t> X_dims{2, 128, 128};
  std::vector<int64_t> Y_dims{128};
  std::vector<int64_t> axes{0, 1};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_MidTensor) {
  std::vector<int64_t> X_dims{2, 512, 3072};
  std::vector<int64_t> Y_dims{3072};
  std::vector<int64_t> axes{0, 1};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_LargeTensor) {
  std::vector<int64_t> X_dims{4, 512, 30528};
  std::vector<int64_t> Y_dims{30528};
  std::vector<int64_t> axes{0, 1};
  bool keepdims = false;
  double per_sample_tolerance = 5e-4;
  double relative_per_sample_tolerance = 5e-2;
  TestReduceSum(X_dims, Y_dims, axes, keepdims, per_sample_tolerance, relative_per_sample_tolerance);
}

}  // namespace test
}  // namespace onnxruntime
