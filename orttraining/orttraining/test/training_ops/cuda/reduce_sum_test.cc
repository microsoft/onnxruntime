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

static void TestReduceSum(const std::vector<int64_t>& X_dims,
                          const std::vector<int64_t>& Y_dims,
                          const std::vector<int64_t>& axes,
                          bool keepdims,
                          double per_sample_tolerance = 2e-4,
                          double relative_per_sample_tolerance = 2e-4) {
  CompareOpTester test("ReduceSum");
  if (!axes.empty()) test.AddAttribute("axes", axes);
  test.AddAttribute("keepdims", int64_t(keepdims));

  // create rand inputs
  RandomValueGenerator random{};
  const bool is_positive = random.Uniform<int>({1}, 0, 2)[0] == 0;
  const float range_begin = is_positive ? 1.0f : -10.0f;
  const float range_end = is_positive ? 10.0f : -1.0f;
  const std::vector<float> X_data = random.Uniform<float>(X_dims, range_begin, range_end);
  test.AddInput<float>("X", X_dims, X_data);

  const std::vector<float> Y_data = FillZeros<float>(Y_dims);
  test.AddOutput<float>("Y", Y_dims, Y_data);

  test.CompareWithCPU(kGpuExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
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
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_SmallTensorTrailingAxes) {
  std::vector<int64_t> X_dims{128, 2, 128};
  std::vector<int64_t> Y_dims{128};
  std::vector<int64_t> axes{1, 2};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_MidTensorTrailingAxes) {
  std::vector<int64_t> X_dims{3072, 2, 512};
  std::vector<int64_t> Y_dims{3072};
  std::vector<int64_t> axes{1, 2};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_LargeTensorTrailingAxes) {
  std::vector<int64_t> X_dims{30528, 4, 512};
  std::vector<int64_t> Y_dims{30528};
  std::vector<int64_t> axes{1, 2};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_OneDimsOptimization) {
  std::vector<int64_t> X_dims{2, 3, 1, 4, 1, 5};
  std::vector<int64_t> Y_dims{3, 4, 5};
  std::vector<int64_t> axes{0, 2, 4};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_ReduceOnOneDims) {
  std::vector<int64_t> X_dims{2, 1, 1};
  std::vector<int64_t> Y_dims{2};
  std::vector<int64_t> axes{1, 2};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

TEST(CudaKernelTest, ReduceSum_AllOneDims) {
  std::vector<int64_t> X_dims{1, 1};
  std::vector<int64_t> Y_dims{};
  std::vector<int64_t> axes{};
  bool keepdims = false;
  TestReduceSum(X_dims, Y_dims, axes, keepdims);
}

}  // namespace test
}  // namespace onnxruntime
