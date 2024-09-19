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

template <typename T>
static void TestSoftmax(const std::vector<int64_t>& X_dims,
                        const std::vector<int64_t>& Y_dims,
                        int axis = 1,
                        bool is_log_softmax = false,
                        double per_sample_tolerance = 1e-4,
                        double relative_per_sample_tolerance = 1e-4) {
  const char* op = is_log_softmax ? "LogSoftmax" : "Softmax";
  CompareOpTester test(op);
  test.AddAttribute<int64_t>("axis", axis);

  // Use fixed random seed because those tests are not stable.
  // It's impossible to debug if the test fails randomly.
  RandomValueGenerator random{5566};
  // create rand inputs
  std::vector<T> X_data = random.Uniform<T>(X_dims, -10.0f, 10.0f);
  test.AddInput<T>("X", X_dims, X_data);

  std::vector<T> Y_data = FillZeros<T>(Y_dims);
  test.AddOutput<T>("Y", Y_dims, Y_data);

  test.CompareWithCPU(kGpuExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

// small tensor to check softmax_warp_forward
// note: keep nelem <= 1024 to invoke softmax_warp_forward!
TEST(CudaKernelTest, Softmax_SmallTensor_LastAxis) {
  std::vector<int64_t> X_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  TestSoftmax<float>(X_dims, Y_dims, 2, false);
}

TEST(CudaKernelTest, Softmax_SmallTensor_AllAxis) {
  std::vector<int64_t> X_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  TestSoftmax<float>(X_dims, Y_dims, 0, false);
  TestSoftmax<float>(X_dims, Y_dims, 1, false);
}

// large tensor to check cuda DNN softmax forward
TEST(CudaKernelTest, Softmax_LargeTensor_LastAxis) {
  std::vector<int64_t> X_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  TestSoftmax<float>(X_dims, Y_dims, 2, false);
}

TEST(CudaKernelTest, Softmax_LargeTensor_LastAxis_Float16) {
  std::vector<int64_t> X_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 2, false, 1e-3, 1e-3);
}

TEST(CudaKernelTest, Softmax_LargeTensor_LastAxis_Float16_NoPowerOfTwo) {
  std::vector<int64_t> X_dims{8, 16, 1500};
  std::vector<int64_t> Y_dims{8, 16, 1500};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 2, false, 1e-3, 1e-3);
}

TEST(CudaKernelTest, Softmax_LargeTensor_LastAxis_Float16_NoPowerOfTwo2) {
  // at fp16 case, when input is all -65504, the output can't be inf
  std::vector<int64_t> X_dims{8192, 1, 1050};
  std::vector<int64_t> Y_dims{8192, 1, 1050};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 2, false, 1e-3, 1e-3);
  CompareOpTester test("Softmax");
  test.AddAttribute<int64_t>("axis", 1);

  std::vector<MLFloat16> X_data(detail::SizeFromDims(X_dims), (MLFloat16)-65504.0f);
  test.AddInput<MLFloat16>("X", X_dims, X_data);

  std::vector<MLFloat16> Y_data = FillZeros<MLFloat16>(Y_dims);
  test.AddOutput<MLFloat16>("Y", Y_dims, Y_data);

  test.CompareWithCPU(kGpuExecutionProvider, 1e-4, 1e-4);
}

TEST(CudaKernelTest, Softmax_LargeTensor_AllAxis) {
  std::vector<int64_t> X_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  TestSoftmax<float>(X_dims, Y_dims, 0, false);
  TestSoftmax<float>(X_dims, Y_dims, 1, false);
}

TEST(CudaKernelTest, Softmax_LargeTensor_AllAxis_Float16) {
  std::vector<int64_t> X_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 0, false, 1e-3, 1e-3);
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 1, false, 1e-3, 1e-3);
}

TEST(CudaKernelTest, Softmax_LargeTensor_AllAxis_Float16_NoPowerOfTwo) {
  std::vector<int64_t> X_dims{8, 16, 1500};
  std::vector<int64_t> Y_dims{8, 16, 1500};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 0, false, 1e-3, 1e-3);
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 1, false, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmax_SmallTensor_LastAxis) {
  std::vector<int64_t> X_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  TestSoftmax<float>(X_dims, Y_dims, 2, true);
}

TEST(CudaKernelTest, LogSoftmax_SmallTensor_AllAxis) {
  std::vector<int64_t> X_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  TestSoftmax<float>(X_dims, Y_dims, 0, true);
  TestSoftmax<float>(X_dims, Y_dims, 1, true);
}

TEST(CudaKernelTest, LogSoftmax_LargeTensor_LastAxis) {
  std::vector<int64_t> X_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  TestSoftmax<float>(X_dims, Y_dims, 2, true);
}

TEST(CudaKernelTest, LogSoftmax_LargeTensor_LastAxis_Float16) {
  std::vector<int64_t> X_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 2, true, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmax_LargeTensor_LastAxis_Float16_NoPowerOfTwo) {
  std::vector<int64_t> X_dims{8, 16, 1500};
  std::vector<int64_t> Y_dims{8, 16, 1500};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 2, true, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmax_LargeTensor_AllAxis) {
  std::vector<int64_t> X_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  TestSoftmax<float>(X_dims, Y_dims, 0, true);
  TestSoftmax<float>(X_dims, Y_dims, 1, true);
}

TEST(CudaKernelTest, LogSoftmax_LargeTensor_AllAxis_Float16) {
  std::vector<int64_t> X_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 0, true, 1e-3, 1e-3);
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 1, true, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmax_LargeTensor_AllAxis_Float16_NoPowerOfTwo) {
  std::vector<int64_t> X_dims{8, 16, 1500};
  std::vector<int64_t> Y_dims{8, 16, 1500};
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 0, true, 1e-3, 1e-3);
  TestSoftmax<MLFloat16>(X_dims, Y_dims, 1, true, 1e-3, 1e-3);
}

template <typename T>
static void TestSoftmaxGrad(const std::vector<int64_t>& dY_dims,
                            const std::vector<int64_t>& Y_dims,
                            const std::vector<int64_t>& dX_dims,
                            int axis = 1,
                            bool is_log_softmax = false,
                            double per_sample_tolerance = 1e-4,
                            double relative_per_sample_tolerance = 1e-4) {
  const char* op = is_log_softmax ? "LogSoftmaxGrad" : "SoftmaxGrad";
  CompareOpTester test(op, 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", axis);

  // create rand inputs
  RandomValueGenerator random{5566};
  std::vector<T> dY_data = random.Uniform<T>(dY_dims, -1.0f, 1.0f);
  // Add 1e-2 for numerical stability to prevent zero probability.
  std::vector<T> Y_data = random.Uniform<T>(Y_dims, -1.02f, 1.02f);

  test.AddInput<T>("dY", dY_dims, dY_data);
  test.AddInput<T>("Y", Y_dims, Y_data);

  std::vector<T> dX_data = FillZeros<T>(dX_dims);
  test.AddOutput<T>("dX", dX_dims, dX_data);

  test.CompareWithCPU(kGpuExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

// small tensor to check dispatch_softmax_backward
TEST(CudaKernelTest, SoftmaxGrad_SmallTensor_LastAxis) {
  std::vector<int64_t> dY_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  std::vector<int64_t> dX_dims{4, 2, 128};
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 2);
}

TEST(CudaKernelTest, SoftmaxGrad_SmallTensor_AllAxis) {
  std::vector<int64_t> dY_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  std::vector<int64_t> dX_dims{4, 2, 128};
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 0);
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 1);
}

// large tensor to check cuda DNN softmax backward
TEST(CudaKernelTest, SoftmaxGrad_LargeTensor_LastAxis) {
  std::vector<int64_t> dY_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  std::vector<int64_t> dX_dims{8, 16, 2048};
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 2);
}

TEST(CudaKernelTest, SoftmaxGrad_LargeTensor_LastAxis_Float16) {
  std::vector<int64_t> dY_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  std::vector<int64_t> dX_dims{8, 16, 2048};
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 2, false, 1e-3, 1e-3);
}

TEST(CudaKernelTest, SoftmaxGrad_LargeTensor_LastAxis_Float16_NoPowerOfTwo) {
  std::vector<int64_t> dY_dims{8, 16, 1500};
  std::vector<int64_t> Y_dims{8, 16, 1500};
  std::vector<int64_t> dX_dims{8, 16, 1500};
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 2, false, 1e-3, 1e-3);
}

// large tensor to check cuda DNN softmax backward
TEST(CudaKernelTest, SoftmaxGrad_LargeTensor_AllAxis) {
  std::vector<int64_t> dY_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  std::vector<int64_t> dX_dims{8, 16, 512};
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 0);
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 1);
}

TEST(CudaKernelTest, SoftmaxGrad_LargeTensor_AllAxis_Float16) {
  std::vector<int64_t> dY_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  std::vector<int64_t> dX_dims{8, 16, 512};
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 0, false, 1e-3, 1e-3);
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 1, false, 1e-3, 1e-3);
}

TEST(CudaKernelTest, SoftmaxGrad_LargeTensor_AllAxis_Float16_NoPowerOfTwo) {
  std::vector<int64_t> dY_dims{8, 16, 1500};
  std::vector<int64_t> Y_dims{8, 16, 1500};
  std::vector<int64_t> dX_dims{8, 16, 1500};
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 0, false, 1e-3, 1e-3);
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 1, false, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmaxGrad_SmallTensor_LastAxis) {
  std::vector<int64_t> dY_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  std::vector<int64_t> dX_dims{4, 2, 128};
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 2, true);
}

TEST(CudaKernelTest, LogSoftmaxGrad_SmallTensor_AllAxis) {
  std::vector<int64_t> dY_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  std::vector<int64_t> dX_dims{4, 2, 128};
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 0, true);
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 1, true);
}

TEST(CudaKernelTest, LogSoftmaxGrad_LargeTensor_LastAxis) {
  std::vector<int64_t> dY_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  std::vector<int64_t> dX_dims{8, 16, 2048};
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 2, true);
}

TEST(CudaKernelTest, LogSoftmaxGrad_LargeTensor_LastAxis_Float16) {
  std::vector<int64_t> dY_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  std::vector<int64_t> dX_dims{8, 16, 2048};
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 2, true, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmaxGrad_LargeTensor_LastAxis_Float16_NoPowerOfTwo) {
  std::vector<int64_t> dY_dims{8, 16, 1500};
  std::vector<int64_t> Y_dims{8, 16, 1500};
  std::vector<int64_t> dX_dims{8, 16, 1500};
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 2, true, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmaxGrad_LargeTensor_AllAxis) {
  std::vector<int64_t> dY_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  std::vector<int64_t> dX_dims{8, 16, 512};
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 0, true, 1e-3, 1e-3);
  TestSoftmaxGrad<float>(dY_dims, Y_dims, dX_dims, 1, true, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmaxGrad_LargeTensor_AllAxis_Float16) {
  std::vector<int64_t> dY_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  std::vector<int64_t> dX_dims{8, 16, 512};
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 0, true, 1e-3, 1e-3);
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 1, true, 1e-3, 1e-3);
}

TEST(CudaKernelTest, LogSoftmaxGrad_LargeTensor_AllAxis_Float16_NoPowerOfTwo) {
  std::vector<int64_t> dY_dims{8, 16, 1500};
  std::vector<int64_t> Y_dims{8, 16, 1500};
  std::vector<int64_t> dX_dims{8, 16, 1500};
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 0, true, 1e-3, 1e-3);
  TestSoftmaxGrad<MLFloat16>(dY_dims, Y_dims, dX_dims, 1, true, 1e-3, 1e-3);
}

static void TestSoftmaxGrad_13(const std::vector<int64_t>& dY_dims,
                               const std::vector<int64_t>& Y_dims,
                               const std::vector<int64_t>& dX_dims,
                               int axis = 1,
                               bool is_log_softmax = false,
                               double per_sample_tolerance = 1e-4,
                               double relative_per_sample_tolerance = 1e-4) {
  const char* op = is_log_softmax ? "LogSoftmaxGrad_13" : "SoftmaxGrad_13";
  CompareOpTester test(op, 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", axis);

  // create rand inputs
  RandomValueGenerator random{};
  std::vector<float> dY_data = random.Uniform<float>(dY_dims, 0.0f, 1.0f);
  // Add 1e-2 for numerical stability to prevent zero probability.
  std::vector<float> Y_data = random.Uniform<float>(Y_dims, 0.02f, 1.02f);

  test.AddInput<float>("dY", dY_dims, dY_data);
  test.AddInput<float>("Y", Y_dims, Y_data);

  std::vector<float> dX_data = FillZeros<float>(dX_dims);
  test.AddOutput<float>("dX", dX_dims, dX_data);

  test.CompareWithCPU(kGpuExecutionProvider, per_sample_tolerance, relative_per_sample_tolerance);
}

// small tensor to check dispatch_softmax_backward
TEST(CudaKernelTest, SoftmaxGrad_13_SmallTensor_LastAxis) {
  std::vector<int64_t> dY_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  std::vector<int64_t> dX_dims{4, 2, 128};
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 2);
}

TEST(CudaKernelTest, SoftmaxGrad_13_SmallTensor_AllAxis) {
  std::vector<int64_t> dY_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  std::vector<int64_t> dX_dims{4, 2, 128};
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 0);
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 1);
}

// large tensor to check cuda DNN softmax backward
TEST(CudaKernelTest, SoftmaxGrad_13_LargeTensor_LastAxis) {
  std::vector<int64_t> dY_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  std::vector<int64_t> dX_dims{8, 16, 2048};
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 2);
}

// large tensor to check cuda DNN softmax backward
TEST(CudaKernelTest, SoftmaxGrad_13_LargeTensor_AllAxis) {
  std::vector<int64_t> dY_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  std::vector<int64_t> dX_dims{8, 16, 512};
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 0);
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 1);
}

TEST(CudaKernelTest, LogSoftmaxGrad_13_SmallTensor_LastAxis) {
  std::vector<int64_t> dY_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  std::vector<int64_t> dX_dims{4, 2, 128};
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 2, true);
}

TEST(CudaKernelTest, LogSoftmaxGrad_13_SmallTensor_AllAxis) {
  std::vector<int64_t> dY_dims{4, 2, 128};
  std::vector<int64_t> Y_dims{4, 2, 128};
  std::vector<int64_t> dX_dims{4, 2, 128};
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 0, true);
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 1, true);
}

TEST(CudaKernelTest, LogSoftmaxGrad_13_LargeTensor_LastAxis) {
  std::vector<int64_t> dY_dims{8, 16, 2048};
  std::vector<int64_t> Y_dims{8, 16, 2048};
  std::vector<int64_t> dX_dims{8, 16, 2048};
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 2, true);
}

TEST(CudaKernelTest, LogSoftmaxGrad_13_LargeTensor_AllAxis) {
  std::vector<int64_t> dY_dims{8, 16, 512};
  std::vector<int64_t> Y_dims{8, 16, 512};
  std::vector<int64_t> dX_dims{8, 16, 512};
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 0, true);
  TestSoftmaxGrad_13(dY_dims, Y_dims, dX_dims, 1, true);
}

}  // namespace test
}  // namespace onnxruntime
