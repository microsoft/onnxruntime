// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {
TEST(ContribOpTest, Rfft) {
  if (DefaultCudaExecutionProvider() == nullptr) return;

  OpTester test("Rfft", 1, onnxruntime::kMSDomain);
  test.AddAttribute("signal_ndim", static_cast<int64_t>(2));
  test.AddAttribute("onesided", static_cast<int64_t>(1));
  test.AddAttribute("normalized", static_cast<int64_t>(0));
  test.AddInput<float>("X", {4, 4}, std::vector<float>{0.81289f, 1.31077f, -0.87902f, -1.20465f, 0.16614f, -0.98306f, 0.58791f, 0.49175f, 1.25058f, 0.72441f, -2.62604f, -1.12684f, -1.68846f, 1.04393f, -0.25949f, 1.87801f});
  test.AddOutput<float>("Y", {4, 3, 2}, std::vector<float>{0.03999f, 0.00000f, 1.69191f, -2.51542f, -0.17225f, 0.00000f, 0.26275f, 0.00000f, -0.42177f, 1.47481f, 1.24536f, 0.00000f, -1.77790f, 0.00000f, 3.87662f, -1.85126f, -0.97304f, 0.00000f, 0.97399f, 0.00000f, -1.42898f, 0.83408f, -4.86989f, 0.00000f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ContribOpTest, Irfft) {
  if (DefaultCudaExecutionProvider() == nullptr) return;

  OpTester test("Irfft", 1, onnxruntime::kMSDomain);
  test.AddAttribute("signal_ndim", static_cast<int64_t>(2));
  test.AddAttribute("onesided", static_cast<int64_t>(1));
  test.AddAttribute("normalized", static_cast<int64_t>(0));
  test.AddInput<float>("X", {4, 3, 2}, std::vector<float>{0.03999f, 0.00000f, 1.69191f, -2.51542f, -0.17225f, 0.00000f, 0.26275f, 0.00000f, -0.42177f, 1.47481f, 1.24536f, 0.00000f, -1.77790f, 0.00000f, 3.87662f, -1.85126f, -0.97304f, 0.00000f, 0.97399f, 0.00000f, -1.42898f, 0.83408f, -4.86989f, 0.00000f});
  test.AddOutput<float>("Y", {4, 4}, std::vector<float>{0.81289f, 1.31077f, -0.87902f, -1.20465f, 0.16614f, -0.98306f, 0.58791f, 0.49175f, 1.25058f, 0.72441f, -2.62604f, -1.12684f, -1.68846f, 1.04393f, -0.25949f, 1.87801f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
}  // namespace test
}  // namespace onnxruntime
