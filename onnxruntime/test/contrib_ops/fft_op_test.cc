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
  test.AddInput<float>("X", {4, 5}, std::vector<float>{-0.8992f, 0.6117f, -1.6091f, -0.4155f, -0.8346f, -2.1596f, -0.0853f, 0.7232f, 0.1941f, -0.0789f, -2.0329f, 1.1031f, 0.6869f, -0.5042f, 0.9895f, -0.1884f, 0.2858f, -1.5831f, 0.9917f, -0.8356f});
  test.AddOutput<float>("Y", {4, 3, 2}, std::vector<float>{-5.6404f, 0.0000f, -3.6965f, -1.3401f, -6.6836f, -3.5202f, -3.3891f, 0.0769f, 1.4521f, 3.2068f, 5.9398f, -1.2344f, -0.1682f, 0.0000f, 1.9681f, -1.6241f, -3.3442f, 1.6817f, -3.3891f, -0.0769f, 2.9557f, -2.9384f, -1.2900f, -4.8683f});
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
  test.AddInput<float>("X", {4, 3, 2}, std::vector<float>{-5.6404f, 0.0000f, -3.6965f, -1.3401f, -6.6836f, -3.5202f, -3.3891f, 0.0769f, 1.4521f, 3.2068f, 5.9398f, -1.2344f, -0.1682f, 0.0000f, 1.9681f, -1.6241f, -3.3442f, 1.6817f, -3.3891f, -0.0769f, 2.9557f, -2.9384f, -1.2900f, -4.8683f});
  test.AddOutput<float>("Y", {4, 5}, std::vector<float>{-0.8992f, 0.6117f, -1.6091f, -0.4155f, -0.8346f, -2.1596f, -0.0853f, 0.7232f, 0.1941f, -0.0789f, -2.0329f, 1.1031f, 0.6869f, -0.5042f, 0.9895f, -0.1884f, 0.2858f, -1.5831f, 0.9917f, -0.8356f});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
}  // namespace test
}  // namespace onnxruntime
