// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

namespace onnxruntime {
namespace test {
TEST(ContribOpTest, Rfft) {
  if (DefaultCudaExecutionProvider() == nullptr && DefaultRocmExecutionProvider() == nullptr) return;

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (DefaultCudaExecutionProvider() != nullptr) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }
  if (DefaultRocmExecutionProvider() != nullptr) {
    execution_providers.push_back(DefaultRocmExecutionProvider());
  }

  OpTester test("Rfft", 1, onnxruntime::kMSDomain);
  test.AddAttribute("signal_ndim", static_cast<int64_t>(1));
  test.AddAttribute("onesided", static_cast<int64_t>(1));
  test.AddAttribute("normalized", static_cast<int64_t>(0));
  // Target values conputed using PyTorch torch.fft.rfft(X, dim=-1, norm="backward")
  test.AddInput<float>("X", {4, 4}, {0.8129f, 1.3108f, -0.8790f, -1.2046f, 0.1661f, -0.9831f, 0.5879f, 0.4918f, 1.2506f, 0.7244f, -2.6260f, -1.1268f, -1.6885f, 1.0439f, -0.2595f, 1.8780f});
  test.AddOutput<float>("Y", {4, 3, 2}, {0.0400f, 0.0000f, 1.6919f, -2.5154f, -0.1722f, 0.0000f, 0.2627f, 0.0000f, -0.4218f, 1.4748f, 1.2454f, 0.0000f, -1.7779f, 0.0000f, 3.8766f, -1.8512f, -0.9730f, 0.0000f, 0.9740f, 0.0000f, -1.4290f, 0.8341f, -4.8699f, 0.0000f});
  test.SetOutputTolerance(0.0001f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(ContribOpTest, Irfft) {
  if (DefaultCudaExecutionProvider() == nullptr && DefaultRocmExecutionProvider() == nullptr) return;

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  if (DefaultCudaExecutionProvider() != nullptr) {
    execution_providers.push_back(DefaultCudaExecutionProvider());
  }
  if (DefaultRocmExecutionProvider() != nullptr) {
    execution_providers.push_back(DefaultRocmExecutionProvider());
  }

  OpTester test("Irfft", 1, onnxruntime::kMSDomain);
  test.AddAttribute("signal_ndim", static_cast<int64_t>(1));
  test.AddAttribute("onesided", static_cast<int64_t>(1));
  test.AddAttribute("normalized", static_cast<int64_t>(0));
  test.AddInput<float>("X", {4, 3, 2}, {0.0400f, 0.0000f, 1.6919f, -2.5154f, -0.1722f, 0.0000f, 0.2627f, 0.0000f, -0.4218f, 1.4748f, 1.2454f, 0.0000f, -1.7779f, 0.0000f, 3.8766f, -1.8512f, -0.9730f, 0.0000f, 0.9740f, 0.0000f, -1.4290f, 0.8341f, -4.8699f, 0.0000f});
  test.AddOutput<float>("Y", {4, 4}, {0.8129f, 1.3108f, -0.8790f, -1.2046f, 0.1661f, -0.9831f, 0.5879f, 0.4918f, 1.2506f, 0.7244f, -2.6260f, -1.1268f, -1.6885f, 1.0439f, -0.2595f, 1.8780f});
  test.SetOutputTolerance(0.0001f);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
}  // namespace test
}  // namespace onnxruntime
