// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
TEST(PadOpTest, Pad_Wrap_WebGpu_LowerPadExceedsInt32Fails) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  OpTester test("Pad", 19);
  test.AddAttribute("mode", "wrap");
  test.AddInput<float>("data", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<int64_t>("pads", {2}, {2147483648LL, -2147483647LL}, true);
  test.AddOutput<float>("output", {5}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "WebGPU Pad only supports lower pads in the int32 range", {}, nullptr, &eps);
}

TEST(PadOpTest, Pad_Wrap_WebGpu_PadGreaterThanInputDimension) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  OpTester test("Pad", 19);
  test.AddAttribute("mode", "wrap");
  test.AddInput<float>("data", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("pads", {2}, {5, 0}, true);
  test.AddOutput<float>("output", {8}, {2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}
#endif

}  // namespace test
}  // namespace onnxruntime
