// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "gtest/gtest.h"

#include "default_providers.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(Conv_WebGPU, GroupedConvWithPaddingUsesSignedCoordinates) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  // Two independent audio-style channels, represented as a 2D convolution with H=1.
  OpTester test("Conv", 11);
  test.AddAttribute("group", static_cast<int64_t>(2));
  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 3});
  test.AddAttribute("pads", std::vector<int64_t>{0, 1, 0, 1});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", {1, 2, 1, 3},
                       {1.0f, 2.0f, 3.0f,
                        10.0f, 20.0f, 30.0f});
  test.AddInput<float>("W", {2, 1, 1, 3},
                       {1.0f, 2.0f, 1.0f,
                        1.0f, 1.0f, 1.0f});
  test.AddOutput<float>("Y", {1, 2, 1, 3},
                        {4.0f, 8.0f, 8.0f,
                         30.0f, 60.0f, 50.0f});

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime