// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "gtest/gtest.h"

#include "core/providers/webgpu/webgpu_provider_options.h"
#include "default_providers.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Regression test for segmented storage-buffer writes in Conv2dMM shader output paths.
// Large-shape coverage for real device limits (typically >= 128 MiB for
// maxStorageBufferBindingSize). Kept disabled due to memory footprint and
// machine/device variance in CI.
TEST(Conv_WebGPU, DISABLED_LargeFlatOutputIndex_UsesHelperIndexing_RealDeviceLimit) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  constexpr int64_t n = 1;
  constexpr int64_t c = 1;
  constexpr int64_t h = 8192;
  constexpr int64_t w = 4200;
  constexpr int64_t m = 1;
  constexpr int64_t kh = 1;
  constexpr int64_t kw = 1;

  const std::vector<int64_t> x_shape{n, c, h, w};
  const std::vector<int64_t> w_shape{m, c, kh, kw};
  const std::vector<int64_t> y_shape{n, m, h, w};

  const size_t x_size = static_cast<size_t>(n * c * h * w);
  const size_t w_size = static_cast<size_t>(m * c * kh * kw);
  const size_t y_size = static_cast<size_t>(n * m * h * w);

  std::vector<float> x_vals(x_size, 1.0f);
  std::vector<float> w_vals(w_size, 1.0f);
  std::vector<float> expected_vals(y_size, 1.0f);

  OpTester test("Conv", 11);
  test.AddAttribute("group", static_cast<int64_t>(1));
  test.AddAttribute("kernel_shape", std::vector<int64_t>{kh, kw});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});

  test.AddInput<float>("X", x_shape, x_vals);
  test.AddInput<float>("W", w_shape, w_vals);
  test.AddOutput<float>("Y", y_shape, expected_vals);

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime
