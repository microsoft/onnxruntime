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
// We force a small maxStorageBufferBindingSize so the output uses segmented accessors
// (`GetByOffset`/`SetByOffset`) on a modest tensor shape.
TEST(Conv_WebGPU, LargeFlatOutputIndex_UsesHelperIndexing) {
  constexpr uint64_t kForcedMaxStorageBufferBindingSize = 512 * 1024;  // 512 KiB

  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(
      webgpu::options::kMaxStorageBufferBindingSize,
      std::to_string(kForcedMaxStorageBufferBindingSize).c_str()));

  auto webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  constexpr int64_t n = 1;
  constexpr int64_t c = 4;
  constexpr int64_t h = 256;
  constexpr int64_t w = 256;
  constexpr int64_t m = 4;
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

  // For 1x1 conv with all-ones weights and 4 input channels, each output value is 4.
  std::vector<float> expected_vals(y_size, 4.0f);

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
