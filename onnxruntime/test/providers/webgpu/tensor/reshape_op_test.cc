// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
TEST(TensorOpTest, Reshape_int64_webgpu) {
  OpTester test("Reshape", 25);

  test.AddInput<int64_t>("data", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddInput<int64_t>("shape", {3}, {3, 1, 2});
  test.AddOutput<int64_t>("reshaped", {3, 1, 2}, {1, 2, 3, 4, 5, 6});

  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

TEST(TensorOpTest, Reshape_uint8_webgpu) {
  OpTester test("Reshape", 25);

  // Values span the full byte range.
  test.AddInput<uint8_t>("data", {2, 3}, {1, 2, 10, 100, 200, 255});
  test.AddInput<int64_t>("shape", {3}, {3, 1, 2});
  test.AddOutput<uint8_t>("reshaped", {3, 1, 2}, {1, 2, 10, 100, 200, 255});

  // Run on a WebGPU-only session and disable CPU-EP fallback. ORT implicitly adds a CPU EP for
  // fallback, so without this the Reshape node would silently run on CPU (and the test would still
  // pass) if WebGPU did not support uint8.
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(ConfigOptions{});
  if (provider == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  test.Config(options)
      .ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif

}  // namespace test
}  // namespace onnxruntime
