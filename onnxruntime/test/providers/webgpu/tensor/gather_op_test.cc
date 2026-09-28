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
// Validates the WebGPU Gather kernel's packed-byte path for uint8 (bit-shift/mask read,
// OR-shift assembly write). Without this test, a non-WebGPU build would only exercise
// the CPU EP path and miss shader regressions.
TEST(GatherOpTest, Gather_axis1_indices2d_uint8_webgpu) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<uint8_t>("data", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          70, 80, 90});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<uint8_t>("output", {3, 2, 2},
                          {20, 10, 30, 20,
                           50, 40, 60, 50,
                           80, 70, 90, 80});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Validates the WebGPU uint8 Gather kernel with output_size NOT a multiple of 4.
// output shape {3,3} = 9 elements: the last packed u32 word holds only 1 valid byte,
// with the remaining 3 bytes zero-padded (partial-thread boundary path).
TEST(GatherOpTest, Gather_axis0_uint8_non_multiple_of_4_webgpu) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  // 3×3 input with byte-diverse values
  test.AddInput<uint8_t>("data", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          70, 80, 90});
  // 1-D indices of length 3: gather rows 1, 2, 0
  test.AddInput<int32_t>("indices", {3}, {1, 2, 0});
  // output shape {3,3} = 9 bytes (not a multiple of 4)
  test.AddOutput<uint8_t>("output", {3, 3},
                          {40, 50, 60,
                           70, 80, 90,
                           10, 20, 30});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// int64 Gather is gated behind the enableInt64 provider option. Disable CPU-EP fallback so the
// Gather node must run on the WebGPU kernel, and include values outside the int32 range (2^32,
// 2^33+1, a large negative, INT64_MAX) to prove the full 64-bit value is preserved via a raw
// vec2<u32> storage copy rather than truncated to i32.
TEST(GatherOpTest, Gather_int64_webgpu) {
  ConfigOptions provider_options{};
  ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(provider_options);
  if (provider == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }

  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("data", {3, 2},
                         {1, 4294967296,                 // 1, 2^32
                          8589934593, -4294967297,       // 2^33+1, -(2^32+1)
                          100, 9223372036854775807LL});  // 100, INT64_MAX
  test.AddInput<int32_t>("indices", {2}, {2, 0});
  test.AddOutput<int64_t>("output", {2, 2},
                          {100, 9223372036854775807LL,
                           1, 4294967296});

  // Disable CPU-EP fallback so the Gather node must run on the WebGPU kernel.
  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(so)
      .ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
