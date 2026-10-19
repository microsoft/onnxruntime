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
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

// int32 Clip, run explicitly on the WebGPU EP (the 4-byte templated Clip kernel). Uses opset 12 (the
// earliest at which integer Clip is valid) to exercise the 12-12 registration; the uint32 test below
// uses opset 13 for the other registration. Mirrors the shared Clip_int32 reference values.
TEST(MathOpTest, Clip_int32_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int32_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddInput<int32_t>("min", {}, {-10});
  test.AddInput<int32_t>("max", {}, {10});
  test.AddOutput<int32_t>("Y", dims,
                          {-1, 0, 1,
                           -10, 10, -6,
                           -5, 2, 10});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// uint32 Clip, run explicitly on the WebGPU EP. Mirrors the shared Clip_uint32 reference values.
TEST(MathOpTest, Clip_uint32_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 13);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<uint32_t>("X", dims,
                          {0, 0, 1,
                           5, 12, 3,
                           2, 7, 16});
  test.AddInput<uint32_t>("min", {}, {3});
  test.AddInput<uint32_t>("max", {}, {10});
  test.AddOutput<uint32_t>("Y", dims,
                           {3, 3, 3,
                            5, 10, 3,
                            3, 7, 10});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#ifdef USE_WEBGPU
// int64 Clip, run explicitly on the WebGPU EP. WebGPU has no native 64-bit integer type; the EP
// stores int64 as vec2<u32> and the dedicated ClipInt64 kernel clamps on the truncated low 32 bits
// (interpreted as i32), then sign-extends on write. Values are kept within the int32 range -- the
// realistic index/position case and where the truncated clamp matches the reference result.
TEST(MathOpTest, Clip_int64_WebGpu) {
  // int64 Clip is gated behind the enableInt64 option, so build the EP with it enabled.
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 13);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddInput<int64_t>("min", {}, {-10});
  test.AddInput<int64_t>("max", {}, {10});
  test.AddOutput<int64_t>("Y", dims,
                          {-1, 0, 1,
                           -10, 10, -6,
                           -5, 2, 10});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// int64 Clip on WebGPU with min/max OUTSIDE the int32 range. ClipInt64 saturates the bounds into
// the i32 range the shader operates on, so an out-of-i32-range min/max means "no clamp on that
// side" -- the data (all within int32 range here) must pass through unchanged. Without saturation
// the raw int64 bounds would be truncated to bogus i32 values and clamp incorrectly, so this pins
// the saturate_to_i32 behavior.
TEST(MathOpTest, Clip_int64_saturate_WebGpu) {
  // int64 Clip is gated behind the enableInt64 option, so build the EP with it enabled.
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 13);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {-2147483648, -1000, -1,
                          0, 1, 1000,
                          2147483647, 42, -42});
  // -5000000000 < INT32_MIN and 5000000000 > INT32_MAX; both saturate to the i32 limits.
  test.AddInput<int64_t>("min", {}, {-5000000000LL});
  test.AddInput<int64_t>("max", {}, {5000000000LL});
  test.AddOutput<int64_t>("Y", dims,
                          {-2147483648, -1000, -1,
                           0, 1, 1000,
                           2147483647, 42, -42});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// int64 Clip on WebGPU at opset 12 with only the optional `min` input present (max omitted). This
// exercises two paths the opset-13 tests above do not: the opset 12-12 kernel registration, and the
// clip_max_tensor == nullptr branch in ClipInt64 (a missing bound means "no clamp on that side", so
// max defaults to INT32_MAX). Only the lower bound is applied; values above min pass through.
// (Opset 12 is the earliest at which integer Clip is valid; opset 11 constrains T to float types.)
TEST(MathOpTest, Clip_int64_min_only_opset12_WebGpu) {
  // int64 Clip is gated behind the enableInt64 option, so build the EP with it enabled.
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddInput<int64_t>("min", {}, {-10});
  // max omitted: no upper clamp.
  test.AddOutput<int64_t>("Y", dims,
                          {-1, 0, 1,
                           -10, 12, -6,
                           -5, 2, 16});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Symmetric to the min-only test above: int64 Clip on WebGPU at opset 12 with only the optional
// `max` input present (min omitted). This covers the clip_min_tensor == nullptr branch in ClipInt64
// (a missing lower bound means "no clamp on that side", so min defaults to INT32_MIN). Only the
// upper bound is applied; values below max pass through.
TEST(MathOpTest, Clip_int64_max_only_opset12_WebGpu) {
  // int64 Clip is gated behind the enableInt64 option, so build the EP with it enabled.
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddOptionalInputEdge<int64_t>();  // no min
  test.AddInput<int64_t>("max", {}, {10});
  test.AddOutput<int64_t>("Y", dims,
                          {-1, 0, 1,
                           -16, 10, -6,
                           -5, 2, 10});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
