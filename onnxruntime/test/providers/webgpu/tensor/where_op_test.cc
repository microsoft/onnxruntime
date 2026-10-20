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

#ifdef USE_WEBGPU
namespace {
constexpr char kOpName[] = "Where";
constexpr int kOpVersion = 9;
}  // namespace

// Non-broadcast: all inputs have the same shape. Exercises the is_int64_ non-broadcast path.
TEST(WhereOpTest, EnableWebGpuInt64) {
  OpTester test{kOpName, kOpVersion};
  test.AddInput<bool>("condition", {4}, {true, false, true, false});
  test.AddInput<int64_t>("X", {4}, {10, 20, 30, 40});
  test.AddInput<int64_t>("Y", {4}, {1, 2, 3, 4});
  test.AddOutput<int64_t>("output", {4}, {10, 2, 30, 4});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

// Broadcast: condition [1,4] broadcasts over X/Y [2,4]. Exercises the is_int64_ broadcast path
// where BroadcastedIndicesToOffset computes a different source offset per output element.
TEST(WhereOpTest, EnableBroadcastWebGpuInt64) {
  // condition [1,4] broadcasts against X [2,4] and Y [2,4] -> output [2,4]
  OpTester test{kOpName, kOpVersion};
  test.AddInput<bool>("condition", {1, 4}, {true, false, true, false});
  test.AddInput<int64_t>("X", {2, 4}, {10, 20, 30, 40, 50, 60, 70, 80});
  test.AddInput<int64_t>("Y", {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  test.AddOutput<int64_t>("output", {2, 4}, {10, 2, 30, 4, 50, 6, 70, 8});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif

}  // namespace test
}  // namespace onnxruntime
