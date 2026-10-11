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
TEST(ExpandOpTest, Expand_3x3_int64_webgpu) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {1}, {1});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

TEST(ExpandOpTest, Expand_3x1_int64_webgpu) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {3}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 2, 3,
                           1, 2, 3,
                           1, 2, 3});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

TEST(ExpandOpTest, Expand_1x3_int64_webgpu) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {3, 1}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

TEST(ExpandOpTest, Expand_3x1x3x1_int64_webgpu) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {1, 3, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddInput<int64_t>("data_1", {4}, {3, 1, 3, 1});
  test.AddOutput<int64_t>("result", {3, 3, 3, 3},
                          {1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9,
                           1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9,
                           1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

TEST(ExpandOpTest, Expand_3x3_int64_webgpu_max_num_pending_dispatches) {
  OpTester test("Expand", 8);

  test.AddInput<int64_t>("data_0", {1}, {1});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});

  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1});

  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kMaxNumPendingDispatches, "32"));

  auto provider = WebGpuExecutionProviderWithOptions(config_options);

  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif

}  // namespace test
}  // namespace onnxruntime
