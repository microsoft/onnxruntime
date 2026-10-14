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
TEST(ReductionOpTest, ReduceSum_WebGpu_EnableInt64) {
  OpTester test("ReduceSum", 13);
  test.AddInput<int64_t>("data", {3}, {10, 20, 30});
  test.AddInput<int64_t>("axes", {1}, {0}, true);
  test.AddOutput<int64_t>("reduced", {1}, {60});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}

// Size divisible by 4: catches issues if the shader is ever accidentally vectorized for INT64.
TEST(ReductionOpTest, ReduceSum_WebGpu_EnableInt64_SizeDiv4) {
  OpTester test("ReduceSum", 13);
  test.AddInput<int64_t>("data", {4}, {10, 20, 30, 40});
  test.AddInput<int64_t>("axes", {1}, {0}, true);
  test.AddOutput<int64_t>("reduced", {1}, {100});
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  test.ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif

}  // namespace test
}  // namespace onnxruntime
