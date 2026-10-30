// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
TEST(ReductionOpTest, ArgMinMax_WebGpu_Contiguous) {
  constexpr int64_t rows = 7;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  for (const bool is_max : {false, true}) {
    const char* op = is_max ? "ArgMax" : "ArgMin";
    SCOPED_TRACE(op);
    for (const int64_t length : {127, 128, 129, 255, 256, 257, 32769}) {
      SCOPED_TRACE(length);
      const float extreme = is_max ? 5.0f : -5.0f;
      InlinedVector<float> data;
      data.reserve(static_cast<size_t>(rows * length));
      for (int64_t i = 0; i < length; ++i) {
        data.push_back(i == 1 || i == length - 1 ? extreme : 0.0f);
      }
      data.insert(data.end(), length, 7.0f);
      data.push_back(nan);
      data.insert(data.end(), length - 1, extreme);
      for (int64_t i = 0; i < length; ++i) {
        data.push_back(i == 1 ? nan : (i == 2 || i == length - 1 ? extreme : 0.0f));
      }
      data.insert(data.end(), length, is_max ? -infinity : infinity);
      for (int64_t i = 0; i < length; ++i) {
        data.push_back(i % 2 == 0 ? 0.0f : -0.0f);
      }
      data.insert(data.end(), length, nan);

      for (const int64_t select_last : {0, 1}) {
        SCOPED_TRACE(select_last);
        const std::array<int64_t, rows> expected{
            select_last ? length - 1 : 1,
            select_last ? length - 1 : 0,
            0,
            select_last ? length - 1 : 2,
            select_last ? length - 1 : 0,
            select_last ? length - 1 : 0,
            0};
        for (const int64_t keepdims : {0, 1}) {
          SCOPED_TRACE(keepdims);
          ConfigOptions provider_options;
          ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kValidationMode, "full"));
          auto provider = WebGpuExecutionProviderWithOptions(provider_options);
          if (!provider) {
            GTEST_SKIP() << "WebGPU EP is not available";
          }
          SessionOptions session_options;
          ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
          OpTester test(op, 13);
          test.AddAttribute("axis", keepdims ? int64_t{-1} : int64_t{2});
          test.AddAttribute("keepdims", keepdims);
          test.AddAttribute("select_last_index", select_last);
          test.AddInput<float>("data", {1, rows, length}, data.data(), data.size());
          const TensorShapeVector output_shape = keepdims ? TensorShapeVector{1, rows, 1} : TensorShapeVector{1, rows};
          test.AddOutput<int64_t>("reduced", output_shape, expected.data(), expected.size());
          test.Config(session_options).ConfigEp(std::move(provider)).RunWithConfig();
        }
      }
    }
  }
}

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
