// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
void RunSliceOnWebGpu(const std::vector<int64_t>& input_shape, const std::vector<T>& input,
                      const std::vector<int64_t>& starts, const std::vector<int64_t>& ends,
                      const std::vector<int64_t>& axes, const std::vector<int64_t>& steps,
                      const std::vector<int64_t>& output_shape, const std::vector<T>& expected) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  const int64_t count = static_cast<int64_t>(starts.size());
  OpTester test("Slice", 13);
  test.AddInput<T>("data", input_shape, input);
  test.AddInput<int64_t>("starts", {count}, starts, true);
  test.AddInput<int64_t>("ends", {count}, ends, true);
  test.AddInput<int64_t>("axes", {count}, axes, true);
  test.AddInput<int64_t>("steps", {count}, steps, true);
  test.AddOutput<T>("output", output_shape, expected);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

// WebNN conformance "slice int32 2D tensor" previously failed to build because Slice was
// registered for float types only.
TEST(SliceTest, WebGpu_Int32_2D) {
  RunSliceOnWebGpu<int32_t>({3, 4}, {0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11},
                            {1, 1}, {3, 3}, {0, 1}, {1, 1}, {2, 2}, {-5, 6, -9, 10});
}

// A contiguous, four-aligned innermost slice takes the vec4 path.
TEST(SliceTest, WebGpu_Int32_Vectorized) {
  std::vector<int32_t> input(16);
  for (int32_t i = 0; i < 16; ++i) {
    input[i] = (i % 2 == 0) ? i : -i;
  }
  RunSliceOnWebGpu<int32_t>({2, 8}, input, {4}, {8}, {1}, {1}, {2, 4},
                            {4, -5, 6, -7, 12, -13, 14, -15});
}

TEST(SliceTest, WebGpu_Uint32_NegativeStep) {
  RunSliceOnWebGpu<uint32_t>({6}, {10u, 11u, 12u, 13u, 14u, 4000000000u},
                             {5}, {-7}, {0}, {-2}, {3}, {4000000000u, 13u, 11u});
}

}  // namespace test
}  // namespace onnxruntime
