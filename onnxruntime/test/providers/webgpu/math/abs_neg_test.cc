// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

// Runs a unary op on WebGPU only, with CPU fallback disabled so a missing kernel fails the test.
template <typename T>
void RunUnaryOnWebGpu(const char* op_type, int opset, const std::vector<T>& input, const std::vector<T>& expected) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test(op_type, opset);
  // 7 elements, so the trailing vec4 is partial.
  test.AddInput<T>("X", {static_cast<int64_t>(input.size())}, input);
  test.AddOutput<T>("Y", {static_cast<int64_t>(expected.size())}, expected);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::min();
constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();

}  // namespace

// WebNN conformance "abs int32 4D tensor" and "neg int32 4D tensor" failed to build because Abs/Neg
// were registered for float types only. INT32_MIN wraps, as in two's complement.
TEST(AbsTest, WebGpu_Int32) {
  for (int opset : {7, 13}) {
    RunUnaryOnWebGpu<int32_t>("Abs", opset, {-7, 0, 3, kInt32Max, -kInt32Max, -1, kInt32Min},
                              {7, 0, 3, kInt32Max, kInt32Max, 1, kInt32Min});
  }
}

TEST(AbsTest, WebGpu_Uint32) {
  RunUnaryOnWebGpu<uint32_t>("Abs", 13, {0u, 1u, 7u, 0x80000000u, 0xffffffffu, 42u, 3u},
                             {0u, 1u, 7u, 0x80000000u, 0xffffffffu, 42u, 3u});
}

TEST(NegTest, WebGpu_Int32) {
  for (int opset : {7, 13}) {
    RunUnaryOnWebGpu<int32_t>("Neg", opset, {-7, 0, 3, kInt32Max, -kInt32Max, -1, kInt32Min},
                              {7, 0, -3, -kInt32Max, kInt32Max, 1, kInt32Min});
  }
}

}  // namespace test
}  // namespace onnxruntime
