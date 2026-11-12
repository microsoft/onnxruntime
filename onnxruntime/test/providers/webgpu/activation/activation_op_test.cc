// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {
namespace {

// Runs Relu-14 on WebGPU only, with CPU fallback disabled so a missing kernel fails the test.
void RunReluInt32OnWebGpu(const std::vector<int64_t>& shape, const std::vector<int32_t>& input,
                          const std::vector<int32_t>& expected) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test("Relu", 14);
  test.AddInput<int32_t>("X", shape, input);
  test.AddOutput<int32_t>("Y", shape, expected);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

// WebNN conformance "relu int32 4D tensor" failed to build because WebGPU Relu was registered for float types only.
TEST(WebGpuActivationTest, Relu_Int32_Opset14_WebGpu) {
  constexpr int32_t kInt32Min = std::numeric_limits<int32_t>::min();
  constexpr int32_t kInt32Max = std::numeric_limits<int32_t>::max();

  // 7 elements leave a partial trailing vec4.
  RunReluInt32OnWebGpu({7}, {kInt32Min, -1, 0, 1, kInt32Max, -42, 42}, {0, 0, 0, 1, kInt32Max, 0, 42});
  // 8 elements fill two vec4s exactly.
  RunReluInt32OnWebGpu({1, 2, 2, 2}, {kInt32Min, -1, 0, 1, kInt32Max, -kInt32Max, 7, -7},
                       {0, 0, 0, 1, kInt32Max, 0, 7, 0});
}

// PRelu is registered on WebGPU for opsets 7-8, 9-15 and 16 (a single kernel). The PRelu tests
// in cpu/activation run at OpTester's default opset (7); these two cover the other registrations. float16 is
// used because the CPU EP's PRelu is float-only, so these cannot silently fall back to it.
TEST(WebGpuActivationTest, PRelu_Float16_Opset9_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("PRelu", 9);

  std::vector<int64_t> x_dims{1, 2, 2, 4};
  test.AddInput<MLFloat16>("X", x_dims,
                           FloatsToMLFloat16s({1.0f, -2.0f, 3.0f, -4.0f,
                                               -5.0f, 6.0f, -7.0f, 8.0f,
                                               -1.0f, 2.0f, -3.0f, 4.0f,
                                               5.0f, -6.0f, 7.0f, -8.0f}));
  test.AddInput<MLFloat16>("slope", {2, 1, 1}, FloatsToMLFloat16s({0.5f, -0.25f}));
  test.AddOutput<MLFloat16>("Y", x_dims,
                            FloatsToMLFloat16s({1.0f, -1.0f, 3.0f, -2.0f,
                                                -2.5f, 6.0f, -3.5f, 8.0f,
                                                0.25f, 2.0f, 0.75f, 4.0f,
                                                5.0f, 1.5f, 7.0f, 2.0f}));
  test.SetOutputAbsErr("Y", 0.01f);
  test.SetOutputRelErr("Y", 0.01f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(WebGpuActivationTest, PRelu_Float16_Opset16_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("PRelu", 16);

  std::vector<int64_t> x_dims{1, 2, 2, 3};
  test.AddInput<MLFloat16>("X", x_dims,
                           FloatsToMLFloat16s({1.0f, -2.0f, 3.0f,
                                               -4.0f, 5.0f, -6.0f,
                                               -1.0f, 2.0f, -3.0f,
                                               4.0f, -5.0f, 6.0f}));
  test.AddInput<MLFloat16>("slope", {2, 1, 1}, FloatsToMLFloat16s({0.5f, 2.0f}));
  test.AddOutput<MLFloat16>("Y", x_dims,
                            FloatsToMLFloat16s({1.0f, -1.0f, 3.0f,
                                                -2.0f, 5.0f, -3.0f,
                                                -2.0f, 2.0f, -6.0f,
                                                4.0f, -10.0f, 6.0f}));
  test.SetOutputAbsErr("Y", 0.01f);
  test.SetOutputRelErr("Y", 0.01f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
