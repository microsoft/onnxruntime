// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
// uint32 is only exercised on the WebGPU EP here.
TEST(CumSumTest, _1DTestUint32_webgpu) {
  auto provider = WebGpuExecutionProviderWithOptions(ConfigOptions{});
  if (provider == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }

  OpTester test("CumSum", 11, onnxruntime::kOnnxDomain);
  test.AddInput<uint32_t>("x", {5}, {1, 2, 3, 4, 5});
  test.AddInput<int32_t>("axis", {}, {0});
  test.AddOutput<uint32_t>("y", {5}, {1, 3, 6, 10, 15});

  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(so)
      .ConfigEp(std::move(provider))
      .RunWithConfig();
}
#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
