// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// WebNN conformance "erf float16 1D tensor" (tolerance 1/512). Evaluating the approximation in
// float16 was about 2% off for small inputs (0.1279296875 gave 0.146484375, expected 0.1435546875).
TEST(ErfTest, WebGpu_Float16) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  const std::vector<float> input{
      -0.58837890625f, -0.5322265625f, 0.1279296875f, 0.5771484375f,
      0.90478515625f, -0.49853515625f, 0.984375f, 0.2164306640625f,
      0.25244140625f, -0.149169921875f, 0.91357421875f, -0.57373046875f,
      0.740234375f, 0.52001953125f, -0.16552734375f, 0.80078125f,
      0.400146484375f, -0.7490234375f, -0.89453125f, -0.39208984375f,
      0.685546875f, -0.67431640625f, -0.439697265625f, 0.27099609375f};
  const std::vector<float> expected{
      -0.5947265625f, -0.54833984375f, 0.1435546875f, 0.58544921875f,
      0.79931640625f, -0.51904296875f, 0.8359375f, 0.240478515625f,
      0.27880859375f, -0.1671142578125f, 0.8037109375f, -0.5830078125f,
      0.70458984375f, 0.5380859375f, -0.18505859375f, 0.74267578125f,
      0.428466796875f, -0.71044921875f, -0.7939453125f, -0.420654296875f,
      0.66748046875f, -0.65966796875f, -0.466064453125f, 0.298583984375f};

  OpTester test("Erf", 13);
  test.AddInput<MLFloat16>("input", {24}, FloatsToMLFloat16s(input));
  test.AddOutput<MLFloat16>("output", {24}, FloatsToMLFloat16s(expected));
  test.SetOutputAbsErr("output", 1.0f / 512.0f);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime
