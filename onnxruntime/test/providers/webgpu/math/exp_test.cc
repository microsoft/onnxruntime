// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// WebNN conformance allows 1 ULP for float16 exp; a float16 evaluation was 2 ULP off. Expected values
// are exp() of the float16 inputs, rounded to float16.
TEST(ExpTest, WebGpu_Float16) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  std::vector<MLFloat16> input;
  std::vector<MLFloat16> expected;
  for (int i = -40; i <= 40; ++i) {
    const MLFloat16 x(0.2421875f * static_cast<float>(i));
    input.push_back(x);
    expected.emplace_back(static_cast<float>(std::exp(static_cast<double>(static_cast<float>(x)))));
  }

  // Opset 7 resolves to the Exp-6 kernel; the EP needs a model opset of at least 7.
  for (int opset : {7, 13}) {
    OpTester test("Exp", opset);
    test.AddInput<MLFloat16>("input", {static_cast<int64_t>(input.size())}, input);
    test.AddOutput<MLFloat16>("output", {static_cast<int64_t>(expected.size())}, expected);
    test.SetOutputTolerance(0.0f, 0.002f);
    // All outputs are positive, so adjacent float16 values differ by one in their bit patterns.
    test.SetCustomOutputVerifier([&expected](const std::vector<OrtValue>& fetches, const std::string&) {
      const auto actual = fetches[0].Get<Tensor>().DataAsSpan<MLFloat16>();
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        const int ulps = std::abs(static_cast<int>(actual[i].val) - static_cast<int>(expected[i].val));
        EXPECT_LE(ulps, 1) << "i: " << i << ", expected " << expected[i].ToFloat() << ", got " << actual[i].ToFloat();
      }
    });
    test.Config(options).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
  }
}

}  // namespace test
}  // namespace onnxruntime
