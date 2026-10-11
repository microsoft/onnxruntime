// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#if defined(USE_WEBGPU)
TEST(GRUTest, ForwardDefaultActivationsSimpleWeightsNoBiasLayout1) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  // layout=1 stores X as [batch, sequence, input] and Y as
  // [batch, sequence, num_directions, hidden].
  const std::vector<float> X_data{1.f, 10.f,
                                  2.f, 11.f};
  const std::vector<float> W_data{0.1f, 0.2f, 0.3f,
                                  1.f, 2.f, 3.f,
                                  10.f, 11.f, 12.f};
  const std::vector<float> R_data(3 * 3 * 3, 0.1f);
  const std::vector<float> Y_data{
      // batch 0
      0.4750208f, 0.450166f, 0.4255575f,
      0.6027093f, 0.5083023f, 0.44950223f,
      // batch 1
      0.45016602f, 0.40131235f, 0.35434368f,
      0.5754369f, 0.45485455f, 0.3747841f};
  const std::vector<float> Y_h_data{
      0.6027093f, 0.5083023f, 0.44950223f,
      0.5754369f, 0.45485455f, 0.3747841f};

  OpTester test("GRU", 14);
  test.AddShapeToTensorData();
  test.AddAttribute("activations", std::vector<std::string>{"Sigmoid", "Tanh"});
  test.AddAttribute("direction", "forward");
  test.AddAttribute<int64_t>("hidden_size", 3);
  test.AddAttribute<int64_t>("linear_before_reset", 0);
  test.AddAttribute<int64_t>("layout", 1);
  test.AddInput<float>("X", {2, 2, 1}, X_data);
  test.AddInput<float>("W", {1, 9, 1}, W_data, /*is_initializer=*/true);
  test.AddInput<float>("R", {1, 9, 3}, R_data, /*is_initializer=*/true);
  test.AddOutput<float>("Y", {2, 2, 1, 3}, Y_data);
  test.AddOutput<float>("Y_h", {2, 1, 3}, Y_h_data);

  SessionOptions session_options;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

}  // namespace test
}  // namespace onnxruntime
