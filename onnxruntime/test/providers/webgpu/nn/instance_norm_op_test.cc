// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

using std::vector;

#ifdef USE_WEBGPU
TEST(InstanceNormalizationOpTest, InstanceNormNCHW_webgpu) {
  OpTester test("InstanceNormalization");
  test.AddAttribute("epsilon", 0.009999999776482582f);

  vector<float> input = {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f};
  vector<int64_t> input_dims = {1, 2, 1, 3};
  test.AddInput<float>("input", input_dims, input);

  vector<float> scale = {1.0f, 1.0f};
  vector<int64_t> scale_dims = {2};
  test.AddInput<float>("scale", scale_dims, scale);

  vector<float> B = {0.0f, 2.0f};
  vector<int64_t> B_dims = {2};
  test.AddInput<float>("B", B_dims, B);

  vector<float> expected_output = {-1.21566f, 0.0f, 1.21566f, 2.0f, 2.0f, 2.0f};
  test.AddOutput<float>("Y", input_dims, expected_output);

  test.ConfigEp(DefaultWebGpuExecutionProvider(false)).RunWithConfig();
}

TEST(InstanceNormalizationOpTest, InstanceNormNCHW_webgpu_2) {
  OpTester test("InstanceNormalization");
  test.AddAttribute("epsilon", 0.009999999776482582f);

  vector<float> input = {1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  vector<int64_t> input_dims = {1, 2, 2, 2};
  test.AddInput<float>("input", input_dims, input);

  vector<float> scale = {1.0f, 1.0f};
  vector<int64_t> scale_dims = {2};
  test.AddInput<float>("scale", scale_dims, scale);

  vector<float> B = {0.0f, 2.0f};
  vector<int64_t> B_dims = {2};
  test.AddInput<float>("B", B_dims, B);

  vector<float> expected_output = {-1.40028f, 0.0f, 1.40028f, 0.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  test.AddOutput<float>("Y", input_dims, expected_output);

  test.ConfigEp(DefaultWebGpuExecutionProvider(false)).RunWithConfig();
}

namespace {
// WebNN conformance instance-normalization-zero-epsilon: float16 subnormal inputs with epsilon 0.
// inv_std is 2^24, which overflows float16, so the per-channel scale must stay in f32.
template <typename T>
void TestInstanceNormZeroEpsilon(bool nhwc, bool affine) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider(nhwc);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  const float unit = std::ldexp(1.0f, -24);
  const vector<float> input{unit, -unit, unit, -unit, 2 * unit, -2 * unit, 2 * unit, -2 * unit};
  const vector<float> scale = affine ? vector<float>{0.5f, 2.0f} : vector<float>{1.0f, 1.0f};
  const vector<float> bias = affine ? vector<float>{0.25f, -0.5f} : vector<float>{0.0f, 0.0f};
  const vector<float> expected = affine ? vector<float>{0.75f, -0.25f, 0.75f, -0.25f, 1.5f, -2.5f, 1.5f, -2.5f}
                                        : vector<float>{1, -1, 1, -1, 1, -1, 1, -1};
  const auto convert = [](const vector<float>& values) {
    vector<T> result;
    for (float v : values) result.push_back(T(v));
    return result;
  };

  OpTester test("InstanceNormalization", 21);
  test.AddAttribute("epsilon", 0.0f);
  test.AddInput<T>("input", {1, 2, 2, 2}, convert(input));
  test.AddInput<T>("scale", {2}, convert(scale), true);
  test.AddInput<T>("B", {2}, convert(bias), true);
  test.AddOutput<T>("Y", {1, 2, 2, 2}, convert(expected));
  test.SetOutputAbsErr("Y", 1e-3f);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}
}  // namespace

TEST(InstanceNormalizationOpTest, ZeroEpsilonSubnormal_webgpu) {
  for (bool nhwc : {false, true}) {
    for (bool affine : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "nhwc " << nhwc << ", affine " << affine);
      TestInstanceNormZeroEpsilon<float>(nhwc, affine);
      TestInstanceNormZeroEpsilon<MLFloat16>(nhwc, affine);
    }
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
