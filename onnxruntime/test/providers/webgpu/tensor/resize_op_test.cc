// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <string>
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
std::vector<T> Convert(const std::vector<float>& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (float v : values) result.push_back(T(v));
  return result;
}

// WebNN conformance "resample2d(upsample) float32 4D tensor options.axes=[1, 2] options.mode='linear'":
// a channels-last [1, 2, 3, 1] input resized to [1, 4, 6, 1] (scales [1, 2, 2, 1]), half_pixel.
template <typename T>
void TestLinearResizeChannelsLast(float abs_error) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  const std::vector<float> input{59.92947006225586f, 41.98918914794922f, 66.39534759521484f,
                                 90.7006607055664f, 86.95105743408203f, 79.10005187988281f};
  const std::vector<float> expected{
      59.92947006225586f, 55.444400787353516f, 46.47425842285156f,
      48.090728759765625f, 60.29380798339844f, 66.39534759521484f,
      67.62226867675781f, 64.02411651611328f, 56.82780838012695f,
      57.31512451171875f, 65.48605346679688f, 69.57152557373047f,
      83.00786590576172f, 81.18354797363281f, 77.534912109375f,
      75.76390838623047f, 75.87055206298828f, 75.92387390136719f,
      90.7006607055664f, 89.76325988769531f, 87.88845825195312f,
      84.9883041381836f, 81.06280517578125f, 79.10005187988281f};

  OpTester test("Resize", 19);
  test.AddAttribute<std::string>("mode", "linear");
  test.AddInput<T>("X", {1, 2, 3, 1}, Convert<T>(input));
  test.AddInput<float>("roi", {0}, {});
  test.AddInput<float>("", {0}, {});  // omitted scales: the output size comes from sizes
  test.AddInput<int64_t>("sizes", {4}, {1, 4, 6, 1}, true);
  test.AddOutput<T>("Y", {1, 4, 6, 1}, Convert<T>(expected));
  test.SetOutputAbsErr("Y", abs_error);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

TEST(ResizeOpTest, WebGpu_LinearChannelsLast_Float) {
  TestLinearResizeChannelsLast<float>(1e-4f);
}

TEST(ResizeOpTest, WebGpu_LinearChannelsLast_Float16) {
  TestLinearResizeChannelsLast<MLFloat16>(0.1f);
}

}  // namespace test
}  // namespace onnxruntime
