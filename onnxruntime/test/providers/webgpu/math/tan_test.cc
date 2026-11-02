// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

// WebNN conformance "tan float32 1D tensor" inputs. 45.59520721435547 failed with the built-in tan():
// -23.733557 against -23.731182, outside the 1/1024 tolerance.
const std::vector<float> kWptTanInputs{
    52.69781494140625f, 70.06912994384766f, 90.49689483642578f, 24.65666961669922f, 11.66512680053711f,
    -50.95264434814453f, 40.320064544677734f, -9.641122817993164f, -31.567750930786133f, 45.59520721435547f,
    -55.93085861206055f, -44.602970123291016f, 80.4539794921875f, -2.314880847930908f, -25.474767684936523f,
    62.589454650878906f, -70.94403076171875f, 62.84861755371094f, 84.79766845703125f, -95.58502960205078f,
    15.552484512329102f, -55.25654220581055f, -26.884889602661133f, 0.159261092543602f};

std::vector<float> ExpectedTan(const std::vector<float>& input) {
  std::vector<float> expected;
  expected.reserve(input.size());
  for (float x : input) {
    expected.push_back(static_cast<float>(std::tan(static_cast<double>(x))));
  }
  return expected;
}

template <typename T>
void RunTanOnWebGpu(const std::vector<T>& input, const std::vector<T>& expected, float abs_error, float rel_error) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test("Tan", 7);
  test.AddInput<T>("input", {static_cast<int64_t>(input.size())}, input);
  test.AddOutput<T>("output", {static_cast<int64_t>(expected.size())}, expected, /*sort_output=*/false, rel_error,
                    abs_error);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

TEST(TanTest, WebGpu_Float_WebNNConformance) {
  RunTanOnWebGpu<float>(kWptTanInputs, ExpectedTan(kWptTanInputs), 1e-5f, 1e-5f);
}

// Quadrant boundaries, a value next to the pi/2 pole and arguments of both signs up to the largest
// multiple of pi/2 (k = +/-8191) the exact argument reduction handles; larger inputs use the built-in tan().
// The last 14 inputs are f32 values next to odd multiples of pi/2 (k = 3 to 8191), where the reduced angle is
// tiny and any rounding in the k * (pi/2 part) products is amplified: with only the products for the 8- and
// 11-bit parts exact, 6715.154296875 (k = 4275) gave 5773775.5 instead of 5774237.
TEST(TanTest, WebGpu_Float_ReductionRange) {
  const std::vector<float> input{0.0f, 1e-3f, -0.7853982f, 0.7853982f, 1.5707963f, -3.1415927f, 2.3561945f,
                                 100.0f, -100.0f, -500.25f, 750.5f, -750.5f, 123.456f, -123.456f, 9000.75f,
                                 -9000.75f, 12000.25f, -12000.25f, 12866.5f, -12866.5f,
                                 4.71238899230957f, -4.71238899230957f, 252.89820861816406f, -252.89820861816406f,
                                 1807.986572265625f, -1807.986572265625f, 5854.35791015625f, -5854.35791015625f,
                                 6715.154296875f, -6715.154296875f, 11191.923828125f, -11191.923828125f,
                                 12866.392578125f, -12866.392578125f};
  RunTanOnWebGpu<float>(input, ExpectedTan(input), 1e-5f, 1e-5f);
}

// Inputs on both sides of the |k| < 8192 cutoff (|x| ~ 12867.17), of both signs, with lanes of each vec4 taking
// different paths (lane 0 is past the cutoff in the first vec4 and below it in the second). Below the cutoff the
// exact reduction must stay within 1e-5, which the built-in tan() misses by far. Past it the built-in tan() is used;
// on NVIDIA D3D12 it is off by up to 2e-3 rad here and returns NaN next to poles, so those inputs are away from
// poles and checked to 1e-2.
TEST(TanTest, WebGpu_Float_FallbackCutoff) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  // {x, whether it is past the cutoff}
  const std::vector<std::pair<float, bool>> cases{
      // First vec4.
      {12867.5f, true},
      {12867.0f, false},
      {-12868.0f, true},
      {-12866.392578125f, false},
      // Second vec4.
      {-12867.0f, false},
      {-12867.5f, true},
      {12866.392578125f, false},
      {12868.25f, true},
  };
  std::vector<float> input;
  for (const auto& [x, past_cutoff] : cases) {
    // k = round(x * 2/pi), as in the shader.
    ASSERT_EQ(std::abs(std::nearbyint(x * 0.63661977236758134f)) >= 8192.0f, past_cutoff) << "x: " << x;
    input.push_back(x);
  }
  const std::vector<float> expected = ExpectedTan(input);

  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  OpTester test("Tan", 7);
  test.AddInput<float>("input", {static_cast<int64_t>(input.size())}, input);
  test.AddOutput<float>("output", {static_cast<int64_t>(expected.size())}, expected);
  test.SetCustomOutputVerifier([&cases, &input, &expected](const std::vector<OrtValue>& fetches, const std::string&) {
    const auto actual = fetches[0].Get<Tensor>().DataAsSpan<float>();
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
      const float tolerance = cases[i].second ? 1e-2f : 1e-5f + 1e-5f * std::abs(expected[i]);
      EXPECT_NEAR(actual[i], expected[i], tolerance) << "x: " << input[i];
    }
  });
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

TEST(TanTest, WebGpu_Float16_WebNNConformance) {
  std::vector<float> input = kWptTanInputs;
  for (float& x : input) {
    x = MLFloat16(x).ToFloat();
  }
  RunTanOnWebGpu<MLFloat16>(FloatsToMLFloat16s(input), FloatsToMLFloat16s(ExpectedTan(input)), 1.0f / 512, 1e-3f);
}

}  // namespace test
}  // namespace onnxruntime
