// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
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

constexpr float kInf = std::numeric_limits<float>::infinity();

template <typename T>
std::vector<T> Convert(const std::vector<float>& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (float v : values) {
    result.push_back(T(v));
  }
  return result;
}

// Runs a unary op on WebGPU only, with CPU fallback disabled so a missing kernel fails the test.
template <typename T>
void RunUnaryOnWebGpu(const char* op_type, int opset, const std::vector<int64_t>& shape,
                      const std::vector<T>& input, const std::vector<T>& expected, float abs_error = 0.0f) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test(op_type, opset);
  test.AddInput<T>("X", shape, input);
  test.AddOutput<T>("Y", shape, expected);
  if (abs_error > 0.0f) {
    test.SetOutputAbsErr("Y", abs_error);
  }
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// Ties round to even; 9 elements leaves a partial trailing vec4.
const std::vector<float> kRoundInput{0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 2.4f, -2.6f, 7.0f};
const std::vector<float> kRoundExpected{0.0f, 2.0f, 2.0f, -0.0f, -2.0f, -2.0f, 2.0f, -3.0f, 7.0f};

const std::vector<float> kSignInput{-0.946f, 0.9996f, -kInf, kInf, 0.0f, -3.0f, 5.0f};
const std::vector<float> kSignExpected{-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f};

std::vector<float> SoftsignInput() {
  return {-100.0f, -4.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 4.0f, 100.0f};
}

std::vector<float> SoftsignExpected() {
  std::vector<float> expected;
  for (float v : SoftsignInput()) {
    expected.push_back(v / (1.0f + std::abs(v)));
  }
  return expected;
}

}  // namespace

TEST(RoundTest, WebGpu_Float) {
  for (int opset : {11, 22}) {
    RunUnaryOnWebGpu<float>("Round", opset, {3, 3}, kRoundInput, kRoundExpected);
  }
}

TEST(RoundTest, WebGpu_Float16) {
  for (int opset : {11, 22}) {
    RunUnaryOnWebGpu<MLFloat16>("Round", opset, {3, 3}, Convert<MLFloat16>(kRoundInput),
                                Convert<MLFloat16>(kRoundExpected));
  }
}

TEST(SignTest, WebGpu_Float) {
  for (int opset : {9, 13}) {
    RunUnaryOnWebGpu<float>("Sign", opset, {7}, kSignInput, kSignExpected);
  }
}

TEST(SignTest, WebGpu_Float16) {
  for (int opset : {9, 13}) {
    RunUnaryOnWebGpu<MLFloat16>("Sign", opset, {7}, Convert<MLFloat16>(kSignInput),
                                Convert<MLFloat16>(kSignExpected));
  }
}

TEST(SignTest, WebGpu_Int32) {
  for (int opset : {9, 13}) {
    RunUnaryOnWebGpu<int32_t>("Sign", opset, {2, 3},
                              {-7, 0, 3, std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), -1},
                              {-1, 0, 1, -1, 1, -1});
  }
}

TEST(SoftsignTest, WebGpu_Float) {
  // Large magnitudes, where a direct GPU division can flush to 0, must give +/-1.
  std::vector<float> input = SoftsignInput();
  input.insert(input.end(), {3.0e38f, -3.0e38f, 2.0e7f, -2.0e7f});
  std::vector<float> expected;
  for (float v : input) {
    expected.push_back(v / (1.0f + std::abs(v)));
  }
  // Opset 13 resolves to the Softsign-1 kernel; the EP needs a model opset of at least 7.
  for (int opset : {13, 22}) {
    RunUnaryOnWebGpu<float>("Softsign", opset, {static_cast<int64_t>(input.size())}, input, expected, 1e-6f);
  }
}

TEST(SoftsignTest, WebGpu_Float16) {
  for (int opset : {13, 22}) {
    RunUnaryOnWebGpu<MLFloat16>("Softsign", opset, {9}, Convert<MLFloat16>(SoftsignInput()),
                                Convert<MLFloat16>(SoftsignExpected()), 1e-3f);
  }
}

}  // namespace test
}  // namespace onnxruntime
