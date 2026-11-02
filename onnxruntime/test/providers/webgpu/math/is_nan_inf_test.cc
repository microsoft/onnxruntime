// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

// 5-D with 14 elements: not a multiple of 4, so the shader's partial trailing vec4 is exercised.
const std::vector<int64_t> kShape{1, 2, 1, 7, 1};
constexpr size_t kNumValues = 14;
constexpr size_t kPosInfIndex = 7;
constexpr size_t kNegInfIndex = 8;
constexpr size_t kFirstNaNIndex = 9;

// Both bit-pattern lists share the same layout: +0, -0, +/- smallest subnormal, 1.0, +/- max finite,
// +Inf, -Inf, +/- smallest NaN (one ULP past Inf), +/- quiet NaN, all-ones NaN.
constexpr uint32_t kF32Bits[kNumValues] = {0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x3f800000u,
                                           0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7f800001u,
                                           0xff800001u, 0x7fc00000u, 0xffc00000u, 0xffffffffu};
constexpr uint16_t kF16Bits[kNumValues] = {0x0000, 0x8000, 0x0001, 0x8001, 0x3c00, 0x7bff, 0xfbff,
                                           0x7c00, 0xfc00, 0x7c01, 0xfc01, 0x7e00, 0xfe00, 0xffff};

std::vector<float> F32Values() {
  std::vector<float> values(kNumValues);
  for (size_t i = 0; i < kNumValues; ++i) {
    std::memcpy(&values[i], &kF32Bits[i], sizeof(float));
  }
  return values;
}

std::vector<MLFloat16> F16Values() {
  std::vector<MLFloat16> values;
  values.reserve(kNumValues);
  for (uint16_t bits : kF16Bits) {
    values.push_back(MLFloat16::FromBits(bits));
  }
  return values;
}

template <typename T>
void RunOnWebGpu(const char* op_type, int opset, const std::vector<T>& input, const bool (&expected)[kNumValues],
                 std::vector<std::pair<const char*, int64_t>> attributes = {}) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  SessionOptions options;
  ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  OpTester test(op_type, opset);
  for (const auto& [name, value] : attributes) {
    test.AddAttribute<int64_t>(name, value);
  }
  test.AddInput<T>("X", kShape, input);
  test.AddOutput<bool>("Y", kShape, expected, kNumValues);
  test.Config(options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

template <typename T>
void TestIsNaN(const std::vector<T>& input) {
  bool expected[kNumValues] = {};
  for (size_t i = kFirstNaNIndex; i < kNumValues; ++i) {
    expected[i] = true;
  }
  for (int opset : {9, 13, 20}) {
    SCOPED_TRACE(::testing::Message() << "opset " << opset);
    RunOnWebGpu("IsNaN", opset, input, expected);
  }
}

template <typename T>
void TestIsInf(const std::vector<T>& input, std::initializer_list<int> opsets) {
  for (int64_t detect_positive : {0, 1}) {
    for (int64_t detect_negative : {0, 1}) {
      bool expected[kNumValues] = {};
      expected[kPosInfIndex] = detect_positive != 0;
      expected[kNegInfIndex] = detect_negative != 0;
      for (int opset : opsets) {
        SCOPED_TRACE(::testing::Message() << "opset " << opset << ", detect_positive " << detect_positive
                                          << ", detect_negative " << detect_negative);
        RunOnWebGpu("IsInf", opset, input, expected,
                    {{"detect_positive", detect_positive}, {"detect_negative", detect_negative}});
      }
    }
  }
}

}  // namespace

TEST(IsNaNOpTest, WebGpu_Float_SpecialValues) {
  TestIsNaN(F32Values());
}

TEST(IsNaNOpTest, WebGpu_Float16_SpecialValues) {
  TestIsNaN(F16Values());
}

TEST(IsInfTest, WebGpu_Float_SpecialValues) {
  TestIsInf(F32Values(), {10, 20});
}

// IsInf supports float16 from opset 20.
TEST(IsInfTest, WebGpu_Float16_SpecialValues) {
  TestIsInf(F16Values(), {20});
}

}  // namespace test
}  // namespace onnxruntime
