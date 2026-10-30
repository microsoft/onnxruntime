// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/random_generator.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// Q then DQ with round half to even, written independently of the kernels.
template <typename T>
std::vector<float> ReferenceQuantizeDequantize(const std::vector<float>& x, float scale, T zero_point) {
  std::vector<float> y;
  y.reserve(x.size());
  const float lo = static_cast<float>(std::numeric_limits<T>::lowest());
  const float hi = static_cast<float>(std::numeric_limits<T>::max());
  for (float v : x) {
    const float q = std::min(hi, std::max(lo, std::nearbyintf(v / scale) + static_cast<float>(zero_point)));
    y.push_back(static_cast<float>(static_cast<int32_t>(q) - static_cast<int32_t>(zero_point)) * scale);
  }
  return y;
}

template <typename T>
void RunQuantizeDequantizeLinear(const std::vector<int64_t>& dims, const std::vector<float>& x, float scale,
                                 T zero_point, const std::vector<float>& expected) {
  OpTester test("QuantizeDequantizeLinear", 1, kMSDomain);
  test.AddInput<float>("x", dims, x);
  test.AddInput<float>("y_scale", {}, {scale}, true);
  test.AddInput<T>("y_zero_point", {}, {zero_point}, true);
  test.AddOutput<float>("y", dims, expected);
  test.SetOutputTolerance(0.0f, 0.0f);
  test.Run();
}

}  // namespace

TEST(QuantizeDequantizeLinearOpTest, UInt16RoundsHalfToEvenAndSaturates) {
  // x / scale = 0.5, 1.5, 2.5, -0.5, -1.5, and two values far outside [0, 65535] - 32768.
  const std::vector<float> x{1.0f, 3.0f, 5.0f, -1.0f, -3.0f, 1.0e6f, -1.0e6f};
  const std::vector<float> expected{0.0f, 4.0f, 4.0f, 0.0f, -4.0f, 65534.0f, -65536.0f};
  EXPECT_EQ(ReferenceQuantizeDequantize<uint16_t>(x, 2.0f, 32768), expected);
  RunQuantizeDequantizeLinear<uint16_t>({7}, x, 2.0f, 32768, expected);
}

TEST(QuantizeDequantizeLinearOpTest, Int16RoundsHalfToEvenAndSaturates) {
  // x / scale = 0.5, 1.5, -1.5, then values beyond [-32768, 32767] + 3.
  const std::vector<float> x{0.25f, 0.75f, -0.75f, 1.0e6f, -1.0e6f};
  const std::vector<float> expected{0.0f, 1.0f, -1.0f, 16385.0f, -16382.5f};
  EXPECT_EQ(ReferenceQuantizeDequantize<int16_t>(x, 0.5f, -3), expected);
  RunQuantizeDequantizeLinear<int16_t>({5}, x, 0.5f, -3, expected);
}

TEST(QuantizeDequantizeLinearOpTest, LargeTensorMatchesReference) {
  // Not a multiple of the 128-element block, so the parallel split and the tail are both exercised.
  const std::vector<int64_t> dims{3, 1037};
  RandomValueGenerator random{optional<RandomValueGenerator::RandomSeedType>{1234}};
  const std::vector<float> x = random.Uniform<float>(dims, -16.0f, 16.0f);
  RunQuantizeDequantizeLinear<uint16_t>(dims, x, 0.0004f, 32768,
                                        ReferenceQuantizeDequantize<uint16_t>(x, 0.0004f, 32768));
  RunQuantizeDequantizeLinear<int16_t>(dims, x, 0.0003f, 12, ReferenceQuantizeDequantize<int16_t>(x, 0.0003f, 12));
}

TEST(QuantizeDequantizeLinearOpTest, EmptyInput) {
  RunQuantizeDequantizeLinear<uint16_t>({0, 4}, {}, 0.5f, 7, {});
}

TEST(QuantizeDequantizeLinearOpTest, RejectsPerAxisParameters) {
  OpTester test("QuantizeDequantizeLinear", 1, kMSDomain);
  test.AddInput<float>("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.AddInput<float>("y_scale", {2}, {0.5f, 0.25f});
  test.AddInput<uint16_t>("y_zero_point", {2}, {0, 0});
  test.AddOutput<float>("y", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "requires a scalar y_scale and y_zero_point");
}

}  // namespace test
}  // namespace onnxruntime
