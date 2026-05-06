// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_FLOAT8_TYPES)

#include <cmath>
#include <limits>
#include <vector>

#include "core/common/float8.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

TEST(Float8E8M0_Tests, BasicConversion) {
  // Float8E8M0 represents powers of 2: value = 2^(val - 127)
  // val = 127 -> 2^0 = 1.0
  Float8E8M0 one(1.0f);
  EXPECT_EQ(one.val, 127);
  EXPECT_FLOAT_EQ(one.ToFloat(), 1.0f);

  // val = 128 -> 2^1 = 2.0
  Float8E8M0 two(2.0f);
  EXPECT_EQ(two.val, 128);
  EXPECT_FLOAT_EQ(two.ToFloat(), 2.0f);

  // val = 126 -> 2^-1 = 0.5
  Float8E8M0 half(0.5f);
  EXPECT_EQ(half.val, 126);
  EXPECT_FLOAT_EQ(half.ToFloat(), 0.5f);

  // val = 0 -> 2^-127
  Float8E8M0 smallest(0x00, Float8E8M0::FromBits());
  float smallest_val = smallest.ToFloat();
  EXPECT_GT(smallest_val, 0.0f);

  // val = 254 -> 2^127
  Float8E8M0 largest(0xFE, Float8E8M0::FromBits());
  float largest_val = largest.ToFloat();
  EXPECT_GT(largest_val, 0.0f);
}

TEST(Float8E8M0_Tests, NaN) {
  // 0xFF is NaN
  Float8E8M0 nan_val(0xFF, Float8E8M0::FromBits());
  EXPECT_TRUE(nan_val.IsNaN());
  EXPECT_TRUE(std::isnan(nan_val.ToFloat()));

  // Converting NaN float to Float8E8M0
  Float8E8M0 from_nan(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(from_nan.IsNaN());
  EXPECT_EQ(from_nan.val, 0xFF);
}

TEST(Float8E8M0_Tests, Infinity) {
  // Infinity saturates to largest value (0xFE) when saturate=true
  Float8E8M0 from_inf(std::numeric_limits<float>::infinity(), true);
  EXPECT_EQ(from_inf.val, 0xFE);
  EXPECT_FALSE(from_inf.IsNaN());

  // Infinity becomes NaN when saturate=false
  Float8E8M0 from_inf_nosat(std::numeric_limits<float>::infinity(), false);
  EXPECT_TRUE(from_inf_nosat.IsNaN());
}

TEST(Float8E8M0_Tests, NegativeValues) {
  // Negative values saturate to 0 (smallest positive) when saturate=true
  Float8E8M0 from_neg(-1.0f, true);
  EXPECT_EQ(from_neg.val, 0x00);

  // Negative values become NaN when saturate=false
  Float8E8M0 from_neg_nosat(-1.0f, false);
  EXPECT_TRUE(from_neg_nosat.IsNaN());
}

TEST(Float8E8M0_Tests, Zero) {
  // Zero maps to smallest value (2^-127) since there's no zero representation
  Float8E8M0 from_zero(0.0f);
  EXPECT_EQ(from_zero.val, 0x00);
}

TEST(Float8E8M0_Tests, Rounding) {
  // 1.5 should round up to 2.0 (exponent 128)
  Float8E8M0 val_1_5(1.5f);
  EXPECT_EQ(val_1_5.val, 128);  // 2^1 = 2.0

  // 1.25 should round down to 1.0 (mantissa < 0.5)
  Float8E8M0 val_1_25(1.25f);
  EXPECT_EQ(val_1_25.val, 127);  // 2^0 = 1.0

  // 3.0 should round up to 4.0 (mantissa = 0.5)
  Float8E8M0 val_3(3.0f);
  EXPECT_EQ(val_3.val, 129);  // 2^2 = 4.0
}

TEST(Float8E8M0_Tests, FromBits) {
  Float8E8M0 val(0x7F, Float8E8M0::FromBits());
  EXPECT_EQ(val.val, 0x7F);
  EXPECT_FLOAT_EQ(val.ToFloat(), 1.0f);
}

TEST(Float8E8M0_Tests, Operators) {
  Float8E8M0 a(0x7F, Float8E8M0::FromBits());
  Float8E8M0 b(0x7F, Float8E8M0::FromBits());
  Float8E8M0 c(0x80, Float8E8M0::FromBits());

  EXPECT_TRUE(a == b);
  EXPECT_FALSE(a != b);
  EXPECT_TRUE(a != c);
  EXPECT_TRUE(a < c);
}

TEST(Float8E8M0_Tests, NumericLimits) {
  auto max_val = std::numeric_limits<Float8E8M0>::max();
  EXPECT_EQ(max_val.val, 0xFE);

  auto min_val = std::numeric_limits<Float8E8M0>::min();
  EXPECT_EQ(min_val.val, 0x00);

  auto nan_val = std::numeric_limits<Float8E8M0>::quiet_NaN();
  EXPECT_EQ(nan_val.val, 0xFF);

  EXPECT_FALSE(std::numeric_limits<Float8E8M0>::is_signed);
  EXPECT_FALSE(std::numeric_limits<Float8E8M0>::has_infinity);
  EXPECT_TRUE(std::numeric_limits<Float8E8M0>::has_quiet_NaN);
}

TEST(Float8E8M0_Tests, BatchConversion) {
  std::vector<float> floats = {1.0f, 2.0f, 4.0f, 0.5f, 0.25f};
  std::vector<Float8E8M0> fp8(floats.size());

  FloatToFloat8E8M0(floats.data(), fp8.data(), floats.size(), true);

  std::vector<float> result(floats.size());
  Float8E8M0ToFloat(fp8.data(), result.data(), fp8.size());

  for (size_t i = 0; i < floats.size(); i++) {
    EXPECT_FLOAT_EQ(result[i], floats[i]);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_FLOAT8_TYPES)
