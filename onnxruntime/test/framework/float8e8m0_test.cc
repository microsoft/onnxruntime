// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_FLOAT8_TYPES)

#include <cmath>
#include <cstring>
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

  // Converting positive NaN float to Float8E8M0
  Float8E8M0 from_nan(std::numeric_limits<float>::quiet_NaN());
  EXPECT_TRUE(from_nan.IsNaN());
  EXPECT_EQ(from_nan.val, 0xFF);

  // Converting negative NaN float to Float8E8M0 (NaN has no sign semantics)
  float neg_nan;
  uint32_t neg_nan_bits = 0xFFC00000;  // negative quiet NaN
  std::memcpy(&neg_nan, &neg_nan_bits, sizeof(float));
  Float8E8M0 from_neg_nan(neg_nan);
  EXPECT_TRUE(from_neg_nan.IsNaN());
  EXPECT_EQ(from_neg_nan.val, 0xFF);
}

TEST(Float8E8M0_Tests, Infinity) {
  // Positive infinity saturates to largest value (0xFE) when saturate=true
  Float8E8M0 from_inf(std::numeric_limits<float>::infinity(), true);
  EXPECT_EQ(from_inf.val, 0xFE);
  EXPECT_FALSE(from_inf.IsNaN());

  // Positive infinity becomes NaN when saturate=false
  Float8E8M0 from_inf_nosat(std::numeric_limits<float>::infinity(), false);
  EXPECT_TRUE(from_inf_nosat.IsNaN());

  // Negative infinity saturates to smallest value (0x00) when saturate=true
  Float8E8M0 from_neg_inf(-std::numeric_limits<float>::infinity(), true);
  EXPECT_EQ(from_neg_inf.val, 0x00);

  // Negative infinity becomes NaN when saturate=false
  Float8E8M0 from_neg_inf_nosat(-std::numeric_limits<float>::infinity(), false);
  EXPECT_TRUE(from_neg_inf_nosat.IsNaN());
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
  // Zero maps to smallest value (2^-127) when saturate=true since there's no zero representation
  Float8E8M0 from_zero(0.0f, true);
  EXPECT_EQ(from_zero.val, 0x00);

  // Zero produces NaN when saturate=false
  Float8E8M0 from_zero_nosat(0.0f, false);
  EXPECT_TRUE(from_zero_nosat.IsNaN());
  EXPECT_EQ(from_zero_nosat.val, 0xFF);
}

TEST(Float8E8M0_Tests, NegativeZero) {
  // -0.0f should map to 0x00 (same as +0.0f), not trigger negative path
  Float8E8M0 from_neg_zero(-0.0f);
  EXPECT_EQ(from_neg_zero.val, 0x00);
}

TEST(Float8E8M0_Tests, ZeroRoundTrip) {
  // E8M0 cannot represent zero; val=0 maps to 2^(-127)
  Float8E8M0 from_zero(0.0f);
  float round_trip = from_zero.ToFloat();
  EXPECT_NE(round_trip, 0.0f);  // documents non-obvious behavior
  EXPECT_FLOAT_EQ(round_trip, std::ldexp(1.0f, -127));
}

TEST(Float8E8M0_Tests, Rounding) {
  // 1.5 should round up to 2.0 (exponent 128)
  Float8E8M0 val_1_5(1.5f);
  EXPECT_EQ(val_1_5.val, 128);  // 2^1 = 2.0

  // 1.25 rounds up to 2.0 with default "up" (ceiling) mode since mantissa != 0
  Float8E8M0 val_1_25(1.25f);
  EXPECT_EQ(val_1_25.val, 128);  // 2^1 = 2.0

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

TEST(Float8E8M0_Tests, FullBitPatternRoundTrip) {
  // All bit patterns 0x00..0xFE should produce finite positive floats
  // that round-trip back to the same bit pattern
  for (int i = 0; i <= 0xFE; i++) {
    Float8E8M0 original(static_cast<uint8_t>(i), Float8E8M0::FromBits());
    float f = original.ToFloat();
    EXPECT_FALSE(std::isnan(f)) << "val=" << i << " produced NaN";
    EXPECT_GT(f, 0.0f) << "val=" << i << " produced non-positive value";

    // Round-trip: convert back to Float8E8M0
    Float8E8M0 round_tripped(f);
    EXPECT_EQ(round_tripped.val, original.val)
        << "Round-trip failed for val=" << i << " (float=" << f << ")";
  }
}

TEST(Float8E8M0_Tests, NaNIdentity) {
  // NaN != NaN (IEEE semantics)
  Float8E8M0 nan_a(0xFF, Float8E8M0::FromBits());
  float fa = nan_a.ToFloat();
  EXPECT_TRUE(std::isnan(fa));
  EXPECT_FALSE(fa == fa);  // NaN is not equal to itself
}

TEST(Float8E8M0_Tests, SignalingNaN) {
  // Signaling NaN should also map to 0xFF
  uint32_t snan_bits = 0x7F800001;  // positive signaling NaN
  float snan;
  std::memcpy(&snan, &snan_bits, sizeof(float));
  Float8E8M0 from_snan(snan);
  EXPECT_TRUE(from_snan.IsNaN());
  EXPECT_EQ(from_snan.val, 0xFF);

  // Negative signaling NaN
  uint32_t neg_snan_bits = 0xFF800001;
  float neg_snan;
  std::memcpy(&neg_snan, &neg_snan_bits, sizeof(float));
  Float8E8M0 from_neg_snan(neg_snan);
  EXPECT_TRUE(from_neg_snan.IsNaN());
  EXPECT_EQ(from_neg_snan.val, 0xFF);
}

TEST(Float8E8M0_Tests, ExactBoundaries) {
  // 2^(-127) = val 0
  Float8E8M0 min_val(0x00, Float8E8M0::FromBits());
  EXPECT_FLOAT_EQ(min_val.ToFloat(), std::ldexp(1.0f, -127));

  // 2^127 = val 254
  Float8E8M0 max_val(0xFE, Float8E8M0::FromBits());
  EXPECT_FLOAT_EQ(max_val.ToFloat(), std::ldexp(1.0f, 127));
}

TEST(Float8E8M0_Tests, SaturateFalseOverflow) {
  // Value above max with mantissa below rounding threshold
  // still produces NaN with saturate=false when exponent overflows
  float large = std::ldexp(1.0f, 127);  // exactly 2^127 = val 254
  Float8E8M0 exact_max(large, false);
  EXPECT_EQ(exact_max.val, 0xFE);  // exact match, no overflow

  // 1.5 * 2^127 rounds up to 2^128, which overflows
  float above_max = 1.5f * std::ldexp(1.0f, 127);
  Float8E8M0 overflow_nosat(above_max, false);
  EXPECT_TRUE(overflow_nosat.IsNaN());
}

TEST(Float8E8M0_Tests, SubnormalRounding) {
  // Float32 subnormals near the top of the range should round up to 2^-126 (val=1)
  // The largest subnormal is just below 2^-126
  uint32_t largest_subnorm_bits = 0x007FFFFF;
  float largest_subnorm;
  std::memcpy(&largest_subnorm, &largest_subnorm_bits, sizeof(float));
  Float8E8M0 from_largest_subnorm(largest_subnorm, true);
  EXPECT_EQ(from_largest_subnorm.val, 0x01);  // Rounds up to 2^(-126)

  // Small subnormals should round down to 2^-127 (val=0)
  uint32_t small_subnorm_bits = 0x00200000;  // well below midpoint
  float small_subnorm;
  std::memcpy(&small_subnorm, &small_subnorm_bits, sizeof(float));
  Float8E8M0 from_small_subnorm(small_subnorm, true);
  EXPECT_EQ(from_small_subnorm.val, 0x00);  // Rounds down to 2^(-127)

  // In nearest mode, subnormals below the midpoint between 2^-127 and 2^-126
  // round down to 2^-127. Up mode would round this value to 2^-126.
  uint32_t below_midpoint_bits = 0x00500000;
  float below_midpoint;
  std::memcpy(&below_midpoint, &below_midpoint_bits, sizeof(float));
  Float8E8M0 nearest_below_midpoint(below_midpoint, true, Float8E8M0::RoundMode::Nearest);
  EXPECT_EQ(nearest_below_midpoint.val, 0x00);

  // The exact midpoint ties upward.
  uint32_t midpoint_bits = 0x00600000;
  float midpoint;
  std::memcpy(&midpoint, &midpoint_bits, sizeof(float));
  Float8E8M0 nearest_midpoint(midpoint, true, Float8E8M0::RoundMode::Nearest);
  EXPECT_EQ(nearest_midpoint.val, 0x01);

  // With saturate=false, subnormals within E8M0 range are still valid positive values,
  // so they round normally (not NaN). Largest subnormal rounds up to 2^(-126).
  Float8E8M0 subnorm_nosat(largest_subnorm, false);
  EXPECT_EQ(subnorm_nosat.val, 0x01);
}

TEST(Float8E8M0_Tests, BatchConversionSpecialValues) {
  // Test batch conversion with special values
  std::vector<float> floats = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      0.0f,
      1.0f,
  };
  std::vector<Float8E8M0> fp8(floats.size());
  FloatToFloat8E8M0(floats.data(), fp8.data(), floats.size(), true);

  EXPECT_EQ(fp8[0].val, 0xFF);  // NaN
  EXPECT_EQ(fp8[1].val, 0xFE);  // Inf saturates to max
  EXPECT_EQ(fp8[2].val, 0x00);  // Zero saturates to min
  EXPECT_EQ(fp8[3].val, 127);   // 1.0
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_FLOAT8_TYPES)
