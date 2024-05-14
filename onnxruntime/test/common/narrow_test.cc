// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/narrow.h"

#include <complex>
#include <limits>

#include "gtest/gtest.h"

// These tests were adapted from:
// https://github.com/microsoft/GSL/blob/a3534567187d2edc428efd3f13466ff75fe5805c/tests/utils_tests.cpp#L127-L152

namespace onnxruntime::test {

#if defined(ORT_NO_EXCEPTIONS)

#define NARROW_FAILURE_TEST_SUITE NarrowDeathTest
#define ASSERT_NARROW_FAILURE(expr) \
  ASSERT_DEATH((expr), "narrowing error")

#else  // ^^ defined(ORT_NO_EXCEPTIONS) ^^ / vv !defined(ORT_NO_EXCEPTIONS) vv

#define NARROW_FAILURE_TEST_SUITE NarrowTest
#define ASSERT_NARROW_FAILURE(expr) \
  ASSERT_THROW((expr), gsl::narrowing_error)

#endif  // !defined(ORT_NO_EXCEPTIONS)

TEST(NarrowTest, Basic) {
  constexpr int n = 120;
  constexpr char c = narrow<char>(n);
  EXPECT_EQ(c, 120);

  EXPECT_EQ(narrow<uint32_t>(int32_t(0)), uint32_t{0});
  EXPECT_EQ(narrow<uint32_t>(int32_t(1)), uint32_t{1});
  constexpr auto int32_max = std::numeric_limits<int32_t>::max();
  EXPECT_EQ(narrow<uint32_t>(int32_max), static_cast<uint32_t>(int32_max));

  EXPECT_EQ(narrow<std::complex<float>>(std::complex<double>(4, 2)), std::complex<float>(4, 2));
}

TEST(NARROW_FAILURE_TEST_SUITE, CharOutOfRange) {
  constexpr int n = 300;
  ASSERT_NARROW_FAILURE(narrow<char>(n));
}

TEST(NARROW_FAILURE_TEST_SUITE, MinusOneToUint32OutOfRange) {
  ASSERT_NARROW_FAILURE(narrow<uint32_t>(int32_t(-1)));
}

TEST(NARROW_FAILURE_TEST_SUITE, Int32MinToUint32OutOfRange) {
  constexpr auto int32_min = std::numeric_limits<int32_t>::min();
  ASSERT_NARROW_FAILURE(narrow<uint32_t>(int32_min));
}

TEST(NARROW_FAILURE_TEST_SUITE, UnsignedOutOfRange) {
  constexpr int n = -42;
  ASSERT_NARROW_FAILURE(narrow<unsigned>(n));
}

namespace {
constexpr double kDoubleWithLossyRoundTripFloatConversion = 4.2;
static_assert(static_cast<double>(static_cast<float>(kDoubleWithLossyRoundTripFloatConversion)) !=
              kDoubleWithLossyRoundTripFloatConversion);
}  // namespace

TEST(NARROW_FAILURE_TEST_SUITE, FloatLossyRoundTripConversion) {
  ASSERT_NARROW_FAILURE(narrow<float>(kDoubleWithLossyRoundTripFloatConversion));
}

TEST(NARROW_FAILURE_TEST_SUITE, ComplexFloatLossyRoundTripConversion) {
  ASSERT_NARROW_FAILURE(narrow<std::complex<float>>(std::complex<double>(kDoubleWithLossyRoundTripFloatConversion)));
}

}  // namespace onnxruntime::test
