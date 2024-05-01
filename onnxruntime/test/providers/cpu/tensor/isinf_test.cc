// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <limits>

namespace onnxruntime {
namespace test {

constexpr float FLOAT_INF = std::numeric_limits<float>::infinity();
constexpr float FLOAT_NINF = -std::numeric_limits<float>::infinity();
constexpr float FLOAT_NAN = std::numeric_limits<float>::quiet_NaN();

constexpr double DOUBLE_INF = std::numeric_limits<double>::infinity();
constexpr double DOUBLE_NINF = -std::numeric_limits<double>::infinity();
constexpr double DOUBLE_NAN = std::numeric_limits<double>::quiet_NaN();

template <typename T>
void run_is_inf_test(int opset, int64_t detect_positive, int64_t detect_negative, const std::initializer_list<T>& input, const std::initializer_list<bool>& output) {
  OpTester test("IsInf", opset);
  test.AddAttribute<int64_t>("detect_positive", detect_positive);
  test.AddAttribute<int64_t>("detect_negative", detect_negative);
  test.AddInput<T>("X", {onnxruntime::narrow<int64_t>(input.size())}, input);
  test.AddOutput<bool>("Y", {onnxruntime::narrow<int64_t>(output.size())}, output);
  test.Run();
}

TEST(IsInfTest, test_isinf_float10) {
  std::initializer_list<float> input = {-1.2f, FLOAT_NAN, FLOAT_INF, 2.8f, FLOAT_NINF, FLOAT_INF};
  std::initializer_list<bool> output = {false, false, true, false, true, true};
  run_is_inf_test(10, 1, 1, input, output);
}

TEST(IsInfTest, test_isinf_float20) {
  std::initializer_list<float> input = {-1.2f, FLOAT_NAN, FLOAT_INF, 2.8f, FLOAT_NINF, FLOAT_INF};
  std::initializer_list<bool> output = {false, false, true, false, true, true};
  run_is_inf_test(20, 1, 1, input, output);
}

TEST(IsInfTest, test_isinf_double10) {
  std::initializer_list<double> input = {-1.2, DOUBLE_NAN, DOUBLE_INF, 2.8, DOUBLE_NINF, DOUBLE_INF};
  std::initializer_list<bool> output = {false, false, true, false, true, true};
  run_is_inf_test(10, 1, 1, input, output);
}

TEST(IsInfTest, test_isinf_double20) {
  std::initializer_list<double> input = {-1.2, DOUBLE_NAN, DOUBLE_INF, 2.8, DOUBLE_NINF, DOUBLE_INF};
  std::initializer_list<bool> output = {false, false, true, false, true, true};
  run_is_inf_test(20, 1, 1, input, output);
}

TEST(IsInfTest, test_isinf_positive_float10) {
  std::initializer_list<double> input = {-1.7f, FLOAT_NAN, FLOAT_INF, 3.6f, FLOAT_NINF, FLOAT_INF};
  std::initializer_list<bool> output = {false, false, true, false, false, true};
  run_is_inf_test(10, 1, 0, input, output);
}

TEST(IsInfTest, test_isinf_positive_float20) {
  std::initializer_list<double> input = {-1.7f, FLOAT_NAN, FLOAT_INF, 3.6f, FLOAT_NINF, FLOAT_INF};
  std::initializer_list<bool> output = {false, false, true, false, false, true};
  run_is_inf_test(20, 1, 0, input, output);
}

TEST(IsInfTest, test_isinf_positive_double10) {
  std::initializer_list<double> input = {-1.7, DOUBLE_NAN, DOUBLE_INF, 3.6, DOUBLE_NINF, DOUBLE_INF};
  std::initializer_list<bool> output = {false, false, true, false, false, true};
  run_is_inf_test(10, 1, 0, input, output);
}

TEST(IsInfTest, test_isinf_positive_double20) {
  std::initializer_list<double> input = {-1.7, DOUBLE_NAN, DOUBLE_INF, 3.6, DOUBLE_NINF, DOUBLE_INF};
  std::initializer_list<bool> output = {false, false, true, false, false, true};
  run_is_inf_test(20, 1, 0, input, output);
}

TEST(IsInfTest, test_isinf_negative_float10) {
  std::initializer_list<float> input = {-1.7f, FLOAT_NAN, FLOAT_INF, 3.6f, FLOAT_NINF, FLOAT_INF};
  std::initializer_list<bool> output = {false, false, false, false, true, false};
  run_is_inf_test(10, 0, 1, input, output);
}

TEST(IsInfTest, test_isinf_negative_float20) {
  std::initializer_list<float> input = {-1.7f, FLOAT_NAN, FLOAT_INF, 3.6f, FLOAT_NINF, FLOAT_INF};
  std::initializer_list<bool> output = {false, false, false, false, true, false};
  run_is_inf_test(20, 0, 1, input, output);
}

TEST(IsInfTest, test_isinf_negative_double10) {
  std::initializer_list<double> input = {-1.7, DOUBLE_NAN, DOUBLE_INF, 3.6, DOUBLE_NINF, DOUBLE_INF};
  std::initializer_list<bool> output = {false, false, false, false, true, false};
  run_is_inf_test(10, 0, 1, input, output);
}

TEST(IsInfTest, test_isinf_negative_double20) {
  std::initializer_list<double> input = {-1.7, DOUBLE_NAN, DOUBLE_INF, 3.6, DOUBLE_NINF, DOUBLE_INF};
  std::initializer_list<bool> output = {false, false, false, false, true, false};
  run_is_inf_test(20, 0, 1, input, output);
}

TEST(IsInfTest, test_isinf_mlfloat16) {
  std::initializer_list<MLFloat16> input = {MLFloat16{-1.7f}, MLFloat16::NaN, MLFloat16::Infinity, 3.6_fp16,
                                            MLFloat16::NegativeInfinity, MLFloat16::Infinity};
  std::initializer_list<bool> output = {false, false, true, false, true, true};
  run_is_inf_test(20, 1, 1, input, output);
}

TEST(IsInfTest, test_isinf_positive_mlfloat16) {
  std::initializer_list<MLFloat16> input = {MLFloat16{-1.7f}, MLFloat16::NaN, MLFloat16::Infinity, 3.6_fp16,
                                            MLFloat16::NegativeInfinity, MLFloat16::Infinity};
  std::initializer_list<bool> output = {false, false, true, false, false, true};
  run_is_inf_test(20, 1, 0, input, output);
}

TEST(IsInfTest, test_isinf_negative_mlfloat16) {
  std::initializer_list<MLFloat16> input = {MLFloat16{-1.7f}, MLFloat16::NaN, MLFloat16::Infinity, 3.6_fp16,
                                            MLFloat16::NegativeInfinity, MLFloat16::Infinity};
  std::initializer_list<bool> output = {false, false, false, false, true, false};
  run_is_inf_test(20, 0, 1, input, output);
}

TEST(IsInfTest, test_isinf_bfloat16) {
  std::initializer_list<BFloat16> input = {BFloat16{-1.7f}, BFloat16::NaN, BFloat16::Infinity, 3.6_bfp16,
                                           BFloat16::NegativeInfinity, BFloat16::Infinity};
  std::initializer_list<bool> output = {false, false, true, false, true, true};
  run_is_inf_test(20, 1, 1, input, output);
}

TEST(IsInfTest, test_isinf_positive_bfloat16) {
  std::initializer_list<BFloat16> input = {BFloat16{-1.7f}, BFloat16::NaN, BFloat16::Infinity, 3.6_bfp16,
                                           BFloat16::NegativeInfinity, BFloat16::Infinity};
  std::initializer_list<bool> output = {false, false, true, false, false, true};
  run_is_inf_test(20, 1, 0, input, output);
}

TEST(IsInfTest, test_isinf_negative_bfloat16) {
  std::initializer_list<BFloat16> input = {BFloat16{-1.7f}, BFloat16::NaN, BFloat16::Infinity, 3.6_bfp16,
                                           BFloat16::NegativeInfinity, BFloat16::Infinity};
  std::initializer_list<bool> output = {false, false, false, false, true, false};
  run_is_inf_test(20, 0, 1, input, output);
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(IsInfTest, test_Float8E4M3FN) {
  std::initializer_list<Float8E4M3FN> input = {
      Float8E4M3FN(-1.0f), Float8E4M3FN(FLOAT_NAN, false), Float8E4M3FN(1.0f), Float8E4M3FN(FLOAT_NINF, false), Float8E4M3FN(FLOAT_NINF, false), Float8E4M3FN(FLOAT_INF, false)};
  std::initializer_list<bool> output = {false, false, false, false, false, false};
  run_is_inf_test(20, 1, 1, input, output);
}

TEST(IsInfTest, test_Float8E4M3FNUZ) {
  std::initializer_list<Float8E4M3FNUZ> input = {
      Float8E4M3FNUZ(-1.0f), Float8E4M3FNUZ(FLOAT_NAN, false), Float8E4M3FNUZ(1.0f), Float8E4M3FNUZ(FLOAT_NINF, false), Float8E4M3FNUZ(FLOAT_NINF, false), Float8E4M3FNUZ(FLOAT_INF, false)};
  std::initializer_list<bool> output = {false, false, false, false, false, false};
  run_is_inf_test(20, 1, 1, input, output);
}

TEST(IsInfTest, test_Float8E5M2_detect_both) {
  std::initializer_list<Float8E5M2> input = {
      Float8E5M2(-1.0f), Float8E5M2(FLOAT_NINF, false), Float8E5M2(1.0f), Float8E5M2(FLOAT_NINF, false), Float8E5M2(FLOAT_NAN, false), Float8E5M2(FLOAT_INF, false)};
  std::initializer_list<bool> output = {false, true, false, true, false, true};
  run_is_inf_test(20, 1, 1, input, output);
}

TEST(IsInfTest, test_Float8E5M2_detect_positive) {
  std::initializer_list<Float8E5M2> input = {
      Float8E5M2(-1.0f), Float8E5M2(FLOAT_NINF, false), Float8E5M2(1.0f), Float8E5M2(FLOAT_NINF, false), Float8E5M2(FLOAT_NAN, false), Float8E5M2(FLOAT_INF, false)};
  std::initializer_list<bool> output = {false, false, false, false, false, true};
  run_is_inf_test(20, 1, 0, input, output);
}

TEST(IsInfTest, test_Float8E5M2_detect_negative) {
  std::initializer_list<Float8E5M2> input = {
      Float8E5M2(-1.0f), Float8E5M2(FLOAT_NINF, false), Float8E5M2(1.0f), Float8E5M2(FLOAT_NINF, false), Float8E5M2(FLOAT_NAN, false), Float8E5M2(FLOAT_INF, false)};
  std::initializer_list<bool> output = {false, true, false, true, false, false};
  run_is_inf_test(20, 0, 1, input, output);
}

TEST(IsInfTest, test_Float8E5M2_none) {
  std::initializer_list<Float8E5M2> input = {
      Float8E5M2(-1.0f), Float8E5M2(FLOAT_NINF, false), Float8E5M2(1.0f), Float8E5M2(FLOAT_NINF, false), Float8E5M2(FLOAT_NAN, false), Float8E5M2(FLOAT_INF, false)};
  std::initializer_list<bool> output = {false, false, false, false, false, false};
  run_is_inf_test(20, 0, 0, input, output);
}

TEST(IsInfTest, test_Float8E5M2FNUZ) {
  std::initializer_list<Float8E5M2FNUZ> input = {
      Float8E5M2FNUZ(-1.0f), Float8E5M2FNUZ(FLOAT_NINF, false), Float8E5M2FNUZ(1.0f), Float8E5M2FNUZ(FLOAT_NINF, false), Float8E5M2FNUZ(FLOAT_NAN, false), Float8E5M2FNUZ(FLOAT_INF, false)};
  std::initializer_list<bool> output = {false, false, false, false, false, false};
  run_is_inf_test(20, 1, 1, input, output);
}
#endif
}  // namespace test
}  // namespace onnxruntime
