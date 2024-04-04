// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include <cmath>  // NAN
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

template <typename T>
void run_is_nan_test(int opset, const std::vector<int64_t>& dims, const std::initializer_list<T>& input, const std::initializer_list<bool>& output) {
  OpTester test("IsNaN", opset, kOnnxDomain);
  test.AddInput<T>("X", dims, input);
  test.AddOutput<bool>("Y", dims, output);
  test.Run();
}

TEST(IsNaNOpTest, IsNaNFloat9) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<float> input = {1.0f, NAN, 2.0f, NAN};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(9, dims, input, output);
}

TEST(IsNaNOpTest, IsNaNFloat20) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<float> input = {1.0f, NAN, 2.0f, NAN};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(20, dims, input, output);
}

TEST(IsNaNOpTest, IsNaNFloat16_9) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<MLFloat16> input = {MLFloat16(1.0f), MLFloat16::NaN, MLFloat16(2.0f), MLFloat16::NaN};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(9, dims, input, output);
}

TEST(IsNaNOpTest, IsNaNFloat16_13) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<MLFloat16> input = {MLFloat16::One, MLFloat16::NaN, MLFloat16(2.0f), MLFloat16::NaN};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(13, dims, input, output);
}

TEST(IsNaNOpTest, IsNaNFloat16_20) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<MLFloat16> input = {MLFloat16::One, MLFloat16::NaN, MLFloat16(2.0f), MLFloat16::NaN};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(20, dims, input, output);
}

TEST(IsNaNOpTest, IsNaNBFloat16_20) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<BFloat16> input = {BFloat16::One, BFloat16::NaN, BFloat16(2.0f), BFloat16::NaN};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(20, dims, input, output);
}

TEST(IsNaNOpTest, IsNaNDouble9) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<double> input = {1.0, NAN, 2.0, NAN};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(9, dims, input, output);
}

TEST(IsNaNOpTest, IsNaNDouble20) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<double> input = {1.0, NAN, 2.0, NAN};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(20, dims, input, output);
}

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(IsNaNOpTest, IsNaNFloat8E4M3FN) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<Float8E4M3FN> input = {Float8E4M3FN(1.0f), Float8E4M3FN(-NAN), Float8E4M3FN(2.0f), Float8E4M3FN(NAN)};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(20, dims, input, output);
}

TEST(IsNaNOpTest, IsNaN_Float8E4M3FNUZ) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<Float8E4M3FNUZ> input = {Float8E4M3FNUZ(1.0f), Float8E4M3FNUZ(-NAN), Float8E4M3FNUZ(2.0f), Float8E4M3FNUZ(-NAN)};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(20, dims, input, output);
}

TEST(IsNaNOpTest, IsNaNFloat8E5M2) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<Float8E5M2> input = {Float8E5M2(1.0f), Float8E5M2(-NAN), Float8E5M2(2.0f), Float8E5M2(NAN)};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(20, dims, input, output);
}

TEST(IsNaNOpTest, IsNaN_Float8E5M2FNUZ) {
  std::vector<int64_t> dims{2, 2};
  std::initializer_list<Float8E5M2FNUZ> input = {Float8E5M2FNUZ(1.0f), Float8E5M2FNUZ(-NAN), Float8E5M2FNUZ(2.0f), Float8E5M2FNUZ(NAN)};
  std::initializer_list<bool> output = {false, true, false, true};
  run_is_nan_test(20, dims, input, output);
}
#endif
}  // namespace test
}  // namespace onnxruntime
