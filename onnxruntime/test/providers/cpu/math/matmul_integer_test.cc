// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace test {

TEST(MatmulIntegerOpTest, MatMulInteger_2D) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
  test.AddInput<uint8_t>("T2", {3, 2}, {1, 4, 2, 5, 3, 6});
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {4, 2}, {-38, -83, -44, -98, -50, -113, -56, -128});
  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {1, 1}, {11});
  test.AddInput<uint8_t>("T2", {1, 1}, {13});
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {12});
  test.AddOutput<int32_t>("T3", {1, 1}, {-1});
  test.Run();
}
TEST(MatmulIntegerOpTest, MatMulInteger_WithZero_ZeroPoint) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {4, 3}, {11, 7, 3, 10, 6, 2, 9, 5, 1, 8, 4, 0});
  test.AddInput<uint8_t>("T2", {3, 2}, {1, 4, 2, 5, 3, 6});
  test.AddInput<uint8_t>("a_zero_point", {}, {0});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {4, 2}, {34, 97, 28, 82, 22, 67, 16, 52});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime