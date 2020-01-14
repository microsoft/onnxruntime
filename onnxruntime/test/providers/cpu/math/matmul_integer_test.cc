// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"

#include <random>

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

template <typename T>
std::vector<T> ToVector(const int* value, int size) {
  std::vector<T> data(size);
  for (int i = 0; i < size; i++)
    data[i] = static_cast<T>(value[i]);
  return data;
}

// [M x N] = [M x K] x [K x N] = [batch_seq x input_dim] x [input_dim x embed_dim]
void RunMatMulIntegerU8S8Test(const int M, const int N, const int K) {
  OpTester test("MatMulInteger", 10);
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n_unsigned(0, 127);
  static std::uniform_int_distribution<int> n_signed(-128, 127);
  Eigen::MatrixXi T1 = Eigen::MatrixXi::Random(K, M)
                           .unaryExpr([](int) { return n_unsigned(e); });
  Eigen::MatrixXi T2 = Eigen::MatrixXi::Random(N, K)
                           .unaryExpr([](int) { return n_signed(e); });
  Eigen::MatrixXi T3 = (T2 * T1).eval();

  test.AddInput<uint8_t>("T1", {M, K},
                         ToVector<uint8_t>(T1.data(), M * K));
  test.AddInput<int8_t>("T2", {K, N},
                        ToVector<int8_t>(T2.data(), K * N), /*is_initializer*/ true);
  test.AddOutput<int32_t>("T3", {M, N},
                          ToVector<int32_t>(T3.data(), M * N));

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider});  // currently nGraph provider does not support gemm_u8s8
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_Scalar) {
  RunMatMulIntegerU8S8Test(1, 1, 32);
  RunMatMulIntegerU8S8Test(1, 1, 260);
  RunMatMulIntegerU8S8Test(1, 1, 288);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_GEMV) {
  RunMatMulIntegerU8S8Test(1, 2, 16);
  RunMatMulIntegerU8S8Test(1, 2, 64);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_GEMM) {
  RunMatMulIntegerU8S8Test(2, 2, 40);
  RunMatMulIntegerU8S8Test(2, 48, 33);
  RunMatMulIntegerU8S8Test(2, 51, 40);
}

}  // namespace test
}  // namespace onnxruntime
