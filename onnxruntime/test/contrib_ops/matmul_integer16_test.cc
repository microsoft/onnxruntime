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

TEST(MatmulInteger16OpTest, MatMulInteger16_Scalar) {
  OpTester test("MatMulInteger16", 1, onnxruntime::kMSDomain);
  test.AddInput<int16_t>("T1", {1, 1}, {15});
  test.AddInput<int16_t>("T2", {1, 1}, {16});
  test.AddOutput<int32_t>("T3", {1, 1}, {240});
  test.Run();
}

TEST(MatmulInteger16OpTest, MatMulInteger16_GEMV) {
  OpTester test("MatMulInteger16", 1, onnxruntime::kMSDomain);
  test.AddInput<int16_t>("T1", {1, 2}, {-7, 10});
  test.AddInput<int16_t>("T2", {2, 1}, {-8, -11});
  test.AddOutput<int32_t>("T3", {1, 1}, {-54});
  test.Run();
}

TEST(MatmulInteger16OpTest, MatMulInteger16_GEMM) {
  OpTester test("MatMulInteger16", 1, onnxruntime::kMSDomain);
  test.AddInput<int16_t>("T1", {3, 2}, {-7, 10, 10, -1113, 22, -356});
  test.AddInput<int16_t>("T2", {2, 4}, {-8, -11, 13, 14, -99, 1234, 321, -6});
  test.AddOutput<int32_t>("T3", {3, 4}, {-934, 12417, 3119, -158, 110107, -1373552, -357143, 6818, 35068, -439546, -113990, 2444});
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
void RuntMatMulInteger16Test(const int M, const int N, const int K) {
  OpTester test("MatMulInteger16", 1, onnxruntime::kMSDomain);
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n(-8192, 8191);
  Eigen::MatrixXi T1 = Eigen::MatrixXi::Random(K, M)
                           .unaryExpr([](int) { return n(e); });
  Eigen::MatrixXi T2 = Eigen::MatrixXi::Random(N, K)
                           .unaryExpr([](int) { return n(e); });
  Eigen::MatrixXi T3 = (T2 * T1).eval();

  test.AddInput<int16_t>("T1", {M, K},
                         ToVector<int16_t>(T1.data(), M * K));
  test.AddInput<int16_t>("T2", {K, N},
                         ToVector<int16_t>(T2.data(), K * N), /*is_initializer*/ true);
  test.AddOutput<int32_t>("T3", {M, N},
                          ToVector<int32_t>(T3.data(), M * N));
  test.Run();
}

TEST(MatmulInteger16OpTest, MatMulInteger16) {
  // GEMV
  RuntMatMulInteger16Test(1, 32, 64);
  RuntMatMulInteger16Test(1, 40, 80);
  RuntMatMulInteger16Test(1, 64, 512);
  RuntMatMulInteger16Test(1, 80, 530);
  // GEMM
  RuntMatMulInteger16Test(4, 32, 64);
  RuntMatMulInteger16Test(7, 40, 80);
  RuntMatMulInteger16Test(6, 64, 512);
  RuntMatMulInteger16Test(9, 80, 530);
}

}  // namespace test
}  // namespace onnxruntime