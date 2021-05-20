// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>
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

TEST(MatmulIntegerOpTest, MatMulInteger_2D_empty_input) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {0, 3}, {});
  test.AddInput<uint8_t>("T2", {3, 2}, {1, 4, 2, 5, 3, 6});
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {0, 2}, {});
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

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t) {
  if (!DefaultCudaExecutionProvider() || !HasCudaEnvironment(530 /*min_cuda_architecture*/)) return;

  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("T2",
                        {4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zero_point", {}, {5});
  test.AddInput<int8_t>("b_zero_point", {}, {5});
  test.AddOutput<int32_t>("T3",
                          {2, 4},
                          {-55, 16, 89, -44,
                           122, 154, 68, -39});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_A_ND) {
  if (!DefaultCudaExecutionProvider() || !HasCudaEnvironment(530 /*min_cuda_architecture*/)) return;

  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7,

                         7, -4, 3, 6,
                         -4, -5, 5, 7});

  test.AddInput<int8_t>("T2",
                        {4, 3},
                        {5, -3, 7,
                         8, -6, -8,
                         -3, 6, 7,
                         9, 9, -5});

  test.AddInput<int8_t>("a_zero_point", {}, {3});
  test.AddInput<int8_t>("b_zero_point", {}, {4});
  test.AddOutput<int32_t>("T3",
                          {2, 2, 3},
                          {-49, -39, 21,
                           -46, 103, 78,

                           -9, 57, 69,
                           -33, 153, 45});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_B_ND) {
  if (!DefaultCudaExecutionProvider() || !HasCudaEnvironment(530 /*min_cuda_architecture*/)) return;

  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("T2",
                        {2, 4, 3},
                        {5, -3, 7,
                         8, -6, -8,
                         -3, 6, 7,
                         9, 9, -5,

                         5, -3, 7,
                         8, -6, -8,
                         -3, 6, 7,
                         9, 9, -5});
  test.AddInput<int8_t>("a_zero_point", {}, {1});
  test.AddInput<int8_t>("b_zero_point", {}, {2});
  test.AddOutput<int32_t>("T3",
                          {2, 2, 3},
                          {-45, -61, -11,
                           -20, 103, 68,

                           -45, -61, -11,
                           -20, 103, 68});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_A_ND_B_ND) {
  if (!DefaultCudaExecutionProvider() || !HasCudaEnvironment(530 /*min_cuda_architecture*/)) return;

  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7,

                         -3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("T2",
                        {2, 4, 4},
                        {5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7,

                         5, -3, 7, 8,
                         -6, -8, -3, 6,
                         7, 9, 9, -5,
                         8, 7, -6, 7});
  test.AddInput<int8_t>("a_zero_point", {}, {5});
  test.AddInput<int8_t>("b_zero_point", {}, {5});
  test.AddOutput<int32_t>("T3",
                          {2, 2, 4},
                          {-55, 16, 89, -44,
                           122, 154, 68, -39,

                           -55, 16, 89, -44,
                           122, 154, 68, -39});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_A_Has_Zero_Point) {
  if (!DefaultCudaExecutionProvider() || !HasCudaEnvironment(530 /*min_cuda_architecture*/)) return;

  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 2, 4},
                        {-3, 7, 5, -6,
                         4, -5, 8, 7,

                         -3, 7, 5, -6,
                         4, -5, 8, 7});
  test.AddInput<int8_t>("T2",
                        {2, 4, 4},
                        {0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2,

                         0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2});
  test.AddInput<int8_t>("a_zero_point", {}, {5});
  test.AddOutput<int32_t>("T3",
                          {2, 2, 4},
                          {-55, 16, 89, -44,
                           122, 154, 68, -39,

                           -55, 16, 89, -44,
                           122, 154, 68, -39});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_No_Zero_Point) {
  if (!DefaultCudaExecutionProvider() || !HasCudaEnvironment(530 /*min_cuda_architecture*/)) return;

  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 2, 4},
                        {-8, 2, 0, -11,
                         -1, -10, 3, 2,

                         -8, 2, 0, -11,
                         -1, -10, 3, 2});
  test.AddInput<int8_t>("T2",
                        {2, 4, 4},
                        {0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2,

                         0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2});
  test.AddOutput<int32_t>("T3",
                          {2, 2, 4},
                          {-55, 16, 89, -44,
                           122, 154, 68, -39,

                           -55, 16, 89, -44,
                           122, 154, 68, -39});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
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

TEST(MatmulIntegerOpTest, MatMulInteger_PerColumn_ND) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1",
                         {2, 2, 4},
                         {125, 135, 133, 122,
                          132, 123, 136, 135,

                          125, 135, 133, 122,
                          132, 123, 136, 135});
  test.AddInput<int8_t>("T2",
                        {2, 4, 4},
                        {0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2,

                         0, -8, 2, 3,
                         -11, -13, -8, 1,
                         2, 4, 4, -10,
                         3, 2, -11, 2});
  test.AddInput<uint8_t>("a_zero_point", {}, {133});
  test.AddInput<int8_t>("b_zero_point",
                        {2, 1, 4},
                        {1, -2, 2, -1,
                         2, -4, -1, 0});
  test.AddOutput<int32_t>("T3",
                          {2, 2, 4},
                          {-38, -18, 123, -61,
                           128, 142, 80, -45,

                           -21, -52, 72, -44,
                           134, 130, 62, -39});

  test.Run();
}

template <typename T>
std::vector<T> ToVector(const int* value, int size) {
  std::vector<T> data(size);
  for (int i = 0; i < size; i++) {
    data[i] = static_cast<T>(value[i]);
  }
  return data;
}

template <typename T>
T GetMiddle(const std::vector<T>& v) {
  const auto min_max_pair = std::minmax_element(v.begin(), v.end());
  return (*(min_max_pair.first) + *(min_max_pair.second)) / 2;
}

// [M x N] = [M x K] x [K x N] = [batch_seq x input_dim] x [input_dim x embed_dim]
template <typename ScalarB>
void RunMatMulIntegerU8X8Test(const int M, const int N, const int K, bool non_zero_zp, bool B_is_initializer, bool per_column_zp = false) {
  OpTester test("MatMulInteger", 10);
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n_unsigned(0, 127);
  static std::uniform_int_distribution<int> n_xint8(std::numeric_limits<ScalarB>::min(), std::numeric_limits<ScalarB>::max());

  Eigen::MatrixXi matrix_a = Eigen::MatrixXi::Random(K, M)
                                 .unaryExpr([](int) { return n_unsigned(e); });
  std::vector<uint8_t> matrix_a_data = ToVector<uint8_t>(matrix_a.data(), M * K);
  uint8_t a_zero_point = non_zero_zp ? GetMiddle(matrix_a_data) : 0;
  Eigen::MatrixXi matrix_a_offset = matrix_a - a_zero_point * Eigen::MatrixXi::Ones(K, M);

  Eigen::MatrixXi matrix_b = Eigen::MatrixXi::Random(N, K)
                                 .unaryExpr([](int) { return n_xint8(e); });
  std::vector<ScalarB> matrix_b_data = ToVector<ScalarB>(matrix_b.data(), N * K);
  ScalarB b_zero_point = non_zero_zp ? GetMiddle(matrix_b_data) : 0;
  std::vector<ScalarB> b_zp_per_column(N, b_zero_point);
  Eigen::MatrixXi b_zp_matrix = b_zero_point * Eigen::MatrixXi::Ones(N, K);
  if (non_zero_zp && per_column_zp) {
    for (int i = 0; i < N; i++) {
      b_zp_per_column[i] += i % 2 == 0 ? 1 : -1;
      b_zp_matrix.row(i).setConstant(b_zp_per_column[i]);
    }
  }

  Eigen::MatrixXi matrix_c = ((matrix_b - b_zp_matrix) * matrix_a_offset).eval();

  test.AddInput<uint8_t>("T1", {M, K}, std::move(matrix_a_data));
  test.AddInput<ScalarB>("T2", {K, N}, std::move(matrix_b_data), B_is_initializer);
  if (non_zero_zp) {
    test.AddInput<uint8_t>("a_zero_point", {}, {a_zero_point});
    if (per_column_zp) {
      test.AddInput<ScalarB>("b_zero_point", {N}, b_zp_per_column);
    } else {
      test.AddInput<ScalarB>("b_zero_point", {}, {b_zero_point});
    }
  }

  test.AddOutput<int32_t>("T3", {M, N}, ToVector<int32_t>(matrix_c.data(), M * N));

  // Nuphar provider does not support non-zero zero point
  if (non_zero_zp) {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNupharExecutionProvider});
  } else {
    test.Run();
  }
}

void RunMatMulIntegerU8X8TestBatch(const int M, const int N, const int K) {
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, false /*non_zero_zp*/, false /*B_is_initializer*/, false /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, false /*non_zero_zp*/, true /*B_is_initializer*/, false /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, true /*non_zero_zp*/, false /*B_is_initializer*/, false /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, true /*non_zero_zp*/, true /*B_is_initializer*/, false /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, false /*non_zero_zp*/, false /*B_is_initializer*/, false /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, false /*non_zero_zp*/, true /*B_is_initializer*/, false /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, true /*non_zero_zp*/, false /*B_is_initializer*/, false /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, true /*non_zero_zp*/, true /*B_is_initializer*/, false /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, false /*non_zero_zp*/, false /*B_is_initializer*/, true /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, false /*non_zero_zp*/, true /*B_is_initializer*/, true /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, true /*non_zero_zp*/, false /*B_is_initializer*/, true /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, true /*non_zero_zp*/, true /*B_is_initializer*/, true /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, false /*non_zero_zp*/, false /*B_is_initializer*/, true /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, false /*non_zero_zp*/, true /*B_is_initializer*/, true /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, true /*non_zero_zp*/, false /*B_is_initializer*/, true /*per_column_zp*/);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, true /*non_zero_zp*/, true /*B_is_initializer*/, true /*per_column_zp*/);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_Scalar) {
  RunMatMulIntegerU8X8TestBatch(1, 1, 32);
  RunMatMulIntegerU8X8TestBatch(1, 1, 260);
  RunMatMulIntegerU8X8TestBatch(1, 1, 288);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_GEMV) {
  RunMatMulIntegerU8X8TestBatch(1, 2, 16);
  RunMatMulIntegerU8X8TestBatch(1, 2, 64);
  RunMatMulIntegerU8X8TestBatch(1, 8, 36);
  RunMatMulIntegerU8X8TestBatch(1, 8, 68);
  RunMatMulIntegerU8X8TestBatch(1, 8, 400);
  RunMatMulIntegerU8X8TestBatch(1, 512, 1024);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Uint8_Int8_GEMM) {
  RunMatMulIntegerU8X8TestBatch(2, 2, 40);
  RunMatMulIntegerU8X8TestBatch(2, 48, 33);
  RunMatMulIntegerU8X8TestBatch(2, 51, 40);
  RunMatMulIntegerU8X8TestBatch(4, 8, 68);
}

#ifndef ENABLE_TRAINING  // Prepacking is enabled only on non-training builds
TEST(MatmulIntegerOpTest, SharedPrepackedWeights) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {1, 1}, {11});
  test.AddInput<uint8_t>("T2", {1, 1}, {13}, true);  // Trigger pre-packing
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {12});
  test.AddOutput<int32_t>("T3", {1, 1}, {-1});

  std::vector<uint8_t> t2_init_values(1, 13);

  auto p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<uint8_t>(), TensorShape({1, 1}),
                                           t2_init_values.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
  OrtValue t2;

  t2.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(),
          DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionOptions so;
  // Set up T2 as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("T2", &t2), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {}, &number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container =
      test.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {}, &number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
