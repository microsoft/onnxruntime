// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/quantization_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
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
  if (DefaultCudaExecutionProvider() &&
      !HasCudaEnvironment(530 /*min_cuda_architecture*/)) {
    return;
  }

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

  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_A_ND) {
  if (DefaultCudaExecutionProvider() &&
      !HasCudaEnvironment(530 /*min_cuda_architecture*/)) {
    return;
  }

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

  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_B_ND) {
  if (DefaultCudaExecutionProvider() &&
      !HasCudaEnvironment(530 /*min_cuda_architecture*/)) {
    return;
  }

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

  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_A_ND_B_ND) {
  if (DefaultCudaExecutionProvider() &&
      !HasCudaEnvironment(530 /*min_cuda_architecture*/)) {
    return;
  }

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

  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_A_Has_Zero_Point) {
  if (DefaultCudaExecutionProvider() &&
      !HasCudaEnvironment(530 /*min_cuda_architecture*/)) {
    return;
  }

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

  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_t_No_Zero_Point) {
  if (DefaultCudaExecutionProvider() &&
      !HasCudaEnvironment(530 /*min_cuda_architecture*/)) {
    return;
  }

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

TEST(MatmulIntegerOpTest, MatMulInteger_PerColumn_ND) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

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

TEST(MatmulIntegerOpTest, MatMulInteger_int8_uint8_2D) {
  OpTester test("MatMulInteger", 10);

  test.AddInput<int8_t>("T1",
                        {2, 3},
                        {-3, 7, 5,
                         4, -5, 8});

  test.AddInput<uint8_t>("T2",
                         {3, 2},
                         {5, 12,
                          7, 2,
                          9, 6});

  test.AddInput<int8_t>("a_zero_point", {}, {0});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});

  test.AddOutput<int32_t>("T3",
                          {2, 2},
                          {79, 8,
                           57, 86});

  test.Run();
}

TEST(MatmulIntegerOpTest, MatMulInteger_int8_uint8_PerColumn_ND) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 2, 2},
                        {1, -2,
                         3, 4,

                         -1, 5,
                         2, -3});

  test.AddInput<uint8_t>("T2",
                         {2, 2, 3},
                         {10, 20, 30,
                          1, 2, 3,

                          4, 5, 6,
                          7, 8, 9});

  test.AddInput<int8_t>("a_zero_point", {}, {0});

  test.AddInput<uint8_t>("b_zero_point",
                         {2, 1, 3},
                         {1, 2, 3,
                          4, 5, 6});

  test.AddOutput<int32_t>("T3",
                          {2, 2, 3},
                          {9, 18, 27,
                           27, 54, 81,

                           15, 15, 15,
                           -9, -9, -9});

  // TODO: Unskip when fixed #41968513
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

// [M x N] = [M x K] x [K x N] = [batch_seq x input_dim] x [input_dim x embed_dim]
template <typename WeightType>
void RunMatMulIntegerU8X8Test(const int M, const int N, const int K, bool B_is_initializer) {
  OpTester test("MatMulInteger", 10);
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n_unsigned(0, 127);
  static std::uniform_int_distribution<int> n_xint8(std::numeric_limits<WeightType>::min(), std::numeric_limits<WeightType>::max());

  Eigen::MatrixXi matrix_a = Eigen::MatrixXi::Random(K, M)
                                 .unaryExpr([](int) { return n_unsigned(e); });
  std::vector<uint8_t> matrix_a_data = ToVector<uint8_t>(matrix_a.data(), M * K);
  uint8_t a_zero_point = 0;
  Eigen::MatrixXi matrix_a_offset = matrix_a - a_zero_point * Eigen::MatrixXi::Ones(K, M);

  Eigen::MatrixXi matrix_b = Eigen::MatrixXi::Random(N, K)
                                 .unaryExpr([](int) { return n_xint8(e); });
  std::vector<WeightType> matrix_b_data = ToVector<WeightType>(matrix_b.data(), N * K);
  WeightType b_zero_point = 0;
  std::vector<WeightType> b_zp_per_column(N, b_zero_point);
  Eigen::MatrixXi b_zp_matrix = b_zero_point * Eigen::MatrixXi::Ones(N, K);
  Eigen::MatrixXi matrix_c = ((matrix_b - b_zp_matrix) * matrix_a_offset).eval();

  test.AddInput<uint8_t>("T1", {M, K}, std::move(matrix_a_data));
  test.AddInput<WeightType>("T2", {K, N}, std::move(matrix_b_data), B_is_initializer);

  test.AddOutput<int32_t>("T3", {M, N}, ToVector<int32_t>(matrix_c.data(), M * N));
  test.Run();
}

void RunMatMulIntegerU8X8TestBatch(const int M, const int N, const int K) {
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, false);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, true);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, false);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, true);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, false);
  RunMatMulIntegerU8X8Test<int8_t>(M, N, K, true);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, false);
  RunMatMulIntegerU8X8Test<uint8_t>(M, N, K, true);
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

#ifndef ENABLE_TRAINING
// Prepacking is disabled in full training build so no need to test the feature in a training build.
TEST(MatmulIntegerOpTest, SharedPrepackedWeights) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {1, 1}, {11});
  test.AddInput<uint8_t>("T2", {1, 1}, {13}, true);  // Trigger pre-packing
  test.AddInput<uint8_t>("a_zero_point", {}, {12});
  test.AddInput<uint8_t>("b_zero_point", {}, {12});
  test.AddOutput<int32_t>("T3", {1, 1}, {-1});

  std::vector<uint8_t> t2_init_values(1, 13);

  OrtValue t2;
  Tensor::InitOrtValue(DataTypeImpl::GetType<uint8_t>(), TensorShape({1, 1}),
                       t2_init_values.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), t2);

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

// Regression test: 1D vector dot product with matching K dimension should succeed.
// A=[K], B=[K] -> scalar output (dot product).
TEST(MatmulIntegerOpTest, MatMulInteger_1D_Vector_DotProduct) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {4}, {1, 2, 3, 4});
  test.AddInput<uint8_t>("T2", {4}, {5, 6, 7, 8});
  test.AddInput<uint8_t>("a_zero_point", {}, {0});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  // dot product: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
  test.AddOutput<int32_t>("T3", {}, {70});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

// Same 1D vector dot product test with int8_t types.
TEST(MatmulIntegerOpTest, MatMulInteger_1D_Vector_DotProduct_int8) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1", {3}, {1, -2, 3});
  test.AddInput<int8_t>("T2", {3}, {4, 5, -6});
  test.AddInput<int8_t>("a_zero_point", {}, {0});
  test.AddInput<int8_t>("b_zero_point", {}, {0});
  // dot product: 1*4 + (-2)*5 + 3*(-6) = 4 - 10 - 18 = -24
  test.AddOutput<int32_t>("T3", {}, {-24});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

// Regression test: 1D vectors with mismatched K dimension must fail safely.
// Covers prior invalid-shape handling for A=[K], B=[1] where K > 1.
TEST(MatmulIntegerOpTest, MatMulInteger_1D_Vector_KDimensionMismatch) {
  OpTester test("MatMulInteger", 10);
  test.AddShapeToTensorData(false);
  test.AddInput<uint8_t>("T1", {4}, {1, 1, 1, 1});
  test.AddInput<uint8_t>("T2", {1}, {5});
  test.AddInput<uint8_t>("a_zero_point", {}, {0});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {}, {0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "MatMul dimension mismatch", {kDmlExecutionProvider});
}

// Same regression test with int8_t types.
TEST(MatmulIntegerOpTest, MatMulInteger_int8_1D_Vector_KDimensionMismatch) {
  OpTester test("MatMulInteger", 10);
  test.AddShapeToTensorData(false);
  test.AddInput<int8_t>("T1", {8}, {1, 1, 1, 1, 1, 1, 1, 1});
  test.AddInput<int8_t>("T2", {1}, {5});
  test.AddInput<int8_t>("a_zero_point", {}, {0});
  test.AddInput<int8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {}, {0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "MatMul dimension mismatch", {kDmlExecutionProvider});
}

// 1D dot product: uint8 A x int8 B (mixed types).
TEST(MatmulIntegerOpTest, MatMulInteger_1D_Vector_DotProduct_uint8_int8) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<uint8_t>("T1", {3}, {10, 20, 30});
  test.AddInput<int8_t>("T2", {3}, {1, -1, 2});
  test.AddInput<uint8_t>("a_zero_point", {}, {0});
  test.AddInput<int8_t>("b_zero_point", {}, {0});
  // dot product: 10*1 + 20*(-1) + 30*2 = 10 - 20 + 60 = 50
  test.AddOutput<int32_t>("T3", {}, {50});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

// 1D dot product: int8 A x uint8 B (mixed types).
TEST(MatmulIntegerOpTest, MatMulInteger_1D_Vector_DotProduct_int8_uint8) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1", {3}, {-1, 2, -3});
  test.AddInput<uint8_t>("T2", {3}, {10, 20, 30});
  test.AddInput<int8_t>("a_zero_point", {}, {0});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  // dot product: (-1)*10 + 2*20 + (-3)*30 = -10 + 40 - 90 = -60
  test.AddOutput<int32_t>("T3", {}, {-60});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kDmlExecutionProvider});
}

// int8 A x uint8 B, 2D with zero points.
TEST(MatmulIntegerOpTest, MatMulInteger_int8_uint8_2D_WithZeroPoints) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 3},
                        {-3, 7, 5,
                         4, -5, 8});
  test.AddInput<uint8_t>("T2",
                         {3, 2},
                         {15, 22,
                          17, 12,
                          19, 16});
  test.AddInput<int8_t>("a_zero_point", {}, {2});
  test.AddInput<uint8_t>("b_zero_point", {}, {10});
  // A_offset = A - 2:  {-5, 5, 3, 2, -7, 6}
  // B_offset = B - 10: {5, 12, 7, 2, 9, 6}
  // C[0,0] = (-5)*5 + 5*7 + 3*9 = -25 + 35 + 27 = 37
  // C[0,1] = (-5)*12 + 5*2 + 3*6 = -60 + 10 + 18 = -32
  // C[1,0] = 2*5 + (-7)*7 + 6*9 = 10 - 49 + 54 = 15
  // C[1,1] = 2*12 + (-7)*2 + 6*6 = 24 - 14 + 36 = 46
  test.AddOutput<int32_t>("T3",
                          {2, 2},
                          {37, -32,
                           15, 46});
  test.Run();
}

// int8 A x uint8 B, A=ND B=ND with broadcasting.
TEST(MatmulIntegerOpTest, MatMulInteger_int8_uint8_A_ND_B_ND) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 2, 3},
                        {1, -2, 3,
                         -1, 4, -2,

                         1, -2, 3,
                         -1, 4, -2});
  test.AddInput<uint8_t>("T2",
                         {2, 3, 2},
                         {10, 20,
                          30, 40,
                          50, 60,

                          10, 20,
                          30, 40,
                          50, 60});
  test.AddInput<int8_t>("a_zero_point", {}, {0});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  // batch 0: [[1,-2,3],[−1,4,−2]] x [[10,20],[30,40],[50,60]]
  // [0,0] = 1*10+(-2)*30+3*50 = 10-60+150 = 100
  // [0,1] = 1*20+(-2)*40+3*60 = 20-80+180 = 120
  // [1,0] = (-1)*10+4*30+(-2)*50 = -10+120-100 = 10
  // [1,1] = (-1)*20+4*40+(-2)*60 = -20+160-120 = 20
  test.AddOutput<int32_t>("T3",
                          {2, 2, 2},
                          {100, 120,
                           10, 20,

                           100, 120,
                           10, 20});
  test.Run();
}

// int8 A x uint8 B, A=ND B=2D (broadcast B across batches).
TEST(MatmulIntegerOpTest, MatMulInteger_int8_uint8_A_ND_B_2D) {
  OpTester test("MatMulInteger", 10);
  test.AddInput<int8_t>("T1",
                        {2, 2, 3},
                        {1, -2, 3,
                         -1, 4, -2,

                         2, 0, -1,
                         0, 3, 1});
  test.AddInput<uint8_t>("T2",
                         {3, 2},
                         {10, 20,
                          30, 40,
                          50, 60});
  test.AddInput<int8_t>("a_zero_point", {}, {0});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  // batch 0: same as above = {100,120,10,20}
  // batch 1: [[2,0,-1],[0,3,1]] x [[10,20],[30,40],[50,60]]
  // [0,0] = 2*10+0*30+(-1)*50 = 20-50 = -30
  // [0,1] = 2*20+0*40+(-1)*60 = 40-60 = -20
  // [1,0] = 0*10+3*30+1*50 = 90+50 = 140
  // [1,1] = 0*20+3*40+1*60 = 120+60 = 180
  test.AddOutput<int32_t>("T3",
                          {2, 2, 2},
                          {100, 120,
                           10, 20,

                           -30, -20,
                           140, 180});
  test.Run();
}

// [M x N] = [M x K] x [K x N] with int8 A, parameterized on B type.
template <typename WeightType>
void RunMatMulIntegerS8X8Test(const int M, const int N, const int K, bool B_is_initializer) {
  OpTester test("MatMulInteger", 10);
  static std::default_random_engine e(456);
  static std::uniform_int_distribution<int> n_signed(-64, 63);
  static std::uniform_int_distribution<int> n_xint8(std::numeric_limits<WeightType>::min(), std::numeric_limits<WeightType>::max());

  Eigen::MatrixXi matrix_a = Eigen::MatrixXi::Random(K, M)
                                 .unaryExpr([](int) { return n_signed(e); });
  std::vector<int8_t> matrix_a_data = ToVector<int8_t>(matrix_a.data(), M * K);
  int8_t a_zero_point = 0;
  Eigen::MatrixXi matrix_a_offset = matrix_a - a_zero_point * Eigen::MatrixXi::Ones(K, M);

  Eigen::MatrixXi matrix_b = Eigen::MatrixXi::Random(N, K)
                                 .unaryExpr([](int) { return n_xint8(e); });
  std::vector<WeightType> matrix_b_data = ToVector<WeightType>(matrix_b.data(), N * K);
  WeightType b_zero_point = 0;
  Eigen::MatrixXi b_zp_matrix = b_zero_point * Eigen::MatrixXi::Ones(N, K);
  Eigen::MatrixXi matrix_c = ((matrix_b - b_zp_matrix) * matrix_a_offset).eval();

  test.AddInput<int8_t>("T1", {M, K}, std::move(matrix_a_data));
  test.AddInput<WeightType>("T2", {K, N}, std::move(matrix_b_data), B_is_initializer);

  test.AddOutput<int32_t>("T3", {M, N}, ToVector<int32_t>(matrix_c.data(), M * N));
  test.Run();
}

void RunMatMulIntegerS8X8TestBatch(const int M, const int N, const int K) {
  RunMatMulIntegerS8X8Test<int8_t>(M, N, K, false);
  RunMatMulIntegerS8X8Test<int8_t>(M, N, K, true);
  RunMatMulIntegerS8X8Test<uint8_t>(M, N, K, false);
  RunMatMulIntegerS8X8Test<uint8_t>(M, N, K, true);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Int8_X8_Scalar) {
  RunMatMulIntegerS8X8TestBatch(1, 1, 32);
  RunMatMulIntegerS8X8TestBatch(1, 1, 260);
  RunMatMulIntegerS8X8TestBatch(1, 1, 288);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Int8_X8_GEMV) {
  RunMatMulIntegerS8X8TestBatch(1, 2, 16);
  RunMatMulIntegerS8X8TestBatch(1, 2, 64);
  RunMatMulIntegerS8X8TestBatch(1, 8, 36);
  RunMatMulIntegerS8X8TestBatch(1, 8, 68);
  RunMatMulIntegerS8X8TestBatch(1, 8, 400);
}

TEST(MatmulIntegerOpTest, MatMulInteger_Int8_X8_GEMM) {
  RunMatMulIntegerS8X8TestBatch(2, 2, 40);
  RunMatMulIntegerS8X8TestBatch(2, 48, 33);
  RunMatMulIntegerS8X8TestBatch(2, 51, 40);
  RunMatMulIntegerS8X8TestBatch(4, 8, 68);
}

// Per-row A zero point is not supported by the CPU implementation.
// Verify it is rejected gracefully.
TEST(MatmulIntegerOpTest, MatMulInteger_PerRow_A_ZeroPoint_Rejected) {
  OpTester test("MatMulInteger", 10);
  test.AddShapeToTensorData(false);
  test.AddInput<uint8_t>("T1",
                         {2, 3},
                         {11, 7, 3,
                          10, 6, 2});
  test.AddInput<uint8_t>("T2",
                         {3, 2},
                         {1, 4, 2, 5, 3, 6});
  // per-row A zero point: shape {2, 1} — not scalar
  test.AddInput<uint8_t>("a_zero_point", {2, 1}, {12, 10});
  test.AddInput<uint8_t>("b_zero_point", {}, {0});
  test.AddOutput<int32_t>("T3", {2, 2}, {0, 0, 0, 0});
  test.Run(OpTester::ExpectResult::kExpectFailure, "", {kDmlExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
