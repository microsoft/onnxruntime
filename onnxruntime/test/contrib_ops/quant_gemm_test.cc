// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/quantization_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/quantization/quantization.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>
#include <random>

namespace onnxruntime {
namespace test {

template <typename ScalarB, typename ScalarOutput>
void RunQuantGemmU8X8Test(const int M,
                          const int N,
                          const int K,
                          bool is_A_trans,
                          bool is_B_trans,
                          bool has_C,
                          bool B_is_initializer,
                          bool per_column = false) {
  static std::default_random_engine e(123);
  static std::uniform_int_distribution<int> n_unsigned(0, 127);
  static std::uniform_int_distribution<int> n_xint8(std::numeric_limits<ScalarB>::min(), std::numeric_limits<ScalarB>::max());
  static std::uniform_real_distribution<float> n_apha(1.0f, 2.0f);
  static std::uniform_real_distribution<float> n_scale(0.003f, 0.004f);

  Eigen::MatrixXi matrix_a = Eigen::MatrixXi::Random(K, M)
                                 .unaryExpr([](int) { return n_unsigned(e); });
  std::vector<uint8_t> matrix_a_data;
  if (is_A_trans) {
    Eigen::MatrixXi matrix_a_trans = matrix_a.transpose().eval();
    matrix_a_data = ToVector<uint8_t>(matrix_a_trans.data(), M * K);
  } else {
    matrix_a_data = ToVector<uint8_t>(matrix_a.data(), M * K);
  }
  uint8_t a_zero_point = GetMiddle(matrix_a_data);
  Eigen::MatrixXi matrix_a_offset = matrix_a - a_zero_point * Eigen::MatrixXi::Ones(K, M);
  float a_scale = n_scale(e);

  Eigen::MatrixXi matrix_b = Eigen::MatrixXi::Random(N, K)
                                 .unaryExpr([](int) { return n_xint8(e); });
  std::vector<ScalarB> matrix_b_data;
  if (is_B_trans) {
    Eigen::MatrixXi matrix_b_trans = matrix_b.transpose().eval();
    matrix_b_data = ToVector<ScalarB>(matrix_b_trans.data(), N * K);
  } else {
    matrix_b_data = ToVector<ScalarB>(matrix_b.data(), N * K);
  }
  ScalarB b_zero_point = GetMiddle(matrix_b_data);
  std::vector<float> b_scale({n_scale(e)});
  std::vector<ScalarB> b_zp_per_column({b_zero_point});
  Eigen::MatrixXi b_zp_matrix = b_zero_point * Eigen::MatrixXi::Ones(N, K);
  Eigen::MatrixXf b_scale_matrix = b_scale[0] * Eigen::MatrixXf::Ones(N, M);
  if (per_column) {
    b_zp_per_column.resize(N);
    b_scale.resize(N);
    for (int i = 0; i < N; i++) {
      b_zp_per_column[i] = b_zero_point + i % 2 == 0 ? 1 : -1;
      b_zp_matrix.row(i).setConstant(b_zp_per_column[i]);
      b_scale[i] = n_scale(e);
      b_scale_matrix.row(i).setConstant(b_scale[i]);
    }
  }
  float alpha = n_apha(e);

  Eigen::MatrixXi matrix_c = Eigen::MatrixXi::Random(N, M)
                                 .unaryExpr([](int) { return n_xint8(e); });

  Eigen::MatrixXi matrix_int32 = (matrix_b - b_zp_matrix) * matrix_a_offset;
  if (has_C) {
    matrix_int32 = matrix_int32 + matrix_c;
  }
  Eigen::MatrixXf matrix_output = alpha * a_scale * (b_scale_matrix.cwiseProduct((matrix_int32.eval().cast<float>())));

  OpTester test("QGemm", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("transA", is_A_trans ? 1 : 0);
  test.AddAttribute<int64_t>("transB", is_B_trans ? 1 : 0);
  test.AddAttribute<float>("alpha", alpha);
  test.AddInput<uint8_t>("A", is_A_trans ? std::vector<int64_t>({K, M}) : std::vector<int64_t>({M, K}), std::move(matrix_a_data));
  test.AddInput<float>("a_scale", {}, {a_scale});
  test.AddInput<uint8_t>("a_zero_point", {}, {a_zero_point});
  test.AddInput<ScalarB>("B", is_B_trans ? std::vector<int64_t>({N, K}) : std::vector<int64_t>({K, N}), std::move(matrix_b_data), B_is_initializer);
  test.AddInput<float>("b_scale", {SafeInt<int64_t>(b_scale.size())}, b_scale);
  test.AddInput<ScalarB>("b_zero_point", {SafeInt<int64_t>(b_zp_per_column.size())}, b_zp_per_column);

  if (has_C) {
    test.AddInput<int32_t>("C", {M, N}, ToVector<int32_t>(matrix_c.data(), M * N));
  } else {
    test.AddOptionalInputEdge<int32_t>();
  }

  if constexpr (std::is_same_v<ScalarOutput, float>) {
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<uint8_t>();
    test.AddOutput<float>("Y", {M, N}, std::vector<float>(matrix_output.data(), matrix_output.data() + M * N));
  } else {
    std::vector<uint8_t> quant_output(M * N);
    quantization::Params<uint8_t> quant_param = quantization::QuantizeLinear(matrix_output.data(), quant_output.data(), M * N);
    test.AddInput<float>("y_scale", {}, {quant_param.scale});
    test.AddInput<uint8_t>("y_zero_point", {}, {quant_param.zero_point});
    test.AddOutput<uint8_t>("Y", {M, N}, quant_output);
  }

  test.Run();
}

void RunQuantGemmTest(const int M,
                      const int N,
                      const int K,
                      bool is_A_trans,
                      bool is_B_trans,
                      bool has_C,
                      bool B_is_initializer,
                      bool per_column = false) {
  RunQuantGemmU8X8Test<int8_t, float>(M, N, K, is_A_trans, is_B_trans, has_C, B_is_initializer, per_column);
  RunQuantGemmU8X8Test<int8_t, uint8_t>(M, N, K, is_A_trans, is_B_trans, has_C, B_is_initializer, per_column);
  RunQuantGemmU8X8Test<uint8_t, float>(M, N, K, is_A_trans, is_B_trans, has_C, B_is_initializer, per_column);
  RunQuantGemmU8X8Test<uint8_t, uint8_t>(M, N, K, is_A_trans, is_B_trans, has_C, B_is_initializer, per_column);
}

void RunQuantGemmTestBatch(const int M, const int N, const int K) {
  // No Trans
  RunQuantGemmTest(M, N, K,
                   false /*is_A_trans*/, false /*is_B_trans*/,
                   false /*has_C*/, false /*B_is_initializer*/,
                   false /*per_column*/);
  RunQuantGemmTest(M, N, K,
                   false /*is_A_trans*/, false /*is_B_trans*/,
                   true /*has_C*/, true /*B_is_initializer*/,
                   true /*per_column*/);

  // A Trans
  RunQuantGemmTest(M, N, K,
                   true /*is_A_trans*/, false /*is_B_trans*/,
                   false /*has_C*/, true /*B_is_initializer*/,
                   false /*per_column*/);
  RunQuantGemmTest(M, N, K,
                   true /*is_A_trans*/, false /*is_B_trans*/,
                   true /*has_C*/, false /*B_is_initializer*/,
                   true /*per_column*/);

  // B Trans
  RunQuantGemmTest(M, N, K,
                   false /*is_A_trans*/, true /*is_B_trans*/,
                   false /*has_C*/, false /*B_is_initializer*/,
                   false /*per_column*/);
  RunQuantGemmTest(M, N, K,
                   false /*is_A_trans*/, true /*is_B_trans*/,
                   true /*has_C*/, true /*B_is_initializer*/,  // B uses prepacking
                   true /*per_column*/);

  // A and B Trans
  RunQuantGemmTest(M, N, K,
                   true /*is_A_trans*/, true /*is_B_trans*/,
                   false /*has_C*/, true /*B_is_initializer*/,
                   true /*per_column*/);
  RunQuantGemmTest(M, N, K,
                   true /*is_A_trans*/, true /*is_B_trans*/,
                   true /*has_C*/, false /*B_is_initializer*/,
                   false /*per_column*/);
}

TEST(QuantGemmTest, Scalar) {
  RunQuantGemmTestBatch(1, 1, 32);
  RunQuantGemmTestBatch(1, 1, 260);
  RunQuantGemmTestBatch(1, 1, 288);
}

TEST(QuantGemmTest, GEMV) {
  RunQuantGemmTestBatch(1, 2, 16);
  RunQuantGemmTestBatch(1, 2, 64);
  RunQuantGemmTestBatch(1, 8, 36);
  RunQuantGemmTestBatch(1, 8, 68);
  RunQuantGemmTestBatch(1, 8, 400);
  RunQuantGemmTestBatch(1, 512, 1024);
}

TEST(QuantGemmTest, GEMM) {
  RunQuantGemmTestBatch(2, 2, 40);
  RunQuantGemmTestBatch(2, 48, 33);
  RunQuantGemmTestBatch(2, 51, 40);
  RunQuantGemmTestBatch(4, 8, 68);
}

}  // namespace test
}  // namespace onnxruntime
