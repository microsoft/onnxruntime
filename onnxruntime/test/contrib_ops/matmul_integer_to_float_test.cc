// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/qmath.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;

namespace onnxruntime {
namespace test {

template <typename IType, typename WType, typename OType>
static void CalculateMatMulIntegerToFloat(const int64_t M, const int64_t N, const int64_t K,
                                          const std::vector<IType>& A_data, const std::vector<OType>& A_scale,
                                          const std::vector<IType>& A_zero_point, const std::vector<WType>& B_data,
                                          std::vector<OType>& B_scale, std::vector<WType>& B_zero_point,
                                          const std::vector<OType>& Bias, std::vector<float>& Y_data,
                                          bool per_column, bool has_zp, bool has_bias) {
  if (!per_column) {
    B_zero_point.resize(N, B_zero_point[0]);
    B_scale.resize(N, B_scale[0]);
  }

  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        float A_dequantized = has_zp ? (static_cast<int>(A_data[m * K + k]) - static_cast<int>(A_zero_point[0])) * A_scale[0] : A_data[m * K + k] * A_scale[0];
        float B_dequantized = has_zp ? (static_cast<int>(B_data[k * N + n]) - static_cast<int>(B_zero_point[n])) * B_scale[n] : B_data[k * N + n] * B_scale[n];

        sum += A_dequantized * B_dequantized;
      }
      if (has_bias) {
        sum += Bias[n];
      }
      Y_data[m * N + n] = static_cast<OType>(sum);
    }
  }
}

template <typename IType, typename WType, typename OType>
void TestMatMulIntegerToFloat(bool is_matrix_b_constant,
                              bool per_column = false,
                              bool has_zp = true,
                              bool has_bias = false) {
  // create rand inputs
  RandomValueGenerator random{};
  int64_t M = 4;
  int64_t N = 128;
  int64_t K = 128;
  std::vector<int64_t> A_dims{M, K};
  std::vector<int64_t> B_dims{K, N};
  std::vector<int64_t> Y_dims{M, K};
  std::vector<IType> A_data;
  std::vector<IType> tmp_A_data = random.Uniform<IType>(A_dims,
                                                        std::numeric_limits<IType>::lowest(),
                                                        std::numeric_limits<IType>::max());
  std::transform(tmp_A_data.begin(), tmp_A_data.end(), std::back_inserter(A_data), [](int32_t v) -> IType {
    return static_cast<IType>(v);
  });

  std::vector<WType> B_data;

  std::vector<WType> tmp_B_data;
  tmp_B_data = random.Uniform<WType>(B_dims,
                                     std::is_signed<WType>::value ? std::numeric_limits<int8_t>::lowest() / 2 : std::numeric_limits<uint8_t>::lowest(),
                                     std::numeric_limits<WType>::max() / 2);
  std::transform(tmp_B_data.begin(), tmp_B_data.end(), std::back_inserter(B_data), [](int32_t v) -> WType {
    return static_cast<WType>(v);
  });

  std::vector<OType> A_scale = random.Uniform<OType>(AsSpan<int64_t>({1}), -0.1f, 0.1f);
  std::vector<IType> A_zero_point{(std::numeric_limits<IType>::lowest() + std::numeric_limits<IType>::max() + IType(2)) / 2};

  int64_t b_scale_zp_size = per_column ? B_dims.back() : 1;
  std::vector<OType> B_scale = random.Uniform<OType>(AsSpan({b_scale_zp_size}), -0.1f, 0.1f);

  std::vector<WType> B_zero_point(b_scale_zp_size);
  std::for_each(B_zero_point.begin(),
                B_zero_point.end(),
                [&random](WType& zp) {
                  zp = static_cast<WType>(random.Uniform<WType>(std::array<int64_t, 1>{1},
                                                                std::numeric_limits<WType>::lowest(),
                                                                std::numeric_limits<WType>::max())[0]);
                });

  std::vector<OType> Bias = random.Uniform<OType>(AsSpan({B_dims.back()}), -0.1f, 0.1f);

  OpTester test("MatMulIntegerToFloat", 1, onnxruntime::kMSDomain);
  test.AddInput<IType>("A", A_dims, A_data);
  test.AddInput<WType>("B", B_dims, B_data, is_matrix_b_constant);
  test.AddInput<OType>("a_scale", {1}, A_scale);
  test.AddInput<OType>("b_scale", {b_scale_zp_size}, B_scale);

  if (has_zp) {
    test.AddInput<IType>("a_zero_point", {1}, A_zero_point);
    test.AddInput<WType>("b_zero_point", {b_scale_zp_size}, B_zero_point);
  } else {
    test.AddOptionalInputEdge<IType>();
    test.AddOptionalInputEdge<WType>();
  }

  if (has_bias) {
    test.AddInput<OType>("bias", {B_dims.back()}, Bias);
  } else {
    test.AddOptionalInputEdge<OType>();
  }

  std::vector<float> Y_data(M * N);
  CalculateMatMulIntegerToFloat<IType, WType, OType>(M, N, K, A_data, A_scale, A_zero_point,
                                                     B_data, B_scale, B_zero_point, Bias, Y_data,
                                                     per_column, has_zp, has_bias);

  if (std::is_same_v<OType, float>) {
    test.AddOutput<float>("Y", {M, N}, Y_data);
    test.SetOutputAbsErr("Y", 0.0001f);
    test.SetOutputRelErr("Y", 0.02f);
  } else {
    test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(Y_data));
    test.SetOutputAbsErr("Y", 0.5f);
  }

  // Only DML EP supports these data type combinations for now
  if (std::is_same_v<OType, MLFloat16> ||
      (std::is_same_v<OType, float> &&
       std::is_same_v<IType, int8_t> &&
       std::is_same_v<WType, uint8_t>)) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultDmlExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  } else {
    test.Run();
  }
}

template <typename IType, typename WType, typename OType, bool HasZeroPoint, bool HasBias>
void RunMatMulIntegerToFloatTest() {
  TestMatMulIntegerToFloat<IType, WType, OType>(
      false,        /*is_matrix_b_constant*/
      false,        /*per_column*/
      HasZeroPoint, /*has_zp*/
      HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<IType, WType, OType>(
      true,         /*is_matrix_b_constant*/
      false,        /*per_column*/
      HasZeroPoint, /*has_zp*/
      HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<IType, WType, OType>(
      false,        /*is_matrix_b_constant*/
      true,         /*per_column*/
      HasZeroPoint, /*has_zp*/
      HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<IType, WType, OType>(
      true,         /*is_matrix_b_constant*/
      true,         /*per_column*/
      HasZeroPoint, /*has_zp*/
      HasBias       /*has_bias*/
  );
}

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_S8S8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, true, false>();
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_S8S8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, false, true>();
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_S8S8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, false, false>();
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_S8S8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, true, true>();
}

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_U8U8) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, float, true, false>();
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_U8U8) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, float, false, true>();
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_U8U8) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, float, false, false>();
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_U8X8) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, float, true, true>();
}

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_U8S8) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t, float, true, false>();
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_U8S8) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t, float, false, true>();
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_U8S8) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t, float, false, false>();
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_U8S8) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t, float, true, true>();
}

// DML EP supports Float16 output type and Signed A Matrix and Unsigned B Matric for Float32 output
#if defined(USE_DML)

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_S8U8) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t, float, true, false>();
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_S8U8) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t, float, false, true>();
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_S8U8) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t, float, false, false>();
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_S8U8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, true, true>();
}

TEST(MatMulIntegerToFloat, MatMulIntegerToFloat_FP16_U8U8) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 5;
  int64_t N = 5;
  int64_t K = 2;

  std::vector<uint8_t> A_data = {1, 5, 2, 1, 9,
                                 1, 1, 3, 7, 2};
  std::vector<uint8_t> B_data = {3, 7, 2, 1, 1,
                                 2, 1, 9, 1, 1};
  std::vector<MLFloat16> A_scale = ToFloat16({3.0f});
  std::vector<MLFloat16> B_scale = ToFloat16({2.0f});
  test.AddInput<uint8_t>("A", {M, K}, A_data);
  test.AddInput<uint8_t>("B", {K, N}, B_data);
  std::vector<uint8_t> A_zero_point = {1};
  std::vector<uint8_t> B_zero_point = {1};

  test.AddInput<MLFloat16>("a_scale", {1}, A_scale);
  test.AddInput<MLFloat16>("b_scale", {1}, B_scale);
  test.AddInput<uint8_t>("a_zero_point", {1}, A_zero_point);
  test.AddInput<uint8_t>("b_zero_point", {1}, B_zero_point);

  std::vector<float> Y_data(M * N);
  CalculateMatMulIntegerToFloat<uint8_t, uint8_t, MLFloat16>(M, N, K, A_data, A_scale, A_zero_point,
                                                             B_data, B_scale, B_zero_point, {}, Y_data,
                                                             false, true, false);

  test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(Y_data));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultDmlExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatMulIntegerToFloat, MatMulIntegerToFloat_FP16_U8S8) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 5;
  int64_t N = 5;
  int64_t K = 2;

  std::vector<uint8_t> A_data = {3, 7, 2, 1, 1,
                                 2, 1, 9, 1, 1};
  std::vector<int8_t> B_data = {2, -1, -9, 1, 1,
                                -1, 0, -3, 1, -4};
  std::vector<MLFloat16> A_scale = ToFloat16({-4.0f});
  std::vector<MLFloat16> B_scale = ToFloat16({2.0f});
  test.AddInput<uint8_t>("A", {M, K}, A_data);
  test.AddInput<int8_t>("B", {K, N}, B_data);
  std::vector<uint8_t> A_zero_point = {1};
  std::vector<int8_t> B_zero_point = {3};
  std::vector<MLFloat16> Bias = ToFloat16({11.0f, -17.0f, 1.0f, -3.0f, 12.0f});

  test.AddInput<MLFloat16>("a_scale", {1}, A_scale);
  test.AddInput<MLFloat16>("b_scale", {1}, B_scale);
  test.AddInput<uint8_t>("a_zero_point", {1}, A_zero_point);
  test.AddInput<int8_t>("b_zero_point", {1}, B_zero_point);

  std::vector<float> Y_data(M * N);
  CalculateMatMulIntegerToFloat<uint8_t, int8_t, MLFloat16>(M, N, K, A_data, A_scale, A_zero_point,
                                                            B_data, B_scale, B_zero_point, {}, Y_data,
                                                            false, true, false);

  test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(Y_data));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultDmlExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatMulIntegerToFloat, MatMulIntegerToFloat_FP16_S8S8) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 5;
  int64_t N = 5;
  int64_t K = 2;

  std::vector<int8_t> A_data = {3, 7, -2, 1, 1,
                                2, -1, -9, 1, 1};
  std::vector<int8_t> B_data = {2, -1, -9, 1, 1,
                                -1, 0, -3, 1, -4};
  std::vector<MLFloat16> A_scale = ToFloat16({-4.0f});
  std::vector<MLFloat16> B_scale = ToFloat16({2.0f});
  test.AddInput<int8_t>("A", {M, K}, A_data);
  test.AddInput<int8_t>("B", {K, N}, B_data);
  std::vector<int8_t> A_zero_point = {-1};
  std::vector<int8_t> B_zero_point = {3};
  std::vector<MLFloat16> Bias = ToFloat16({11.0f, -17.0f, 1.0f, -3.0f, 12.0f});

  test.AddInput<MLFloat16>("a_scale", {1}, A_scale);
  test.AddInput<MLFloat16>("b_scale", {1}, B_scale);
  test.AddInput<int8_t>("a_zero_point", {1}, A_zero_point);
  test.AddInput<int8_t>("b_zero_point", {1}, B_zero_point);
  test.AddInput<MLFloat16>("bias", {N}, Bias);

  std::vector<float> Y_data(M * N);
  CalculateMatMulIntegerToFloat<int8_t, int8_t, MLFloat16>(M, N, K, A_data, A_scale, A_zero_point,
                                                           B_data, B_scale, B_zero_point, Bias, Y_data,
                                                           false, true, true);

  test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(Y_data));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultDmlExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatMulIntegerToFloat, MatMulIntegerToFloat_FP16_S8U8) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 5;
  int64_t N = 5;
  int64_t K = 2;

  std::vector<int8_t> A_data = {3, 7, -2, 1, 1,
                                2, -1, -9, 1, 1};
  std::vector<uint8_t> B_data = {3, 7, 2, 1, 1,
                                 2, 1, 9, 1, 1};
  std::vector<MLFloat16> A_scale = ToFloat16({-4.0f});
  std::vector<MLFloat16> B_scale = ToFloat16({2.0f});
  test.AddInput<int8_t>("A", {M, K}, A_data);
  test.AddInput<uint8_t>("B", {K, N}, B_data);
  std::vector<int8_t> A_zero_point = {-1};
  std::vector<uint8_t> B_zero_point = {1};
  std::vector<MLFloat16> Bias = ToFloat16({11.0f, -17.0f, 1.0f, -3.0f, 12.0f});

  test.AddInput<MLFloat16>("a_scale", {1}, A_scale);
  test.AddInput<MLFloat16>("b_scale", {1}, B_scale);
  test.AddInput<int8_t>("a_zero_point", {1}, A_zero_point);
  test.AddInput<uint8_t>("b_zero_point", {1}, B_zero_point);
  test.AddInput<MLFloat16>("bias", {N}, Bias);

  std::vector<float> Y_data(M * N);
  CalculateMatMulIntegerToFloat<int8_t, uint8_t, MLFloat16>(M, N, K, A_data, A_scale, A_zero_point,
                                                            B_data, B_scale, B_zero_point, Bias, Y_data,
                                                            false, true, true);

  test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(Y_data));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultDmlExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatMulIntegerToFloat, MatMulIntegerToFloat_FP16) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 2;
  int64_t N = 2;
  int64_t K = 3;

  std::vector<int8_t> A_data = {11, -2, 5,
                                -1, 3, 10};
  std::vector<int8_t> B_data = {-13, -2,
                                9, 55,
                                -1, 23};
  std::vector<MLFloat16> A_scale = ToFloat16({0.910f});
  std::vector<MLFloat16> B_scale = ToFloat16({1.10f, 1.123f});

  std::vector<int8_t> A_zero_point = {113};
  std::vector<int8_t> B_zero_point = {98, 71};

  std::vector<MLFloat16> Bias = ToFloat16({0.10f, 1.123f});

  test.AddInput<int8_t>("A", {M, K}, A_data);
  test.AddInput<int8_t>("B", {K, N}, B_data);

  test.AddInput<MLFloat16>("a_scale", {}, {A_scale});
  test.AddInput<MLFloat16>("b_scale", {N}, B_scale);
  test.AddInput<int8_t>("a_zero_point", {}, {A_zero_point});
  test.AddInput<int8_t>("b_zero_point", {N}, B_zero_point);
  test.AddInput<MLFloat16>("bias", {N}, Bias);

  std::vector<float> Y_data(M * N);
  CalculateMatMulIntegerToFloat<int8_t, int8_t, MLFloat16>(M, N, K, A_data, A_scale, A_zero_point,
                                                           B_data, B_scale, B_zero_point, Bias, Y_data,
                                                           true, true, true);

  test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(Y_data));
  test.SetOutputRelErr("Y", 2e-2f);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultDmlExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

TEST(MatMulIntegerToFloat, MatMulInteger_With_ZeroPoint) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& weights_shape,
                       const std::vector<int64_t>& b_scale_zp_shape) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<uint8_t>(input_shape,
                                                   std::numeric_limits<uint8_t>::min(),
                                                   std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape,
                                                     std::numeric_limits<int8_t>::min() / 2,
                                                     std::numeric_limits<int8_t>::max() / 2);

      // add MatMulInteger
      auto* matmul_integer_output = builder.MakeIntermediate();
      auto* A_zp_arg = builder.MakeInput<uint8_t>({1},
                                                  std::numeric_limits<uint8_t>::min(),
                                                  std::numeric_limits<uint8_t>::max());
      auto* B_zp_arg = builder.MakeInput<int8_t>(b_scale_zp_shape,
                                                 std::numeric_limits<int8_t>::min() / 2,
                                                 std::numeric_limits<int8_t>::max() / 2);
      builder.AddNode("MatMulInteger", {input_arg, weight, A_zp_arg, B_zp_arg}, {matmul_integer_output});

      // add Cast
      auto* cast_output = builder.MakeIntermediate();
      Node& cast_node = builder.AddNode("Cast", {matmul_integer_output}, {cast_output});
      cast_node.AddAttribute("to", (int64_t)1);

      // add Mul1
      auto* A_scale_arg = builder.MakeInput<float>({1}, -0.1f, 0.f);
      auto* B_scale_arg = builder.MakeInput<float>(b_scale_zp_shape, -0.1f, 0.f);
      auto* mul1_output = builder.MakeIntermediate();
      builder.AddNode("Mul", {A_scale_arg, B_scale_arg}, {mul1_output});

      // add Mul2
      builder.AddNode("Mul", {mul1_output, cast_output}, {output_arg});
    };

    auto check_mp_reshape_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
    };

    TransformerTester(build_test_case,
                      check_mp_reshape_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      1e-5 /*per_sample_tolerance*/,
                      1e-5 /*relative_per_sample_tolerance*/);
  };

  // Scale Scalar
  test_case({5, 4, 3}, {3, 4}, {1});

  // 2D B per-column
  test_case({5, 4, 3}, {3, 4}, {4});
  test_case({5, 4, 3}, {3, 4}, {1, 4});

  // ND B per-column
  test_case({15, 14, 13}, {15, 13, 27}, {15, 1, 27});
}

}  // namespace test
}  // namespace onnxruntime
