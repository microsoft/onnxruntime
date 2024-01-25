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
void TestMatMulIntegerToFloat(const std::vector<int64_t>& A_dims,
                              std::vector<int64_t> B_dims,
                              const std::string& reference_model,
                              bool is_matrix_b_constant,
                              bool per_column = false,
                              bool has_zp = true,
                              bool has_bias = false) {
  // create rand inputs
  RandomValueGenerator random{2502124740};
  per_column = reference_model.length() < 0;

  std::vector<IType> A_data;
  std::vector<IType> tmp_A_data = random.Uniform<IType>(A_dims,
                                                        std::numeric_limits<IType>::lowest(),
                                                        std::numeric_limits<IType>::max());
  std::transform(tmp_A_data.begin(), tmp_A_data.end(), std::back_inserter(A_data), [](int32_t v) -> IType {
    //v = 1;
    return static_cast<IType>(v);
  });

  std::vector<WType> B_data;

//#if defined(USE_DML)
//  std::vector<int> tmp_B_data = random.Uniform<int32_t>(B_dims,
//                                                        (constexpr(std::is_same_v<WType, int8_t>) ? -2 : 1),
//                                                        5);
//#else
  std::vector<WType> tmp_B_data = random.Uniform<WType>(B_dims,
                                                        std::numeric_limits<WType>::lowest(),
                                                        std::numeric_limits<WType>::max());
//#endif

  std::transform(tmp_B_data.begin(), tmp_B_data.end(), std::back_inserter(B_data), [](int32_t v) -> WType {
    //v = 1;
      return static_cast<WType>(v);
  });

  //std::vector<OType> A_scale = random.Uniform<OType>(AsSpan<int64_t>({1}), -0.1f, 0.1f);
  std::vector<OType> A_scale(1, static_cast<OType>(1.0f));
  std::vector<IType> A_zero_point{(std::numeric_limits<IType>::lowest() + std::numeric_limits<IType>::max() + IType(2)) / 2};

  int64_t b_scale_zp_size = per_column ? B_dims.back() : 1;
  //int64_t b_scale_zp_size = B_dims.back();
  std::vector<OType> B_scale = random.Uniform<OType>(AsSpan({b_scale_zp_size}), static_cast<OType>(-0.1f), static_cast<OType>(0.1f));
  //std::vector<OType> B_scale (b_scale_zp_size, static_cast<OType>(1.0f));

  //std::vector<WType> B_zero_point(b_scale_zp_size, 1);

  std::vector<WType> B_zero_point(b_scale_zp_size);
  if (has_zp) {
    std::for_each(B_zero_point.begin(),
                  B_zero_point.end(),
                  [&random](WType& zp) {
                    zp = static_cast<WType>(random.Uniform<WType>(std::array<int64_t, 1>{1},
                                                                  std::numeric_limits<WType>::lowest(),
                                                                  std::numeric_limits<WType>::max() / 2)[0]);
                  });
  } else {
    B_zero_point = {0};
  }

  //std::vector<OType> Bias = random.Uniform<OType>(AsSpan({B_dims.back()}), -0.1f, 0.1f);
  std::vector<OType> Bias(B_dims.back(), static_cast<OType>(0.0f));

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
  int64_t M = 10;
  int64_t N = 10;
  int64_t K = 10;
  std::vector<float> expected_vals(M * N);

  //if (constexpr(std::is_same_v<OType, float>))
  //{
    for (int64_t m = 0; m < M; m++) {
      for (int64_t n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
          float AIntermediate = has_zp ? (A_data[m * K + k] - A_zero_point[0]) : A_data[m * K + k];
          float BIntermediate = has_zp ? (B_data[k * N + n] - B_zero_point[0]) : B_data[k * N + n];
          sum += (AIntermediate * A_scale[0]) * (BIntermediate * B_scale[0]);
        }
        if (has_bias) {
          // sum += Bias[m * N + n];
          sum += Bias[n];
        }
        expected_vals[m * N + n] = static_cast<OType>(sum);
      }
    }
    if (constexpr(std::is_same_v<OType, float>)) {
      test.AddOutput<float>("Y", {M, N}, expected_vals);
    } else {
    test.AddOutput<MLFloat16>("Y", {M, N}, ToFloat16(expected_vals));
    }
  //} else {
  //  MLFloat16 AZP = static_cast<MLFloat16>(A_zero_point[0]);
  //  MLFloat16 BZP = static_cast<MLFloat16>(B_zero_point[0]);
  //  for (int64_t m = 0; m < M; m++) {
  //    for (int64_t n = 0; n < N; n++) {
  //      MLFloat16 sum = static_cast<MLFloat16>(0.0f);
  //      for (int64_t k = 0; k < K; k++) {
  //        MLFloat16 AIntermediate = (has_zp ? (A_data[m * K + k] - AZP) : A_data[m * K + k]);
  //        MLFloat16 BIntermediate = (has_zp ? (B_data[k * N + n] - BZP) : B_data[k * N + n]);
  //        sum += (AIntermediate * A_scale[0]) * (BIntermediate * B_scale[0]);
  //      }
  //      if (has_bias) {
  //        // sum += Bias[m * N + n];
  //        sum += static_cast<MLFloat16>(Bias[n]);
  //      }
  //      expected_vals[m * N + n] = static_cast<OType>(sum);
  //    }
  //  }
  //  test.AddOutput<MLFloat16>("Y", {M, N}, expected_vals);
  //}

  //test.AddReferenceOutputs(reference_model);
//#if defined(USE_DML)
//  if constexpr (std::is_same_v<OType, float>) {
//    test.SetOutputRelErr("Y", 2e-2f);
//  } else {
//    //test.SetOutputRelErr("Y", 1.0f);
//    test.SetOutputAbsErr("Y", 0.5f);
//    //test.SetOutputRelErr("Y", 2e-2f);
//  }
//#else
//  test.SetOutputRelErr("Y", 1e-4f);
//#endif

  if (constexpr(std::is_same_v<OType, float>) && constexpr(std::is_same_v<IType, uint8_t>) && constexpr(std::is_same_v<WType, uint8_t>)) {
    test.Run();
  } else {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
  }
}

template <typename IType, typename WType, typename OType, bool HasZeroPoint, bool HasBias>
void RunMatMulIntegerToFloatTest(const string& model_path) {
  std::vector<int64_t> A_dims{10, 10};
  std::vector<int64_t> B_dims{10, 10};
  std::vector<int64_t> Y_dims{10, 10};

  TestMatMulIntegerToFloat<IType, WType, OType>(
      A_dims,
      B_dims,
      model_path,
      false,        /*is_matrix_b_constant*/
      false,        /*per_column*/
      HasZeroPoint, /*has_zp*/
      HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<IType, WType, OType>(
      A_dims,
      B_dims,
      model_path,
      true,         /*is_matrix_b_constant*/
      false,        /*per_column*/
      HasZeroPoint, /*has_zp*/
      HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<IType, WType, OType>(
      A_dims,
      B_dims,
      model_path,
      false,        /*is_matrix_b_constant*/
      true,         /*per_column*/
      HasZeroPoint, /*has_zp*/
      HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<IType, WType, OType>(
      A_dims,
      B_dims,
      model_path,
      true,         /*is_matrix_b_constant*/
      true,         /*per_column*/
      HasZeroPoint, /*has_zp*/
      HasBias       /*has_bias*/
  );
}

#if USE_DML
//TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_U8X8_FP16) {
//  RunMatMulIntegerToFloatTest<uint8_t, int8_t, MLFloat16, true, false>("testdata/matmul_integer_to_float16_int8.onnx");
//  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, MLFloat16, true, false>("testdata/matmul_integer_to_float16_uint8.onnx");
//}
//
//TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_U8X8_FP16) {
//  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, MLFloat16, false, false>("testdata/matmul_integer_to_float16_uint8.onnx");
//}
//
//TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_U8X8_FP16) {
//  RunMatMulIntegerToFloatTest<uint8_t, int8_t, MLFloat16, false, true>("testdata/matmul_integer_to_float16_int8_bias.onnx");
//  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, MLFloat16, false, true>("testdata/matmul_integer_to_float16_uint8_bias.onnx");
//}
//
//TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_U8X8_FP16) {
//  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, MLFloat16, true, true>("testdata/matmul_integer_to_float16_uint8_bias.onnx");
//}
//
//TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_S8S8_FP16) {
//  RunMatMulIntegerToFloatTest<int8_t, int8_t, MLFloat16, true, false>("testdata/matmul_integer_to_float16_int8_int8.onnx");
//}
//
//TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_S8S8_FP16) {
//  RunMatMulIntegerToFloatTest<int8_t, int8_t, MLFloat16, false, true>("testdata/matmul_integer_to_float16_int8_int8_bias.onnx");
//}
//
//TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_S8S8_FP16) {
//  RunMatMulIntegerToFloatTest<int8_t, int8_t, MLFloat16, true, true>("testdata/matmul_integer_to_float16_int8_int8_bias.onnx");
//}
//
//TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_S8S8_FP16) {
//  RunMatMulIntegerToFloatTest<int8_t, int8_t, MLFloat16, false, false>("testdata/matmul_integer_to_float16_int8_int8.onnx");
//}
#endif  // USE_DML

#if USE_DML

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_U8S8_FP16) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t , MLFloat16, true, false>("testdata/matmul_integer_to_float16_int8.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_U8S8_FP16) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t , MLFloat16, false, true>("testdata/matmul_integer_to_float16_int8_bias.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_U8S8_FP16) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t , MLFloat16, false, false>("testdata/matmul_integer_to_float16_uint8.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_U8S8_FP16) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t , MLFloat16, true, true>("testdata/matmul_integer_to_float16_uint8_bias.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_S8S8_FP16) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t , MLFloat16, true, false>("testdata/matmul_integer_to_float16_int8_int8.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_S8U8_FP16) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t , MLFloat16, true, false>("testdata/matmul_integer_to_float16_int8_int8.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_S8S8_FP16) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t , MLFloat16, false, true>("testdata/matmul_integer_to_float16_int8_int8_bias.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_S8U8_FP16) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t , MLFloat16, false, true>("testdata/matmul_integer_to_float16_int8_int8_bias.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_S8S8_FP16) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t , MLFloat16, false, false>("testdata/matmul_integer_to_float16_int8_int8.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_S8U8_FP16) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t , MLFloat16, false, false>("testdata/matmul_integer_to_float16_int8_int8.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_S8S8_FP16) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t , MLFloat16, true, true>("testdata/matmul_integer_to_float16_int8_int8_bias.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_S8U8_FP16) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t , MLFloat16, true, true>("testdata/matmul_integer_to_float16_int8_int8_bias.onnx");
}

#endif

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_U8U8_FP16) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t , MLFloat16, true, false>("testdata/matmul_integer_to_float16_uint8.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_U8U8_FP16) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t , MLFloat16, false, true>("testdata/matmul_integer_to_float16_uint8_bias.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_U8U8_FP16) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t , MLFloat16, false, false>("testdata/matmul_integer_to_float16_uint8.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_U8X8_FP16) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t , MLFloat16, true, true>("testdata/matmul_integer_to_float16_uint8_bias.onnx");
}




















#if USE_DML

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_U8S8) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t, float, true, false>("testdata/matmul_integer_to_float_int8.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_U8S8) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t, float, false, true>("testdata/matmul_integer_to_float_int8_bias.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_U8S8) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t, float, false, false>("testdata/matmul_integer_to_float_uint8.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_U8S8) {
  RunMatMulIntegerToFloatTest<uint8_t, int8_t, float, true, true>("testdata/matmul_integer_to_float_uint8_bias.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_S8S8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, true, false>("testdata/matmul_integer_to_float_int8_int8.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_S8U8) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t, float, true, false>("testdata/matmul_integer_to_float_int8_int8.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_S8S8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, false, true>("testdata/matmul_integer_to_float_int8_int8_bias.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_S8U8) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t, float, false, true>("testdata/matmul_integer_to_float_int8_int8_bias.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_S8S8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, false, false>("testdata/matmul_integer_to_float_int8_int8.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_S8U8) {
  RunMatMulIntegerToFloatTest<int8_t, uint8_t, float, false, false>("testdata/matmul_integer_to_float_int8_int8.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_S8S8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, true, true>("testdata/matmul_integer_to_float_int8_int8_bias.onnx");
}

TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_S8U8) {
  RunMatMulIntegerToFloatTest<int8_t, int8_t, float, true, true>("testdata/matmul_integer_to_float_int8_int8_bias.onnx");
}

#endif

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test_U8U8) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, float, true, false>("testdata/matmul_integer_to_float_uint8.onnx");
}


TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test_U8U8) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, float, false, true>("testdata/matmul_integer_to_float_uint8_bias.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_NoBias_test_U8U8) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, float, false, false>("testdata/matmul_integer_to_float_uint8.onnx");
}


TEST(MatMulIntegerToFloat, HasZeroPoint_HasBias_test_U8X8) {
  RunMatMulIntegerToFloatTest<uint8_t, uint8_t, float, true, true>("testdata/matmul_integer_to_float_uint8_bias.onnx");
}

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

TEST(MatMulIntegerToFloat, CustomMatMul) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 2;
  int64_t N = 2;
  int64_t K = 2;

  std::vector<uint8_t> AMatrix = {1, 1,
                              1, 1};
  std::vector<uint8_t> BMatrix = {1, 1,
                              1, 1};
  test.AddInput<uint8_t>("A", {M,K}, AMatrix);
  test.AddInput<uint8_t>("B", {N,K}, BMatrix);

  test.AddInput<float>("a_scale", {}, {1.0f});
  test.AddInput<float>("b_scale", {}, {1.0f});
  //test.AddInput<uint8_t>("a_zero_point", {}, {113});

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += AMatrix[m * K + k] * BMatrix[k * N + n];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddOutput<float>("Y", {M , N}, expected_vals);

  test.Run();
}

TEST(MatMulIntegerToFloat, CustomZPMatMul) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 2;
  int64_t N = 2;
  int64_t K = 2;

  std::vector<uint8_t> AMatrix = {1, 1,
                                  1, 1};
  std::vector<int8_t> BMatrix = {1, -1,
                                  1, 1};
  float AScale = 1.0f;
  float BScale = 1.0f;

  uint8_t AZP = 113;
  int8_t BZP = -16;

  test.AddInput<uint8_t>("A", {M, K}, AMatrix);
  test.AddInput<int8_t>("B", {N, K}, BMatrix);

  test.AddInput<float>("a_scale", {}, {AScale});
  test.AddInput<float>("b_scale", {}, {BScale});
  test.AddInput<uint8_t>("a_zero_point", {}, {AZP});
  test.AddInput<int8_t>("b_zero_point", {}, {BZP});

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += ((AMatrix[m * K + k] - AZP) * AScale) * ((BMatrix[k * N + n] - BZP) * BScale);
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

TEST(MatMulIntegerToFloat, CustomScaleMatMul) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 2;
  int64_t N = 2;
  int64_t K = 2;

  std::vector<uint8_t> AMatrix = {1, 1,
                                  1, 1};
  std::vector<uint8_t> BMatrix = {1, 1,
                                  1, 1};
  float AScale = 0.910f;
  float BScale = 1.10f;

  uint8_t AZP = 1;
  uint8_t BZP= 1;

  test.AddInput<uint8_t>("A", {M, K}, AMatrix);
  test.AddInput<uint8_t>("B", {N, K}, BMatrix);

  test.AddInput<float>("a_scale", {}, {AScale});
  test.AddInput<float>("b_scale", {}, {BScale});
  test.AddInput<uint8_t>("a_zero_point", {}, {AZP});
  test.AddInput<uint8_t>("b_zero_point", {}, {BZP});

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += ((AMatrix[m * K + k] - AZP) * AScale) * ((BMatrix[k * N + n] - BZP) * BScale);
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

TEST(MatMulIntegerToFloat, CustomMatMul1) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 2;
  int64_t N = 2;
  int64_t K = 2;

  std::vector<int8_t> AMatrix = {11, -2,
                                  -1, 3};
  std::vector<int8_t> BMatrix = {-13, -2,
                                 -1, 23};
  float AScale = 0.910f;
  float BScale = 1.10f;

  int8_t AZP = 113;
  int8_t BZP = 98;

  test.AddInput<int8_t>("A", {M, K}, AMatrix);
  test.AddInput<int8_t>("B", {N, K}, BMatrix);

  test.AddInput<float>("a_scale", {}, {AScale});
  test.AddInput<float>("b_scale", {}, {BScale});
  test.AddInput<int8_t>("a_zero_point", {}, {AZP});
  test.AddInput<int8_t>("b_zero_point", {}, {BZP});

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += ((AMatrix[m * K + k] - AZP) * AScale) * ((BMatrix[k * N + n] - BZP) * BScale);
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

TEST(MatMulIntegerToFloat, CustomMatMul2) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 2;
  int64_t N = 2;
  int64_t K = 2;

  std::vector<int8_t> AMatrix = {11, -2,
                                 -1, 3};
  std::vector<int8_t> BMatrix = {-13, -2,
                                 -1, 23};
  float AScale = 0.910f;
  std::vector<float> BScale = {1.10f, 1.123f};

  int8_t AZP = 113;
  std::vector<int8_t> BZP = {98, 71};

  test.AddInput<int8_t>("A", {M, K}, AMatrix);
  test.AddInput<int8_t>("B", {K, N}, BMatrix);

  test.AddInput<float>("a_scale", {}, {AScale});
  test.AddInput<float>("b_scale", {N}, BScale);
  test.AddInput<int8_t>("a_zero_point", {}, {AZP});
  test.AddInput<int8_t>("b_zero_point", {N}, BZP);

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += ((AMatrix[m * K + k] - AZP) * AScale) * ((BMatrix[k * N + n] - BZP[n]) * BScale[n]);
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

TEST(MatMulIntegerToFloat, CustomBiasMatMul) {
  OpTester test("MatMulIntegerToFloat", 1, kMSDomain);
  int64_t M = 2;
  int64_t N = 2;
  int64_t K = 3;

  std::vector<int8_t> AMatrix = {11, -2, 5,
                                 -1, 3, 10};
  std::vector<int8_t> BMatrix = {-13, -2,
                                 9, 55,
                                 -1, 23};
  float AScale = 0.910f;
  std::vector<float> BScale = {1.10f, 1.123f};

  int8_t AZP = 113;
  std::vector<int8_t> BZP = {98, 71};

  std::vector<float> Bias = {0.10f, 1.123f};

  test.AddInput<int8_t>("A", {M, K}, AMatrix);
  test.AddInput<int8_t>("B", {K, N}, BMatrix);

  test.AddInput<float>("a_scale", {}, {AScale});
  test.AddInput<float>("b_scale", {N}, BScale);
  test.AddInput<int8_t>("a_zero_point", {}, {AZP});
  test.AddInput<int8_t>("b_zero_point", {N}, BZP);
  test.AddInput<float>("bias", {N}, Bias);

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += ((AMatrix[m * K + k] - AZP) * AScale) * ((BMatrix[k * N + n] - BZP[n]) * BScale[n]);
      }
      expected_vals[m * N + n] = sum + Bias[n];
    }
  }

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
