// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
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

template <typename T>
void TestMatMulIntegerToFloat(const std::vector<int64_t>& A_dims,
                              std::vector<int64_t> B_dims,
                              const std::string& reference_model,
                              bool is_matrix_b_constant,
                              bool per_column = false,
                              bool has_zp = true,
                              bool has_bias = false) {
  // create rand inputs
  RandomValueGenerator random{};

  std::vector<uint8_t> A_data;
  std::vector<int> tmp_A_data = random.Uniform<int32_t>(A_dims, 0, 255);
  std::transform(tmp_A_data.begin(), tmp_A_data.end(), std::back_inserter(A_data), [](int32_t v) -> T {
    return static_cast<uint8_t>(v);
  });

  std::vector<T> B_data;
  std::vector<int> tmp_B_data = random.Uniform<int32_t>(B_dims, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  std::transform(tmp_B_data.begin(), tmp_B_data.end(), std::back_inserter(B_data), [](int32_t v) -> T {
    return static_cast<T>(v);
  });

  std::vector<float> A_scale = random.Uniform<float>({1}, -0.1f, 0.1f);
  std::vector<uint8_t> A_zero_point{127};

  int64_t b_scale_zp_size = per_column ? B_dims.back() : 1;
  std::vector<float> B_scale = random.Uniform<float>({b_scale_zp_size}, -0.1f, 0.1f);

  std::vector<T> B_zero_point(b_scale_zp_size);
  std::for_each(B_zero_point.begin(),
                B_zero_point.end(),
                [&random](T& zp) {
                  zp = static_cast<T>(random.Uniform<int32_t>({1}, std::numeric_limits<T>::min(), std::numeric_limits<T>::max())[0]);
                });

  std::vector<float> Bias = random.Uniform<float>({B_dims.back()}, -0.1f, 0.1f);

  OpTester test("MatMulIntegerToFloat", 1, onnxruntime::kMSDomain);
  test.AddInput<uint8_t>("A", A_dims, A_data);
  test.AddInput<T>("B", B_dims, B_data, is_matrix_b_constant);
  test.AddInput<float>("a_scale", {1}, A_scale);
  test.AddInput<float>("b_scale", {b_scale_zp_size}, B_scale);

  if (has_zp) {
    test.AddInput<uint8_t>("a_zero_point", {1}, A_zero_point);
    test.AddInput<T>("b_zero_point", {b_scale_zp_size}, B_zero_point);
  } else {
    test.AddMissingOptionalInput<T>();
    test.AddMissingOptionalInput<T>();
  }

  if (has_bias) {
    test.AddInput<float>("bias", {B_dims.back()}, Bias);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  test.AddReferenceOutputs(reference_model);
  test.SetOutputRelErr("Y", 1e-4f);
  test.Run();
}

template <typename Scalar, bool HasZeroPoint, bool HasBias>
void RunMatMulIntegerToFloatTest(const string& model_path) {
  std::vector<int64_t> A_dims{4, 128};
  std::vector<int64_t> B_dims{128, 128};
  std::vector<int64_t> Y_dims{4, 128};

  TestMatMulIntegerToFloat<Scalar>(A_dims,
                                   B_dims,
                                   model_path,
                                   false,        /*is_matrix_b_constant*/
                                   false,        /*per_column*/
                                   HasZeroPoint, /*has_zp*/
                                   HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<Scalar>(A_dims,
                                   B_dims,
                                   model_path,
                                   true,         /*is_matrix_b_constant*/
                                   false,        /*per_column*/
                                   HasZeroPoint, /*has_zp*/
                                   HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<Scalar>(A_dims,
                                   B_dims,
                                   model_path,
                                   false,        /*is_matrix_b_constant*/
                                   true,         /*per_column*/
                                   HasZeroPoint, /*has_zp*/
                                   HasBias       /*has_bias*/
  );

  TestMatMulIntegerToFloat<Scalar>(A_dims,
                                   B_dims,
                                   model_path,
                                   true,         /*is_matrix_b_constant*/
                                   true,         /*per_column*/
                                   HasZeroPoint, /*has_zp*/
                                   HasBias       /*has_bias*/
  );
}

TEST(MatMulIntegerToFloat, HasZeroPoint_NoBias_test) {
  RunMatMulIntegerToFloatTest<int8_t, true, false>("testdata/matmul_integer_to_float_int8.onnx");
  RunMatMulIntegerToFloatTest<uint8_t, true, false>("testdata/matmul_integer_to_float_uint8.onnx");
}

TEST(MatMulIntegerToFloat, NoZeroPoint_HasBias_test) {
  RunMatMulIntegerToFloatTest<int8_t, false, true>("testdata/matmul_integer_to_float_int8_bias.onnx");
  RunMatMulIntegerToFloatTest<uint8_t, false, true>("testdata/matmul_integer_to_float_uint8_bias.onnx");
}

TEST(MatMulIntegerToFloat, MatMulInteger_Nuphar) {
  auto test_case = [&](const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& weights_shape,
                       const std::vector<int64_t>& scale_shape_1,
                       const std::vector<int64_t>& scale_shape_2) {
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
      builder.AddNode("MatMulInteger", {input_arg, weight}, {matmul_integer_output});

      // add Cast
      auto* cast_output = builder.MakeIntermediate();
      Node& cast_node = builder.AddNode("Cast", {matmul_integer_output}, {cast_output});
      cast_node.AddAttribute("to", (int64_t)1);

      // add Mul1
      auto* mul1_input1_arg = builder.MakeInput<float>(scale_shape_1, -0.1f, 0.f);
      auto* mul1_input2_arg = builder.MakeInput<float>(scale_shape_2, -0.1f, 0.f);
      auto* mul1_output = builder.MakeIntermediate();
      builder.AddNode("Mul", {mul1_input1_arg, mul1_input2_arg}, {mul1_output});

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
  test_case({5, 4, 3}, {3, 4}, {1}, {1});

  // B per-column
  test_case({5, 4, 3}, {3, 4}, {1}, {4});

  // A per-row, B per-column
  test_case({5, 4, 3}, {3, 4}, {5, 4, 1}, {4});
  test_case({5, 4, 3}, {3, 4}, {5, 4, 1}, {1, 4});
  test_case({5, 4, 3}, {3, 4}, {1, 4}, {5, 4, 1});
  test_case({5, 4, 3}, {3, 4}, {4}, {5, 4, 1});

  // A per-row, B per-column, and offset
  test_case({15, 14, 13}, {15, 13, 27}, {15, 14, 1}, {15, 1, 27});
  test_case({15, 14, 13}, {15, 13, 27}, {15, 1, 27}, {15, 14, 1});
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

}  // namespace test
}  // namespace onnxruntime
