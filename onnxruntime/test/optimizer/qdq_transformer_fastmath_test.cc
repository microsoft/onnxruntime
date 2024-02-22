// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// Licensed under the MIT License.

#include "core/framework/compute_capability.h"
#include "core/graph/model.h"
#include "core/graph/onnx_protobuf.h"
#include "core/mlas/inc/mlas.h"
#include "core/optimizer/qdq_transformer/qdq_final_cleanup.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/optimizer/utils.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"

#include "test/compare_ortvalue.h"
#include "test/test_environment.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"

#include "qdq_test_utils.h"

#if defined(__aarch64__) && defined(__linux__) && !defined(DISABLE_CONTRIB_OPS)

struct QDQOpKeys {
  const char* quantize_linear;
  const char* dequantize_linear;
};

constexpr QDQOpKeys GetQDQOpKeys(bool use_contrib_qdq) {
  if (use_contrib_qdq) {
    return {"com.microsoft.QuantizeLinear", "com.microsoft.DequantizeLinear"};
  }
  return {"QuantizeLinear", "DequantizeLinear"};
}

namespace onnxruntime {
namespace test {

#if !defined(DISABLE_CONTRIB_OPS)

TEST(QDQTransformerTests, DQ_S8_to_U8_FastMath) {
  auto test_case = [](bool use_contrib_qdq) {
    const std::vector<int64_t>& input_shape = {19, 37};
    const std::vector<int64_t>& weights_shape = {37, 23};

    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input_shape, -1.f, 1.f);

      // Use full range weight values to expose u8s8 overflow problems
      auto* weight = builder.MakeInitializer<int8_t>(weights_shape, -128, 127);
      auto* output_arg = builder.MakeOutput();

      // add QDQ activation
      typedef std::numeric_limits<uint8_t> Input1Limits;
      auto* dq1_output = AddQDQNodePair<int8_t>(builder, input1_arg, .039f,
                                                (int8_t)((Input1Limits::max() + Input1Limits::min()) / 2 + 1),
                                                use_contrib_qdq);

      // add DQ weight
      auto* dq_w_output = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(weight, .003f, -10, dq_w_output, use_contrib_qdq);

      builder.AddNode("MatMul", {dq1_output, dq_w_output}, {output_arg});
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
      EXPECT_EQ(op_to_count["MatMul"], 0);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 1);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
    };

    auto add_session_options = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "1"));
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/, /*using NAN as a magic number to trigger cosine similarity*/
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options);

    auto add_session_options_disable_fm = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "0"));
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options_disable_fm);
  };

  test_case(false);  // Use ONNX QDQ ops
  test_case(true);   // Use com.microsoft QDQ ops
}

template <typename Input1Type, typename Input2Type, typename OutputType>
void QDQTransformerMatMulTests(bool has_output_q, bool disable_fastmath = false) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      typedef std::numeric_limits<Input1Type> Input1Limits;
      typedef std::numeric_limits<Input2Type> Input2Limits;
      typedef std::numeric_limits<OutputType> OutputTypeLimits;

      // add QDQ 1
      auto* q1_output = builder.MakeIntermediate();
      auto* dq1_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input1Type>(input1_arg,
                                                .039f,
                                                (Input1Limits::max() + Input1Limits::min()) / 2 + 1,
                                                q1_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<Input1Type>(q1_output,
                                                  .039f,
                                                  (Input2Limits::max() + Input1Limits::min()) / 2 + 1,
                                                  dq1_output, use_contrib_qdq);

      // add QDQ 2
      auto* q2_output = builder.MakeIntermediate();
      auto* dq2_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input2Type>(input2_arg,
                                                .04f,
                                                (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                q2_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<Input2Type>(q2_output,
                                                  .04f,
                                                  (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                  dq2_output, use_contrib_qdq);

      if (has_output_q) {
        // add binary operator
        auto* matmul_op_output = builder.MakeIntermediate();
        builder.AddNode("MatMul", {dq1_output, dq2_output}, {matmul_op_output});

        // add QDQ output
        auto* q3_output = builder.MakeIntermediate();
        builder.AddQuantizeLinearNode<OutputType>(matmul_op_output,
                                                  .039f,
                                                  (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                  q3_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<OutputType>(q3_output,
                                                    .039f,
                                                    (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                    output_arg, use_contrib_qdq);
      } else {
        builder.AddNode("MatMul", {dq1_output, dq2_output}, {output_arg});
      }
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if (has_output_q) {
        if constexpr (std::is_same<Input1Type, OutputType>::value &&
                      (std::is_same<Input1Type, uint8_t>::value ||
                       QDQIsInt8Allowed() && std::is_same<Input2Type, int8_t>::value)) {
          EXPECT_EQ(op_to_count["QLinearMatMul"], 1);
          EXPECT_EQ(op_to_count["MatMul"], 0);
          EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
          EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
        } else {
          EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
          EXPECT_EQ(op_to_count["MatMul"], 1);
          EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 3);
          EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 3);
        }
      } else {
        if constexpr (std::is_same<Input1Type, uint8_t>::value ||
                      (QDQIsInt8Allowed() && std::is_same<Input2Type, int8_t>::value)) {
          EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
          EXPECT_EQ(op_to_count["MatMul"], 0);
          EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
          EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
        } else {
          EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 0);
          EXPECT_EQ(op_to_count["MatMul"], 1);
          EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
          EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
        }
      }
    };

    auto add_session_options = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "1"));
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                      add_session_options);

    if (disable_fastmath) {
      auto add_session_options = [&](SessionOptions& so) {
        ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
            kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "0"));
      };

      TransformerTester(build_test_case,
                        check_graph,
                        TransformerLevel::Level1,
                        TransformerLevel::Level2,
                        12 /*opset_version*/,
                        NAN /*per_sample_tolerance*/,
                        NAN /*relative_per_sample_tolerance*/,
                        std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                        add_session_options);
    }
  };

  test_case({1, 2, 2}, {1, 2, 4});
  test_case({1, 23, 13, 13}, {13, 13});
  test_case({1, 22, 11, 13, 15}, {1, 22, 11, 15, 15});
  test_case({1, 2, 2}, {1, 2, 4}, true);  // Use com.microsoft QDQ ops
}

TEST(QDQTransformerTests, MatMul_U8U8U8_FastMath) {
  QDQTransformerMatMulTests<uint8_t, uint8_t, uint8_t>(false);
  QDQTransformerMatMulTests<uint8_t, uint8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8S8S8_FastMath) {
  QDQTransformerMatMulTests<uint8_t, int8_t, int8_t>(false);
  QDQTransformerMatMulTests<uint8_t, int8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8U8S8_FastMath) {
  QDQTransformerMatMulTests<uint8_t, uint8_t, int8_t>(false);
  QDQTransformerMatMulTests<uint8_t, uint8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_U8S8U8_FastMath) {
  QDQTransformerMatMulTests<uint8_t, int8_t, uint8_t>(false);
  QDQTransformerMatMulTests<uint8_t, int8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8S8S8_FastMath) {
  QDQTransformerMatMulTests<int8_t, int8_t, int8_t>(false);
  QDQTransformerMatMulTests<int8_t, int8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8U8U8_FastMath) {
  QDQTransformerMatMulTests<int8_t, uint8_t, uint8_t>(false);
  QDQTransformerMatMulTests<int8_t, uint8_t, uint8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8U8S8_FastMath) {
  QDQTransformerMatMulTests<int8_t, uint8_t, int8_t>(false);
  QDQTransformerMatMulTests<int8_t, uint8_t, int8_t>(true);
}

TEST(QDQTransformerTests, MatMul_S8S8U8_FastMath) {
  QDQTransformerMatMulTests<int8_t, int8_t, uint8_t>(false);
  QDQTransformerMatMulTests<int8_t, int8_t, uint8_t>(true);
}

// dummy test to disable the fastmath session op
TEST(QDQTransformerTests, MatMul_S8S8U8_DisableFastMath) {
  QDQTransformerMatMulTests<int8_t, int8_t, uint8_t>(false, true);
  QDQTransformerMatMulTests<int8_t, int8_t, uint8_t>(true, true);
}

template <typename Input1Type, typename Input2Type, typename OutputType, typename BiasType = int32_t>
void QDQTransformerGemmTests(bool has_output_q, bool has_bias, bool beta_not_one = false, bool disable_fastmath = false) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq = false) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      typedef std::numeric_limits<Input1Type> Input1Limits;
      typedef std::numeric_limits<Input2Type> Input2Limits;
      typedef std::numeric_limits<OutputType> OutputTypeLimits;

      std::vector<NodeArg*> input_args;

      // add QDQ A
      auto* q1_output = builder.MakeIntermediate();
      auto* dq1_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input1Type>(input1_arg,
                                                .039f,
                                                (Input1Limits::max() + Input1Limits::min()) / 2 + 1,
                                                q1_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<Input1Type>(q1_output,
                                                  .039f,
                                                  (Input2Limits::max() + Input1Limits::min()) / 2 + 1,
                                                  dq1_output, use_contrib_qdq);

      input_args.push_back(dq1_output);

      // add QDQ B
      auto* q2_output = builder.MakeIntermediate();
      auto* dq2_output = builder.MakeIntermediate();
      builder.AddQuantizeLinearNode<Input2Type>(input2_arg,
                                                .04f,
                                                (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                q2_output, use_contrib_qdq);
      builder.AddDequantizeLinearNode<Input2Type>(q2_output,
                                                  .04f,
                                                  (Input2Limits::max() + Input2Limits::min()) / 2 + 1,
                                                  dq2_output, use_contrib_qdq);
      input_args.push_back(dq2_output);

      if (has_bias) {
        auto* dq_bias_output = builder.MakeIntermediate();
        auto* bias = builder.MakeInitializer<BiasType>({input2_shape[1]}, static_cast<BiasType>(0), static_cast<BiasType>(127));
        builder.AddDequantizeLinearNode<BiasType>(bias, 0.00156f,
                                                  0,
                                                  dq_bias_output, use_contrib_qdq);
        input_args.push_back(dq_bias_output);
      }

      Node* gemm_node = nullptr;

      if (has_output_q) {
        auto* gemm_op_output = builder.MakeIntermediate();
        gemm_node = &builder.AddNode("Gemm", input_args, {gemm_op_output});

        // add QDQ output
        auto* q3_output = builder.MakeIntermediate();
        builder.AddQuantizeLinearNode<OutputType>(gemm_op_output,
                                                  .039f,
                                                  (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                  q3_output, use_contrib_qdq);
        builder.AddDequantizeLinearNode<OutputType>(q3_output,
                                                    .039f,
                                                    (OutputTypeLimits::max() + OutputTypeLimits::min()) / 2 + 1,
                                                    output_arg, use_contrib_qdq);
      } else {
        gemm_node = &builder.AddNode("Gemm", input_args, {output_arg});
      }

      if (beta_not_one) {
        gemm_node->AddAttribute("beta", 2.0f);
      }
    };

    auto check_binary_op_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      if ((!has_output_q || std::is_same_v<Input1Type, OutputType>)&&(!has_bias || (std::is_same_v<BiasType, int32_t> && !beta_not_one)) &&
          (std::is_same_v<Input1Type, uint8_t> || std::is_same_v<Input2Type, int8_t>)) {
        EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 1);
        EXPECT_EQ(op_to_count["Gemm"], 0);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], has_output_q ? 1 : 0);
      } else {
        int q_count = 2;   // Q for A and B
        int dq_count = 2;  // DQ for A and B
        if (has_bias) {
          dq_count++;
        }
        if (has_output_q) {
          q_count++;
          dq_count++;
        }
        EXPECT_EQ(op_to_count["com.microsoft.QGemm"], 0);
        EXPECT_EQ(op_to_count["Gemm"], 1);
        EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], q_count);
        EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], dq_count);
      }
    };

    auto add_session_options = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "1"));
    };

    TransformerTester(build_test_case,
                      check_binary_op_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_binary_op_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      18 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                      add_session_options);
    TransformerTester(build_test_case,
                      check_binary_op_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                      add_session_options);

    if (disable_fastmath) {
      auto add_session_options = [&](SessionOptions& so) {
        ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
            kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "0"));
      };

      TransformerTester(build_test_case,
                        check_binary_op_graph,
                        TransformerLevel::Level1,
                        TransformerLevel::Level2,
                        12 /*opset_version*/,
                        NAN /*per_sample_tolerance*/,
                        NAN /*relative_per_sample_tolerance*/,
                        std::make_unique<QDQSelectorActionTransformer>(QDQIsInt8Allowed()),
                        add_session_options);
    }
  };

  test_case({2, 2}, {2, 4});
  test_case({13, 15}, {15, 15});
  test_case({2, 2}, {2, 4}, true);  // Use com.microsoft QDQ ops
}

template <typename Input1Type, typename Input2Type, typename OutputType, typename BiasType = int32_t>
void QDQTransformerGemmTests() {
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, false);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, false);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(false, true, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, false, true);
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, true, true);
  // dummy test to disable the fastmath session
  QDQTransformerGemmTests<Input1Type, Input2Type, OutputType, BiasType>(true, true, true, true);
}

TEST(QDQTransformerTests, Gemm_U8U8U8_FastMath) {
  QDQTransformerGemmTests<uint8_t, uint8_t, uint8_t>();
  QDQTransformerGemmTests<uint8_t, uint8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8S8S8_FastMath) {
  QDQTransformerGemmTests<uint8_t, int8_t, int8_t>();
  QDQTransformerGemmTests<uint8_t, int8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8U8S8_FastMath) {
  QDQTransformerGemmTests<uint8_t, uint8_t, int8_t>();
  QDQTransformerGemmTests<uint8_t, uint8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_U8S8U8_FastMath) {
  QDQTransformerGemmTests<uint8_t, int8_t, uint8_t>();
  QDQTransformerGemmTests<uint8_t, int8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8S8S8_FastMath) {
  QDQTransformerGemmTests<int8_t, int8_t, int8_t>();
  QDQTransformerGemmTests<int8_t, int8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8U8U8_FastMath) {
  QDQTransformerGemmTests<int8_t, uint8_t, uint8_t>();
  QDQTransformerGemmTests<int8_t, uint8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8U8S8_FastMath) {
  QDQTransformerGemmTests<int8_t, uint8_t, int8_t>();
  QDQTransformerGemmTests<int8_t, uint8_t, int8_t, uint8_t>();
}

TEST(QDQTransformerTests, Gemm_S8S8U8_FastMath) {
  QDQTransformerGemmTests<int8_t, int8_t, uint8_t>();
  QDQTransformerGemmTests<int8_t, int8_t, uint8_t, uint8_t>();
}

TEST(QDQTransformerTests, MatMul_No_Fusion_FastMath) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>(input1_shape, -1.f, 1.f);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add QDQ + MatMul
      auto* matmul_output = builder.MakeIntermediate();
      auto* dq_matmul_output1 = AddQDQNodePair<uint8_t>(builder, input1_arg, .004f, 129, use_contrib_qdq);
      builder.AddNode("MatMul", {dq_matmul_output1, input2_arg}, {matmul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(matmul_output, .0039f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["MatMul"], 1);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 1);
    };

    auto add_session_options = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "1"));
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options);

    auto add_session_options_disable_fm = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "0"));
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options_disable_fm);
  };

  test_case({12, 37}, {37, 12}, false /*use_contrib_qdq*/);
  test_case({12, 37}, {37, 12}, true /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, MatMul_1st_Input_Int8_FastMath) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int8_t>(input1_shape, -128, 127);
      auto* input2_arg = builder.MakeInput<float>(input2_shape, -1.f, 1.f);
      auto* output_arg = builder.MakeOutput();

      // add DQ with type int8
      auto* dq_output_1 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<int8_t>(input1_arg, .004f, 1, dq_output_1, use_contrib_qdq);

      // add QDQ + MatMul
      auto* matmul_output = builder.MakeIntermediate();
      auto* dq_matmul_output2 = AddQDQNodePair<uint8_t>(builder, input2_arg, .004f, 129, use_contrib_qdq);
      builder.AddNode("MatMul", {dq_output_1, dq_matmul_output2}, {matmul_output});

      // add Q
      builder.AddQuantizeLinearNode<uint8_t>(matmul_output, .0039f, 135, output_arg, use_contrib_qdq);
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["MatMul"], 1);
      EXPECT_EQ(op_to_count["QLinearMatMul"], 0);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 2);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 2);
    };

    auto add_session_options = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "1"));
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options);

    auto add_session_options_disable_fm = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "0"));
    };

    TransformerTester(build_test_case, check_graph, TransformerLevel::Level1, TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr, add_session_options_disable_fm);
  };

  test_case({12, 37}, {37, 12}, false /*use_contrib_qdq*/);
  test_case({12, 37}, {37, 12}, true /*use_contrib_qdq*/);
  test_case({23, 13, 13}, {13, 13}, false /*use_contrib_qdq*/);
  test_case({22, 11, 13, 15}, {15, 13}, false /*use_contrib_qdq*/);
}

TEST(QDQTransformerTests, MatMulIntegerToFloat_FastMath) {
  auto test_case = [&](const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& input2_shape,
                       bool use_contrib_qdq) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<uint8_t>(input1_shape,
                                                    std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* input2_arg = builder.MakeInput<uint8_t>(input2_shape,
                                                    std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
      auto* output_arg = builder.MakeOutput();

      // add DQ
      auto* dq_output_1 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input1_arg, .0035f, 135, dq_output_1, use_contrib_qdq);

      auto* dq_output_2 = builder.MakeIntermediate();
      builder.AddDequantizeLinearNode<uint8_t>(input2_arg, .0035f, 135, dq_output_2, use_contrib_qdq);

      builder.AddNode("MatMul", {dq_output_1, dq_output_2}, {output_arg});
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      auto op_to_count = CountOpsInGraph(session.GetGraph());
      const QDQOpKeys qdq_keys = GetQDQOpKeys(use_contrib_qdq);
      EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 1);
      EXPECT_EQ(op_to_count[qdq_keys.quantize_linear], 0);
      EXPECT_EQ(op_to_count[qdq_keys.dequantize_linear], 0);
    };

    auto add_session_options = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "1"));
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr,
                      add_session_options);
    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      19 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr,
                      add_session_options);

    auto add_session_options_disable_fm = [&](SessionOptions& so) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "0"));
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level1,
                      TransformerLevel::Level2,
                      12 /*opset_version*/,
                      NAN /*per_sample_tolerance*/,
                      NAN /*relative_per_sample_tolerance*/,
                      nullptr,
                      add_session_options_disable_fm);
  };

  test_case({12, 37}, {37, 12}, false /*use_contrib_qdq*/);
  test_case({12, 37}, {37, 12}, true /*use_contrib_qdq*/);
  test_case({23, 13, 13}, {13, 13}, false /*use_contrib_qdq*/);
  test_case({22, 11, 13, 15}, {15, 13}, false /*use_contrib_qdq*/);
}

#endif  // !defined(DISABLE_CONTRIB_OPS) && defined(__aarch64)

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(__aarch64) && defined(__linux__) && !defined(DISABLE_CONTRIB_OPS)
