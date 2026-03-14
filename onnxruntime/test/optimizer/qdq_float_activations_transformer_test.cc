// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/int4.h"
#include "core/graph/model.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/qdq_transformer/qdq_float_activations_transformer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/unittest_util/qdq_test_utils.h"

#if !defined(DISABLE_CONTRIB_OPS)

namespace onnxruntime {
namespace test {

// Use scale=0.008 with zp=128 for uint8 to cover the [-1, 1] input range without clipping.
// Range: [-128*0.008, 127*0.008] = [-1.024, 1.016]. Max rounding error = 0.004.
constexpr float kTestScale = 0.008f;
constexpr uint8_t kTestZp = 128;
constexpr float kTestTolerance = 0.005f;

// For int8: scale=0.008, zp=0 covers [-128*0.008, 127*0.008] = [-1.024, 1.016]
constexpr int8_t kTestZpInt8 = 0;

// Test: Simple Q->DQ pair removed by QDQFloatActivationsTransformer
// Graph: Input -> Q -> DQ -> Relu -> Output
// Expected: Input -> Relu -> Output (Q and DQ removed)
TEST(QDQFloatActivationsTransformerTests, RemoveSimpleQDQPair) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 4, 8}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    auto* q_output = builder.MakeIntermediate();
    auto* dq_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q_output);
    builder.AddDequantizeLinearNode<uint8_t>(q_output, kTestScale, kTestZp, dq_output);

    builder.AddNode("Relu", {dq_output}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    EXPECT_EQ(op_to_count["Relu"], 1);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    kTestTolerance,
                    0.0f,
                    nullptr, add_session_options);
}

// Test: Q with multiple DQ consumers all get removed
// Graph: Input -> Q -> DQ1 -> Relu -> Out1
//                   -> DQ2 -> Sigmoid -> Out2
// Expected: Input -> Relu -> Out1, Input -> Sigmoid -> Out2
TEST(QDQFloatActivationsTransformerTests, RemoveQDQPairMultipleDQConsumers) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 4, 8}, -1.f, 1.f);
    auto* output1 = builder.MakeOutput();
    auto* output2 = builder.MakeOutput();

    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q_output);

    auto* dq1_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(q_output, kTestScale, kTestZp, dq1_output);
    builder.AddNode("Relu", {dq1_output}, {output1});

    auto* dq2_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(q_output, kTestScale, kTestZp, dq2_output);
    builder.AddNode("Sigmoid", {dq2_output}, {output2});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    EXPECT_EQ(op_to_count["Relu"], 1);
    EXPECT_EQ(op_to_count["Sigmoid"], 1);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    kTestTolerance,
                    0.0f,
                    nullptr, add_session_options);
}

// Test: Q->DQ pair not removed when scale/zp mismatch
TEST(QDQFloatActivationsTransformerTests, NoRemovalOnScaleMismatch) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 4, 8}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    auto* q_output = builder.MakeIntermediate();
    auto* dq_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q_output);
    builder.AddDequantizeLinearNode<uint8_t>(q_output, 0.009f, kTestZp, dq_output);

    builder.AddNode("Relu", {dq_output}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
    EXPECT_EQ(op_to_count["Relu"], 1);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    1.0f,
                    1.0f,
                    nullptr, add_session_options);
}

// Test: Option disabled - no Q->DQ removal
TEST(QDQFloatActivationsTransformerTests, OptionDisabledNoRemoval) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 4, 8}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    auto* q_output = builder.MakeIntermediate();
    auto* dq_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q_output);
    builder.AddDequantizeLinearNode<uint8_t>(q_output, kTestScale, kTestZp, dq_output);

    builder.AddNode("Relu", {dq_output}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["Relu"], 1);
    // Without kOrtSessionOptionsQDQFloatActivations, our transformer doesn't run.
    // With enable_quant_qdq_cleanup defaulting to "0", Q->DQ nodes should remain.
    EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/);
}

// Test: DQ producing graph output - handled via Identity node
// Graph: Input -> Q -> DQ -> (graph output)
TEST(QDQFloatActivationsTransformerTests, RemoveQDQPairWithGraphOutput) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 4, 8}, -1.f, 1.f);

    auto* q_output = builder.MakeIntermediate();
    auto* dq_output = builder.MakeOutput();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q_output);
    builder.AddDequantizeLinearNode<uint8_t>(q_output, kTestScale, kTestZp, dq_output);
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    kTestTolerance,
                    0.0f,
                    nullptr, add_session_options);
}

// Test: Multiple chained Q->DQ pairs all removed
// Graph: Input -> Q -> DQ -> Relu -> Q -> DQ -> Sigmoid -> Output
// Expected: Input -> Relu -> Sigmoid -> Output
TEST(QDQFloatActivationsTransformerTests, RemoveMultipleChainedQDQPairs) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 4, 8}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    auto* q1 = builder.MakeIntermediate();
    auto* dq1 = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q1);
    builder.AddDequantizeLinearNode<uint8_t>(q1, kTestScale, kTestZp, dq1);

    auto* relu_out = builder.MakeIntermediate();
    builder.AddNode("Relu", {dq1}, {relu_out});

    auto* q2 = builder.MakeIntermediate();
    auto* dq2 = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(relu_out, kTestScale, kTestZp, q2);
    builder.AddDequantizeLinearNode<uint8_t>(q2, kTestScale, kTestZp, dq2);

    builder.AddNode("Sigmoid", {dq2}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    EXPECT_EQ(op_to_count["Relu"], 1);
    EXPECT_EQ(op_to_count["Sigmoid"], 1);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    kTestTolerance,
                    0.0f,
                    nullptr, add_session_options);
}

// Test: int8_t Q->DQ pair removed
TEST(QDQFloatActivationsTransformerTests, RemoveInt8QDQPair) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 4, 8}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    auto* q_output = builder.MakeIntermediate();
    auto* dq_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<int8_t>(input_arg, kTestScale, kTestZpInt8, q_output);
    builder.AddDequantizeLinearNode<int8_t>(q_output, kTestScale, kTestZpInt8, dq_output);

    builder.AddNode("Relu", {dq_output}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    EXPECT_EQ(op_to_count["Relu"], 1);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    kTestTolerance,
                    0.0f,
                    nullptr, add_session_options);
}

// Test: With qdq_float_activations, data-movement ops keep Q/DQ adjacent so removal works.
// Graph: Input -> Q -> DQ -> Reshape -> Q -> DQ -> Relu -> Output
// With our option: DropQDQNodesRules skipped, Q->DQ pairs around Reshape stay adjacent,
//   and our transformer removes them.
TEST(QDQFloatActivationsTransformerTests, SkipDataMovementRules) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 3, 2, 2}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    auto* q1 = builder.MakeIntermediate();
    auto* dq1 = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q1);
    builder.AddDequantizeLinearNode<uint8_t>(q1, kTestScale, kTestZp, dq1);

    auto* reshape_shape = builder.Make1DInitializer<int64_t>({1, 12});
    auto* reshape_out = builder.MakeIntermediate();
    builder.AddNode("Reshape", {dq1, reshape_shape}, {reshape_out});

    auto* q2 = builder.MakeIntermediate();
    auto* dq2 = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(reshape_out, kTestScale, kTestZp, q2);
    builder.AddDequantizeLinearNode<uint8_t>(q2, kTestScale, kTestZp, dq2);

    builder.AddNode("Relu", {dq2}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    EXPECT_EQ(op_to_count["Reshape"], 1);
    EXPECT_EQ(op_to_count["Relu"], 1);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    kTestTolerance,
                    0.0f,
                    nullptr, add_session_options);
}

// Test: Conv with QDQ is fused into QLinearConv by Level1, while surrounding
// activation Q->DQ pairs are removed by our Level2 transformer.
// Graph: Input -> Q -> DQ -> Conv(QDQ weight) -> Q -> DQ -> Q -> DQ -> Identity -> Output
// After Level1: Input -> Q -> QLinearConv -> DQ -> Q -> DQ -> Identity -> Output
// After Level2: Input -> Q -> QLinearConv -> DQ -> Identity -> Output
TEST(QDQFloatActivationsTransformerTests, ConvQDQFusionWithActivationRemoval) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 3, 8, 8}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    // Input Q -> DQ pair (consumed by Conv QDQ fusion)
    constexpr float conv_input_scale = 0.04f;
    constexpr uint8_t conv_input_zp = 128;
    auto* dq_conv_input = AddQDQNodePair<uint8_t>(builder, input_arg, conv_input_scale, conv_input_zp);

    // Weight: constant uint8 initializer + DQ
    auto* weight = builder.MakeInitializer<uint8_t>({16, 3, 3, 3}, uint8_t(0), uint8_t(255));
    auto* dq_w_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<uint8_t>(weight, 0.03f, uint8_t(118), dq_w_output);

    // Conv
    auto* conv_output = builder.MakeIntermediate();
    builder.AddNode("Conv", {dq_conv_input, dq_w_output}, {conv_output});

    // Conv output Q -> DQ pair (Q consumed by Conv fusion, DQ remains to dequantize QLinearConv output)
    constexpr float conv_output_scale = 0.039f;
    constexpr uint8_t conv_output_zp = 135;
    auto* q_conv_out = builder.MakeIntermediate();
    auto* dq_conv_out = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(conv_output, conv_output_scale, conv_output_zp, q_conv_out);
    builder.AddDequantizeLinearNode<uint8_t>(q_conv_out, conv_output_scale, conv_output_zp, dq_conv_out);

    // Activation Q -> DQ pair (removed by our transformer).
    // Scale=0.05 with zp=128 gives representable range [-6.4, 6.35] which covers
    // the full Conv output range of [-5.265, 4.68] from DQ(0.039, 135).
    constexpr float act_scale = 0.05f;
    constexpr uint8_t act_zp = 128;
    auto* q_final = builder.MakeIntermediate();
    auto* dq_final = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(dq_conv_out, act_scale, act_zp, q_final);
    builder.AddDequantizeLinearNode<uint8_t>(q_final, act_scale, act_zp, dq_final);

    builder.AddNode("Identity", {dq_final}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    // Conv should be fused to QLinearConv by Level1 QDQSelectorActionTransformer
    EXPECT_EQ(op_to_count["QLinearConv"], 1);
    EXPECT_EQ(op_to_count["Conv"], 0);
    // Q_input remains (feeds QLinearConv), DQ_conv_out remains (dequantizes QLinearConv output)
    // Q_final and DQ_final removed by our transformer
    EXPECT_EQ(op_to_count["QuantizeLinear"], 1);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 1);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    0.04f /*per_sample_tolerance*/,
                    0.04f /*relative_per_sample_tolerance*/,
                    nullptr, add_session_options);
}

// Test: Activation Q->DQ removal enables DQ(blockwise)->MatMul fusion to MatMulNBits.
// At Level1, the MatMul has 2 DQ inputs (activation DQ + weight DQ), so DQMatMulNodeGroupSelector
// rejects it (requires exactly 1 DQ). Our Level2 transformer removes the activation Q->DQ (Sub-pass A),
// then Sub-pass B sees DQ(blockwise weight)->MatMul with 1 DQ and fuses it to MatMulNBits.
//
// Graph: Input -> Q -> DQ -> MatMul(DQ_blockwise(int4 weight)) -> Q -> DQ -> Identity -> Output
// After Level2: Input -> MatMulNBits -> Identity -> Output
TEST(QDQFloatActivationsTransformerTests, MatMulNBitsWithActivationRemoval) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    constexpr int64_t K = 64;
    constexpr int64_t N = 32;
    constexpr int64_t block_size = 32;
    constexpr int64_t num_blocks = (K + block_size - 1) / block_size;  // 2

    auto* input_arg = builder.MakeInput<float>({1, K}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    // Input activation Q -> DQ pair (removed by Sub-pass A)
    auto* q_input = builder.MakeIntermediate();
    auto* dq_act_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q_input);
    builder.AddDequantizeLinearNode<uint8_t>(q_input, kTestScale, kTestZp, dq_act_output);

    // Blockwise DQ for int4 weight (fused with MatMul to MatMulNBits by Sub-pass B)
    auto* weight_arg = builder.MakeInitializer<Int4x2>({K, N}, Int4x2(Int4x2::min_val, 0),
                                                       Int4x2(Int4x2::max_val, 0));
    auto* scale_arg = builder.MakeInitializer<float>({num_blocks, N}, 0.5f, 2.0f);
    auto* zp_arg = builder.MakeInitializer<Int4x2>({num_blocks, N}, Int4x2(0, 0), Int4x2(2, 0));

    auto* dq_weight_output = builder.MakeIntermediate();
    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_weight_output}, "", &dq_attrs);

    // MatMul with activation (input A) and blockwise-dequantized weight (input B)
    auto* matmul_output = builder.MakeIntermediate();
    builder.AddNode("MatMul", {dq_act_output, dq_weight_output}, {matmul_output});

    // Output activation Q -> DQ pair (removed by Sub-pass A).
    // MatMul output range can be large (K=64 * weight range), so use a wide scale.
    // Scale=8.0 with zp=128 covers [-1024, 1016].
    constexpr float output_scale = 8.0f;
    auto* q_output = builder.MakeIntermediate();
    auto* dq_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(matmul_output, output_scale, kTestZp, q_output);
    builder.AddDequantizeLinearNode<uint8_t>(q_output, output_scale, kTestZp, dq_output);

    builder.AddNode("Identity", {dq_output}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    // Both activation Q->DQ pairs removed, DQ(blockwise)->MatMul fused to MatMulNBits
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(op_to_count["MatMul"], 0);
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    5.0f /*per_sample_tolerance - output Q->DQ rounding error up to scale/2=4*/,
                    5.0f /*relative_per_sample_tolerance*/,
                    nullptr, add_session_options);
}

// Test: Weight DQ on constant initializer is constant-folded (Sub-pass C)
// Graph: Input -> Q -> DQ -> Add(DQ(int8 weight constant)) -> Output
// Expected: Input -> Add(float weight) -> Output (activation Q->DQ removed, weight DQ constant-folded)
TEST(QDQFloatActivationsTransformerTests, WeightDQConstantFolding) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({1, 4, 8}, -1.f, 1.f);
    auto* output_arg = builder.MakeOutput();

    // Activation Q->DQ on input
    auto* q_output = builder.MakeIntermediate();
    auto* dq_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint8_t>(input_arg, kTestScale, kTestZp, q_output);
    builder.AddDequantizeLinearNode<uint8_t>(q_output, kTestScale, kTestZp, dq_output);

    // Weight: int8 constant initializer with DQ (per-tensor, not blockwise)
    auto* weight_init = builder.MakeInitializer<int8_t>({1, 4, 8}, -64, 64);
    constexpr float weight_scale = 0.01f;
    constexpr int8_t weight_zp = 0;
    auto* weight_dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<int8_t>(weight_init, weight_scale, weight_zp, weight_dq_output);

    // Add: activation + weight
    builder.AddNode("Add", {dq_output, weight_dq_output}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    // Activation Q->DQ removed (Sub-pass A)
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    // Weight DQ constant-folded (Sub-pass C)
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
    EXPECT_EQ(op_to_count["Add"], 1);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    kTestTolerance,
                    0.0f,
                    nullptr, add_session_options);
}

// Test: Gemm with activation Q->DQ, uint8 weight DQ, and int32 bias DQ
// Full QDQ quantization commonly use this pattern: all inputs are quantized,
// with the activation as uint16, weight as uint8, and bias as int32.
// GemmQDQRules rejects uint16 activations (16-bit not allowed), so the Gemm stays
// unfused in pass 1. After activation Q->DQ removal (Sub-pass A), the remaining
// DQ(uint8 weight)->Gemm with DQ(int32 bias) should be fused to MatMulNBits (Sub-pass B).
// Graph before: Input -> Q(u16) -> DQ(u16) -> Gemm(DQ(u8 weight), DQ(i32 bias)) -> Q(u16) -> DQ(u16) -> Output
// Expected:     Input -> MatMulNBits(bias) -> Output
TEST(QDQFloatActivationsTransformerTests, GemmWithInt32BiasDQ) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    constexpr int64_t K = 64;
    constexpr int64_t N = 32;
    constexpr float act_scale = 0.00002f;  // uint16 covers [-0.655, 1.311] with zp=32768
    constexpr uint16_t act_zp = 32768;

    auto* input_arg = builder.MakeInput<float>({1, K}, -0.5f, 0.5f);
    auto* output_arg = builder.MakeOutput();

    // Input activation Q -> DQ pair using uint16 (removed by Sub-pass A)
    // uint16 is rejected by GemmQDQRules so Gemm stays unfused in first pass.
    auto* q_input = builder.MakeIntermediate();
    auto* dq_act_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint16_t>(input_arg, act_scale, act_zp, q_input);
    builder.AddDequantizeLinearNode<uint16_t>(q_input, act_scale, act_zp, dq_act_output);

    // Per-channel DQ for uint8 weight (axis=1, scale shape [N])
    auto* weight_arg = builder.MakeInitializer<uint8_t>({K, N}, static_cast<uint8_t>(0), static_cast<uint8_t>(255));
    auto* dq_weight_output = builder.MakeIntermediate();
    std::vector<float> weight_scales(static_cast<size_t>(N), 0.03f);
    std::vector<uint8_t> weight_zps(static_cast<size_t>(N), static_cast<uint8_t>(128));
    builder.AddDequantizeLinearNode<uint8_t>(weight_arg, weight_scales, weight_zps, dq_weight_output);

    // Bias DQ: int32 quantized bias -> float (common in QNN quantization)
    auto* bias_quantized = builder.MakeInitializer<int32_t>({N}, static_cast<int32_t>(-1000), static_cast<int32_t>(1000));
    auto* bias_dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode<int32_t>(bias_quantized, 0.0001f, static_cast<int32_t>(0), bias_dq_output);

    // Gemm with activation, weight, and bias
    auto* gemm_output = builder.MakeIntermediate();
    builder.AddNode("Gemm", {dq_act_output, dq_weight_output, bias_dq_output}, {gemm_output});

    // Output activation Q -> DQ pair using uint16 (removed by Sub-pass A)
    constexpr float output_scale = 0.1f;
    auto* q_output = builder.MakeIntermediate();
    auto* dq_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<uint16_t>(gemm_output, output_scale, act_zp, q_output);
    builder.AddDequantizeLinearNode<uint16_t>(q_output, output_scale, act_zp, dq_output);

    builder.AddNode("Identity", {dq_output}, {output_arg});
  };

  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    // Gemm fused to MatMulNBits, activation Q->DQ pairs removed
    EXPECT_EQ(op_to_count["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(op_to_count["Gemm"], 0);
    EXPECT_EQ(op_to_count["QuantizeLinear"], 0);
    // Bias DQ constant-folded by Sub-pass C (constant int32 initializer)
    EXPECT_EQ(op_to_count["DequantizeLinear"], 0);
  };

  auto add_session_options = [](SessionOptions& so) {
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsQDQFloatActivations, "1"));
  };

  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Level1,
                    TransformerLevel::Level2,
                    21 /*opset_version*/,
                    5.0f /*per_sample_tolerance - output Q->DQ rounding error up to scale/2=4*/,
                    5.0f /*relative_per_sample_tolerance*/,
                    nullptr, add_session_options);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS)
