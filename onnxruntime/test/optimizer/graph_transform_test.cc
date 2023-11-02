// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <random>
#include "core/graph/onnx_protobuf.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"

#include "asserts.h"
#include "core/common/span_utils.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/bias_dropout_fusion.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/bias_softmax_fusion.h"
#include "core/optimizer/cast_elimination.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/concat_slice_elimination.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/constant_sharing.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_act_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/matmul_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/div_mul_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/dynamic_quantize_matmul_fusion.h"
#include "core/optimizer/expand_elimination.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/gather_fusion.h"
#include "core/optimizer/gelu_approximation.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/gemm_sum_fusion.h"
#include "core/optimizer/gemm_transpose_fusion.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_config.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/isinf_reducesum_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/matmul_integer_to_float.h"
#include "core/optimizer/matmul_scale_fusion.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/optimizer/noop_elimination.h"
#include "core/optimizer/not_where_fusion.h"
#include "core/optimizer/propagate_cast_ops.h"
#include "core/optimizer/quick_gelu_fusion.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/utils.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/util/math.h"
#include "test/capturing_sink.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/compare_ortvalue.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/temp_dir.h"
#include "test/util/include/test_utils.h"
#include "core/optimizer/pre_shape_node_elimination.h"
#include "core/optimizer/double_qdq_pairs_remover.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#ifdef ENABLE_TRAINING
#include "orttraining/core/optimizer/bitmask_dropout_replacement.h"
#endif

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")
TEST_F(GraphTransformationTests, IdentityElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "abs-id-max.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateIdentity>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

TEST_F(GraphTransformationTests, IdentityEliminationWithGraphOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "abs-id.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateIdentity>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

TEST_F(GraphTransformationTests, IdentityWithSharedNodeArgNotEliminated) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "id-elim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 2);
  ASSERT_TRUE(op_to_count["Add"] == 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<CommonSubexpressionElimination>(),
                                                     TransformerLevel::Level1));
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateIdentity>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // after CommonSubexpressionElimination, Add would have 1 output def and 2 edges
  // each edge would share the same input node arg 0. Thus after execution, only one of the 2 outputs
  // has data. Thus skip.
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 2);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

// Runs a model to ensure that common subexpression elimination does not eliminate
// DequantizeLinear nodes.
TEST_F(GraphTransformationTests, DequantizeLinearNodeNotEliminated) {
  auto test_case = [](const ORTCHAR_T* model_uri,
                      bool use_contrib_qdq,
                      const logging::Logger& logger) {
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count[dq_key], 25);

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<CommonSubexpressionElimination>(),
                                                       TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

    // CommonSubexpressionElimination should skip the DequantizeLinear nodes
    op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count[dq_key], 25);
  };

  test_case(MODEL_FOLDER "qdq_with_multi_consumer_dq_nodes.fixed.onnx",
            false,  // use_contrib_qdq
            *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  // Test with 8-bit com.microsoft.DequantizeLinear
  test_case(MODEL_FOLDER "qdq_with_multi_consumer_dq_nodes.fixed.qdq_contrib.onnx",
            true,  // use_contrib_qdq
            *logger_);
  // Test with 16-bit com.microsoft.DequantizeLinear
  test_case(MODEL_FOLDER "qdq_with_multi_consumer_dq_nodes.fixed.qdq16_contrib.onnx",
            true,  // use_contrib_qdq
            *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

TEST_F(GraphTransformationTests, IdentityInputIsGraphOutputNotEliminated) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "scan9_sum.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  // tips: to dump the subgraph, can use python tool - dump_subgraphs.py
  // or click on one of the input to see the drop down graph list and view subgraph

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateIdentity>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Identity's input in subgraph is also graph output. Thus skip.
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);
}

TEST_F(GraphTransformationTests, NoopElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "noop-add.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 5);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<NoopElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] + CountOpsInGraph(graph)["Sub"] + CountOpsInGraph(graph)["Mul"] +
                           CountOpsInGraph(graph)["Div"] ==
                       1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] + CountOpsInGraph(graph)["Sub"] + CountOpsInGraph(graph)["Mul"] +
                           CountOpsInGraph(graph)["Div"] ==
                       0);
    return Status::OK();
  };

  // x+0, float.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<float>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<float>({}, {0.0f});
      auto* add_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Add", {matmul_output, initializer_arg}, {add_out});
      builder.AddNode("Identity", {add_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // 0+x, fp16.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<MLFloat16>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<MLFloat16>({1}, {MLFloat16(0.0f)});
      auto* add_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Add", {initializer_arg, matmul_output}, {add_out});
      builder.AddNode("Identity", {add_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x-0, double.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<double>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<double>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<double>({1, 1}, {static_cast<double>(0.0f)});
      auto* sub_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Sub", {matmul_output, initializer_arg}, {sub_out});
      builder.AddNode("Identity", {sub_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x*1, int32.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int32_t>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<int32_t>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<int32_t>({1, 1, 1}, {1});
      auto* mul_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Mul", {matmul_output, initializer_arg}, {mul_out});
      builder.AddNode("Identity", {mul_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // 1*x, int64.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int64_t>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<int64_t>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<int64_t>({1, 1, 1, 1}, {static_cast<int64_t>(1)});
      auto* mul_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Mul", {initializer_arg, matmul_output}, {mul_out});
      builder.AddNode("Identity", {mul_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x/1, float.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<float>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<float>({}, {1.0f});
      auto* div_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Div", {matmul_output, initializer_arg}, {div_out});
      builder.AddNode("Identity", {div_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Invalid case: x+1.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<float>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<float>({}, {1.0f});
      auto* add_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Add", {matmul_output, initializer_arg}, {add_out});
      builder.AddNode("Identity", {add_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: initializer rank is larger.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<MLFloat16>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<MLFloat16>({1, 1, 1, 1, 1}, {MLFloat16(0.0f)});
      auto* add_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Add", {initializer_arg, matmul_output}, {add_out});
      builder.AddNode("Identity", {add_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: 0-x.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<double>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<double>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<double>({1, 1}, {static_cast<double>(0.0f)});
      auto* sub_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Sub", {initializer_arg, matmul_output}, {sub_out});
      builder.AddNode("Identity", {sub_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: x-1.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<double>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<double>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<double>({1, 1}, {static_cast<double>(1.0f)});
      auto* sub_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Sub", {matmul_output, initializer_arg}, {sub_out});
      builder.AddNode("Identity", {sub_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: 0*x.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int32_t>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<int32_t>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<int32_t>({1, 1, 1}, {0});
      auto* mul_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Mul", {initializer_arg, matmul_output}, {mul_out});
      builder.AddNode("Identity", {mul_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: output is graph output.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int64_t>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<int64_t>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<int64_t>({1, 1, 1, 1}, {static_cast<int64_t>(1)});
      auto* mul_out = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Mul", {initializer_arg, matmul_output}, {mul_out});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }

  // Invalid case: 1/x.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* input2_arg = builder.MakeInput<float>({{3, 3}});
      auto* matmul_output = builder.MakeIntermediate();
      auto* initializer_arg = builder.MakeInitializer<float>({}, {1.0f});
      auto* div_out = builder.MakeIntermediate();
      auto* identity_output = builder.MakeOutput();

      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_output});
      builder.AddNode("Div", {initializer_arg, matmul_output}, {div_out});
      builder.AddNode("Identity", {div_out}, {identity_output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<NoopElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, pre_graph_checker));
  }
}

TEST_F(GraphTransformationTests, DropoutElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "dropout.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 6);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateDropout>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // Of the 6 Dropout nodes in the graph, all but the ones named `d1` and `d6` should have been removed.
  // A Dropout node can be removed if its second, optional output `mask` is either missing or unused downstream.
  // `d1` cannot be removed because an Identity node has its `mask` output as an input;
  // `d6` cannot be removed because its `mask` output is marked as a graph output.
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 2);
}

TEST_F(GraphTransformationTests, SliceElimination) {
  std::vector<std::basic_string<ORTCHAR_T>> model_names = {ORT_TSTR("slice-v1-elim.onnx"), ORT_TSTR("slice-v11-elim.onnx")};
  for (const auto& model_name : model_names) {
    PathString model_uri = PathString(MODEL_FOLDER) + model_name;
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    int initial_slice_num = op_to_count["Slice"];

    auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateSlice>()));
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    op_to_count = CountOpsInGraph(graph);
    // Only one Slice operator is redundant and is removed.
    ASSERT_TRUE(op_to_count["Slice"] == --initial_slice_num);
  }
}

TEST_F(GraphTransformationTests, ConstantFolding) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingNodesOnDifferentEP) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/), TransformerLevel::Level1));

  // assign all nodes to CUDA. the constant folding should override this to perform the constant folding on cpu
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);

  // all remaining nodes should still be on CUDA
  for (auto& node : graph.Nodes()) {
    EXPECT_STREQ(node.GetExecutionProviderType().c_str(), kCudaExecutionProvider);
  }
}

TEST_F(GraphTransformationTests, ConstantFoldingUnsupportedFloat16) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "constant_float16_mul.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Mul"] == 1);
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/), TransformerLevel::Level1));

  // assign all nodes to CUDA. the constant folding should try folding the node on the CPU and fail, thus leaving the
  // EP as CUDA and not constant folding the node.
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Mul"] == 1);

  // all nodes should still be on CUDA
  for (auto& node : graph.Nodes()) {
    EXPECT_STREQ(node.GetExecutionProviderType().c_str(), kCudaExecutionProvider);
  }
}

TEST_F(GraphTransformationTests, ConstantFoldingSubgraph) {
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  TypeProto float_tensor_type;
  float_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_subgraph = [&](GraphProto& graph_proto) {
    // create subgraph that has an Add node to add a local and parent graph initializer
    Model model("ConstantFoldingSubgraphTest_subgraph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, *logger_);
    auto& graph = model.MainGraph();

    TensorProto local_constant(value_tensor);
    local_constant.set_name("local_constant");
    graph.AddInitializedTensor(local_constant);

    auto& local_constant_arg = graph.GetOrCreateNodeArg("local_constant", &float_tensor_type);
    auto& parent_constant_arg = graph.GetOrCreateNodeArg("parent_constant", &float_tensor_type);
    graph.AddOuterScopeNodeArg("parent_constant");

    auto& add_out = graph.GetOrCreateNodeArg("add_out", &float_tensor_type);
    graph.AddNode("add", "Add", "Add two inputs.", {&parent_constant_arg, &local_constant_arg}, {&add_out});

    auto& subgraph_out = graph.GetOrCreateNodeArg("subgraph_out", &float_tensor_type);
    graph.AddNode("identity", "Identity", "So Add isn't providing graph output.", {&add_out}, {&subgraph_out});

    ASSERT_STATUS_OK(graph.Resolve());
    graph_proto = graph.ToGraphProto();
  };

  Model model("ConstantFoldingSubgraphTest_main_graph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, *logger_);
  auto& graph = model.MainGraph();

  // add initializer at parent level
  TensorProto parent_value_tensor(value_tensor);
  parent_value_tensor.set_name("parent_constant");
  graph.AddInitializedTensor(parent_value_tensor);

  // put the subgraph in an If node
  TypeProto if_cond_type;
  if_cond_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  if_cond_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& if_cond_input = graph.GetOrCreateNodeArg("if_in", &if_cond_type);
  auto& if_output = graph.GetOrCreateNodeArg("if_out", &float_tensor_type);

  auto& if_node = graph.AddNode("if", "If", "If node", {&if_cond_input}, {&if_output});

  GraphProto subgraph;
  create_subgraph(subgraph);

  if_node.AddAttribute("then_branch", subgraph);
  if_node.AddAttribute("else_branch", subgraph);

  ASSERT_STATUS_OK(graph.Resolve());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 2);  // one in each subgraph
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0)
      << "Constant folding should have been able to remove the Add node in both subgraphs";
}

TEST_F(GraphTransformationTests, ConstantFoldingWithShapeToInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_with_shape_to_initializer.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 2);
  ASSERT_TRUE(op_to_count["MatMul"] == 2);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 3);

  InlinedHashSet<std::string_view> compatible_eps;
  InlinedHashSet<std::string> excluded_initializers;
  excluded_initializers.insert("matmul_weight");
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(),
                                        false /*skip_dequantize_linear*/,
                                        compatible_eps,
                                        excluded_initializers),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 2);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingWithScalarShapeToInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_with_scalar_shape_to_initializer.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);

  InlinedHashSet<std::string_view> compatible_eps;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(),
                                        false /*skip_dequantize_linear*/,
                                        compatible_eps),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

TEST_F(GraphTransformationTests, ConstantFoldingForOpsWithMissingOptionalInputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_for_ops_having_missing_optional_inputs.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());

  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // The Resize node has some missing optional inputs (roi and scales)
  ASSERT_TRUE(op_to_count["Resize"] == 1);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  InlinedHashSet<std::string_view> compatible_eps;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(),
                                        false /*skip_dequantize_linear*/,
                                        compatible_eps),
      TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // The Resize node is constant folded
  ASSERT_TRUE(op_to_count["Resize"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);
}

static void VerifyConstantFoldingWithDequantizeLinear(const std::unordered_map<std::string, int>& expected_op_count,
                                                      Graph& graph,
                                                      SessionOptions& session_options,
                                                      const Logger& logger) {
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  bool has_constant_folding = false;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level1, session_options, *e.get(), {});
  for (auto& transformer : transformers) {
    if (transformer->Name() == "ConstantFolding") {
      ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(transformer), TransformerLevel::Level1));
      has_constant_folding = true;
    }
  }

  ASSERT_TRUE(has_constant_folding);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  for (const auto& entry : expected_op_count) {
    if (entry.second == 0) {
      ASSERT_TRUE(op_to_count.find(entry.first) == op_to_count.end())
          << entry.first << " should not exist in the graph";
    } else {
      ASSERT_TRUE(op_to_count[entry.first] == entry.second)
          << entry.first << " mismatch. Expected:" << entry.second << " Got:" << op_to_count[entry.first];
    }
  }
}

// Runs a model that checks constant folding with DequantizeLinear nodes.
TEST_F(GraphTransformationTests, ConstantFoldingWithDequantizeLinear) {
  auto test_case = [](const ORTCHAR_T* model_uri,
                      bool use_contrib_qdq,
                      const logging::Logger& logger) {
    const char* q_key = use_contrib_qdq ? "com.microsoft.QuantizeLinear" : "QuantizeLinear";
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[q_key] == 1);
    ASSERT_TRUE(op_to_count[dq_key] == 3);
    ASSERT_TRUE(op_to_count["Conv"] == 1);

    std::unordered_map<std::string, int> expected_op_counts = {{q_key, 1},
                                                               {dq_key, 3},
                                                               {"Conv", 1}};

    SessionOptions session_options;
    // Check DequantizeLinear aren't constant folded for default setting.
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);

    // set kOrtSessionOptionsDisableQuantQDQ to enable it explicitly
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "0"));
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);

    // set SessionOptionsEnableQuantQDQ to disable it
    expected_op_counts[dq_key] = 1;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableQuantQDQ, "1"));
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);
  };

  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.onnx",
            false, *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  // Test with 8-bit contrib QDQ ops
  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.qdq_contrib.onnx",
            true, *logger_);
  // Test with 16-bit contrib QDQ ops
  test_case(MODEL_FOLDER "fusion/constant_folding_dequantizelinear.qdq16_contrib.onnx",
            true, *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

// model with 2 QDQ node units that can be constant folded as they are simple DQ -> Node -> Q where DQ and Node have
// single consumer and do not produce graph outputs. Node is deterministic.
// there are also other DQ nodes that should be ignored.
TEST_F(GraphTransformationTests, ConstantFoldingQDQNodeUnit) {
  auto test_case = [](const ORTCHAR_T* model_uri, bool use_contrib_qdq, const logging::Logger& logger) {
    const char* q_key = use_contrib_qdq ? "com.microsoft.QuantizeLinear" : "QuantizeLinear";
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[q_key] == 3);
    ASSERT_TRUE(op_to_count[dq_key] == 4);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 1);
    ASSERT_TRUE(op_to_count["Transpose"] == 1);

    SessionOptions session_options;

    // 2 QDQ node units should be constant folded and go away
    std::unordered_map<std::string, int> expected_op_counts = {{q_key, 1},
                                                               {dq_key, 2},
                                                               {"Transpose", 0},
                                                               {"Unsqueeze", 0}};

    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);
  };

  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.onnx", false, *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  // Test with 8-bit com.microsoft.Q/DQ
  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.qdq_contrib.onnx", true, *logger_);
  // Test with 16-bit com.microsoft.Q/DQ
  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.qdq16_contrib.onnx", true, *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

// Simple QDQ Node Unit but shouldn't be constant folded as the node in the middle produces a graph output
TEST_F(GraphTransformationTests, ConstantFoldingQDQNodeUnitGraphOutput) {
  auto test_case = [](const ORTCHAR_T* model_uri, bool use_contrib_qdq, const logging::Logger& logger) {
    const char* q_key = use_contrib_qdq ? "com.microsoft.QuantizeLinear" : "QuantizeLinear";
    const char* dq_key = use_contrib_qdq ? "com.microsoft.DequantizeLinear" : "DequantizeLinear";

    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, logger));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[q_key] == 2);
    ASSERT_TRUE(op_to_count[dq_key] == 3);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 1);

    std::unordered_map<std::string, int> expected_op_counts = {{q_key, 2},
                                                               {dq_key, 3},
                                                               {"Unsqueeze", 1}};

    SessionOptions session_options;
    VerifyConstantFoldingWithDequantizeLinear(expected_op_counts, graph, session_options, logger);
  };

  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.graph_output.onnx", false, *logger_);
#if !defined(DISABLE_CONTRIB_OPS)
  // Test with 8-bit contrib QDQ ops
  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.graph_output.qdq_contrib.onnx", true, *logger_);

  // Test with 16-bit contrib QDQ ops
  test_case(MODEL_FOLDER "fusion/constant_folding_qdq_node_unit.graph_output.qdq16_contrib.onnx", true, *logger_);
#endif  // !defined(DISABLE_CONTRIB_OPS)
}

TEST_F(GraphTransformationTests, ConstantFolding_RemoveDanglingInputNodesToConstantFoldedNode) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_remove_dangling_inputs.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);          // Shape node that will be constant folded
  ASSERT_TRUE(op_to_count["Add"] == 1);            // Input node to Shape
  ASSERT_TRUE(op_to_count["RandomUniform"] == 1);  // Input node to Add

  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["RandomUniform"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingAShapeNodeDeepInTheGraph) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "shape-add.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 4);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  // A Shape node very deep in the graph (feeding into an Identity
  // node that produces the graph output) gets constant folded which
  // removes all its ancestors and the Identity node consuming this Shape's
  // output is subsequently constant folded to leave the graph with no
  // nodes.
  ASSERT_TRUE(op_to_count.size() == 0U);
}

// Test we don't fail when constant folding hits a string initializer
TEST_F(GraphTransformationTests, ConstantFoldingStringInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "gh_issue_17392.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Identity"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<CPUExecutionProvider> e = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConstantFolding>(*e.get(), false /*skip_dequantize_linear*/), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count.size(), 0U) << "Identity node should have been removed";
}

TEST_F(GraphTransformationTests, ConstantFoldingIfConstantInlining) {
  // This test covers the following necessary cases:
  // The input refers to the explicit or implicit inputs of If node.
  // The output of the node is the output of the subgraph being inlined.
  // Constant nodes and initializers are promoted to the outer graph.
  // The initializer or a constant node is the output of the subgraph being inlined.
  // Nested subgraphs names are renamed as appropriate.
  // In all If node is constant folded twice. The last If node is not constant
  // folded because the input is indirectly dependent on the size of the input.
  // XXX: Can we constant fold Size() if the graph input shape is fixed?

  const char* code = R"(
  <
  ir_version: 8,
  opset_import: [ "" : 16, "local" : 1 ]
  >
  agraph (float[128] x, float[128] x1) => (float[N] y)
  {
      y = local.aten_gather <dim: int = 1, sparse_grad: int = 0> (x, x1)
  }
  <
    opset_import: [ "" : 16, "local" : 1],
    domain: "local"
  >
  aten_gather <dim>(self, index) => (result_16)
  {
     tmp = Shape (index)
     tmp_0 = Size (tmp)
     int64_0 = Constant <value: tensor = int64 int64_0 {0}> ()
     int64_0_cast = CastLike (int64_0, tmp_0)
     cond = Equal (tmp_0, int64_0_cast)
     result_16 = If (cond) <then_branch: graph = thenGraph_10 () => ( result) {
        result = Identity (self)
     }, else_branch: graph = elseGraph_10 () => ( result_15) {
        tmp_1 = Shape (self)
        tmp_2 = Size (tmp_1)
        int64_0_3 = Constant <value: tensor = int64 int64_0_3 {0}> ()
        int64_0_3_cast = CastLike (int64_0_3, tmp_2)
        cond_4 = Equal (tmp_2, int64_0_3_cast)
        self_8 = If (cond_4) <then_branch: graph = thenGraph_13 () => ( self_6) {
           tmp_5 = Constant <value_ints: ints = [-1]> ()
           self_6 = Reshape (self, tmp_5)
        }, else_branch: graph = elseGraph_13 () => ( self_7) {
           self_7 = Identity (self)
        }>
        tmp_9 = Size (index)
        int64_0_10 = Constant <value: tensor = int64 int64_0_10 {0}> ()
        int64_0_10_cast = CastLike (int64_0_10, tmp_9)
        cond_11 = Equal (tmp_9, int64_0_10_cast)
        result_15 = If (cond_11) <then_branch: graph = thenGraph_15 () => ( result_12) {
           result_12 = CastLike (index, self_8)
        }, else_branch: graph = elseGraph_15 () => ( result_14) {
           index_13 = Cast <to: int = 7> (index)
           result_14 = GatherElements <axis: int = @dim> (self_8, index_13)
        }>
     }>
  }
)";

  ONNX_NAMESPACE::OnnxParser parser(code);
  ONNX_NAMESPACE::ModelProto model_proto;
  auto parse_status = parser.Parse(model_proto);
  ASSERT_TRUE(parse_status.IsOK()) << parse_status.ErrorMessage();
  ASSERT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  {
    // Test that the model is loadable and check the function call node.
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(std::move(model_proto), p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["local.aten_gather"], 1);
    model_proto = p_model->ToProto();
  }

  std::string serialized_model;
  const bool serialization_status = model_proto.SerializeToString(&serialized_model);
  ASSERT_TRUE(serialization_status) << "Failed to serialize proto to string";

  // AOT inlining is necessary in this case, so the If nodes within the function
  // are brought out to the outer scope. So we load this into a session object.

  SessionOptions session_options;
  InferenceSessionWrapper session_object{session_options, GetEnvironment()};

  std::stringstream sstr(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(sstr));
  ASSERT_STATUS_OK(session_object.Initialize());

  // const auto resulting_model_proto = session_object.GetModel().ToProto();
  // std::string printed_model = ONNX_NAMESPACE::ProtoToString(resulting_model_proto);
  // ASSERT_FALSE(printed_model.empty());
  // std::cout << printed_model << std::endl;

  // This is the resulting model proto.
  // The remaining If node is not constant foldable because Size() does not constant fold
  // although the shape is fixed.
  /*
    <
       ir_version: 8,
       opset_import: ["" : 16, "local" : 1, "com.microsoft.nchwc" : 1, "ai.onnx.ml" : 4, "com.ms.internal.nhwc" : 20, "ai.onnx.training" : 1, "ai.onnx.preview.training" : 1, "com.microsoft" : 1, "com.microsoft.experimental" : 1, "org.pytorch.aten" : 1]
    >
    agraph (float[128] x, float[128] x1) => (float[128] y) {
       _if_elseGraph_10__inlfunc_aten_gather_tmp_9 = Size (x1)
       _if_elseGraph_10__inlfunc_aten_gather_cond_11 = Equal (_if_elseGraph_10__inlfunc_aten_gather_tmp_9, ortshared_7_0_1_0_token_10)
       y = If (_if_elseGraph_10__inlfunc_aten_gather_cond_11) <then_branch: graph = thenGraph_15 () => (float[128] _inlfunc_aten_gather_result_12) {
          _inlfunc_aten_gather_result_12 = Cast <to: int = 1> (x1)
       }, else_branch: graph = elseGraph_15 () => (float[128] _inlfunc_aten_gather_result_14) {
          _inlfunc_aten_gather_index_13 = Cast <to: int = 7> (x1)
          _inlfunc_aten_gather_result_14 = GatherElements <axis: int = 1> (x, _inlfunc_aten_gather_index_13)
       }>
    }
  */

  auto& graph = session_object.GetModel().MainGraph();
  auto op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["local.aten_gather"], 0);
  ASSERT_EQ(op_to_count["If"], 1);
}

// Check transformations in the case of a subgraph with constant inputs.
TEST_F(GraphTransformationTests, SubgraphWithConstantInputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "constant-subgraph.onnx";

  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Level2;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));

  ASSERT_STATUS_OK(session_object.Initialize());

  NameMLValMap feeds;
  RunOptions run_options;

  std::vector<std::string> output_names = {"output"};
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));
}

TEST_F(GraphTransformationTests, FuseConvBNNoBias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string bn_output_name;

  // add a missing optional output to BN. this should be fusable
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "BatchNormalization") {
      node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg("", nullptr));
      bn_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), bn_output_name)
          << "fusion should produce the same output name as the last node";
    }
  }
}

TEST_F(GraphTransformationTests, FuseMatmulBNWithInBetweenNodes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-with-reshape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "MatMul") {
      expected_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), expected_output_name)
          << "fusion should produce the same output name as the MatMul node";
    }
  }
}

TEST_F(GraphTransformationTests, FuseMatmulBNWithEmptyOptionalOutputWithInBetweenNodes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-with-reshape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "MatMul") {
      expected_output_name = node.OutputDefs()[0]->Name();
    } else if (node.OpType() == "BatchNormalization") {
      node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg("", nullptr));
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), expected_output_name)
          << "fusion should produce the same output name as the MatMul node";
    }
  }
}

// should not fuse
TEST_F(GraphTransformationTests, FuseMatmulBNWithOptionalOutputWithInBetweenNodes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-with-reshape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "BatchNormalization") {
      // additional non-empty output to batchNormalization
      ONNX_NAMESPACE::TypeProto optional_output_tensor_type;
      optional_output_tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TypeProto::kTensorType);
      auto& arg = graph.GetOrCreateNodeArg("bn_optional_output", &optional_output_tensor_type);
      node.MutableOutputDefs().push_back(&arg);
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

TEST_F(GraphTransformationTests, FuseMatmulBNDirectly) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-directly.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "BatchNormalization") {
      expected_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), expected_output_name)
          << "fusion should produce the same output name as the last node";
    }
  }
}

TEST_F(GraphTransformationTests, FuseMatmulBNWithOnlyReshape) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-only-reshape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "MatMul") {
      expected_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), expected_output_name)
          << "fusion should produce the same output name as the MatMul node";
    }
  }
}

TEST_F(GraphTransformationTests, FuseMatmulBNWithOnlyTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-only-transpose.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::string expected_output_name;
  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "MatMul") {
      expected_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), expected_output_name)
          << "fusion should produce the same output name as the MatMul node";
    }
  }
}

TEST_F(GraphTransformationTests, FuseMatmulBNWithoutBatchNormalization) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-only-transpose.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  GraphViewer graphViewer(graph);
  for (auto& node_index : graphViewer.GetNodesInTopologicalOrder()) {
    auto& node = *graph.GetNode(node_index);
    if (node.OpType() == "BatchNormalization") {
      graph_utils::RemoveNode(graph, node);
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["MatMul"], 1);
}

// should not fuse
TEST_F(GraphTransformationTests, FuseMatmulBNWithNonIgnorableNode) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-matmul-bn-non-ignorable-node.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<MatmulBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["BatchNormalization"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

TEST_F(GraphTransformationTests, DontFuseConvWithBNWithOptionalOutputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  // add an optional output to the BN node. should not fuse if this is present
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "BatchNormalization") {
      auto mean_input = node.InputDefs()[3];
      auto& mean_output = graph.GetOrCreateNodeArg(mean_input->Name() + ".output", mean_input->TypeAsProto());
      node.MutableOutputDefs().push_back(&mean_output);
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 1);
}

TEST_F(GraphTransformationTests, FuseConvBNMulAddUnsqueeze) {
  std::vector<std::basic_string<ORTCHAR_T>> test_models = {ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze.onnx"),
                                                           ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze.negative_axes.onnx"),
                                                           ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze-no-bias.onnx")};
  for (const auto& model : test_models) {
    PathString model_uri = PathString(MODEL_FOLDER) + model;

    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>()));
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
    ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvMulFusion>()));
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);
    ASSERT_TRUE(op_to_count["Mul"] == 0);
    ASSERT_TRUE(op_to_count["Add"] == 0);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  }
}

TEST_F(GraphTransformationTests, DivMulFusion) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/div_mul.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 5);
  ASSERT_TRUE(op_to_count["Mul"] == 5);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<DivMulFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 5);
  ASSERT_TRUE(op_to_count["Mul"] == 2);
}

TEST_F(GraphTransformationTests, NotWhereFusion) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/not_where.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Not"] == 4);
  ASSERT_TRUE(op_to_count["Where"] == 5);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<NotWhereFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Where"] == 5);
  ASSERT_TRUE(op_to_count["Not"] == 1);  // can't remove Not if it is graph output/ has consumer that's not where
}

#if (defined(USE_CUDA) || defined(USE_JSEP)) && !defined(DISABLE_CONTRIB_OPS)
// Conv->Add->Relu will be transformed to FusedConv
TEST_F(GraphTransformationTests, FuseCudaConvAddRelu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);   // Add removed from graph
  ASSERT_TRUE(op_to_count["Relu"] == 0);  // Relu removed from graph
}

// Currently the ConvAddRelu fusion is only backed by a float kernel for the
// the CUDA EP.

// When we see the corresponding pattern for the fp16 data type, the fusion
// should not be triggered as there is no kernel to back the fused pattern.

// TODO(hasesh): Limit the test to using the CUDA EP for now as the level of
// data type support in other compatible EPs is still yet to be ascertained.

// TODO(hasesh): If at all the fp16 type is supported for the fusion, adjust/remove
// this test.
TEST_F(GraphTransformationTests, FuseCudaConvAddRelu_UnsupportedType) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu_fp16.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Add"], 1);
  ASSERT_EQ(op_to_count["Relu"], 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Add"], 1);   // Add not removed from graph (fusion not triggered)
  ASSERT_EQ(op_to_count["Relu"], 1);  // Relu not removed from graph (fusion not triggered)
}

// Conv->Add->Relu will be left intact since there is Identity depend on Add
TEST_F(GraphTransformationTests, FuseCudaConvAddReluIdentity) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu_identity.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  ASSERT_TRUE(op_to_count["Identity"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);       // Add remains
  ASSERT_TRUE(op_to_count["Relu"] == 1);      // Relu remains
  ASSERT_TRUE(op_to_count["Identity"] == 1);  // Identity remains
}

// Conv->Add will be left intact since there is no Relu follows
TEST_F(GraphTransformationTests, FuseCudaConvAdd) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);  // Add remains, no transform applied to the graph
}

#endif

#if !defined(DISABLE_CONTRIB_OPS)
// Conv->Add->Relu will be transformed to FusedConv
TEST_F(GraphTransformationTests, FuseCpuConvAddRelu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);   // Add removed from graph
  ASSERT_TRUE(op_to_count["Relu"] == 0);  // Relu removed from graph
}

// Conv->Add->Relu will be partly fused  to Conv_Add->Relu since there is Identity depend on Add
TEST_F(GraphTransformationTests, FuseCpuConvAddReluIdentity) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu_identity.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  ASSERT_TRUE(op_to_count["Identity"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);       // Add removed
  ASSERT_TRUE(op_to_count["Relu"] == 1);      // Relu remains
  ASSERT_TRUE(op_to_count["Identity"] == 1);  // Identity remains
}

// Conv->Add will be transformed to FusedConv
TEST_F(GraphTransformationTests, FuseCpuConvAdd) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger_));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);  // Add removed
}

#endif

#if !defined(DISABLE_CONTRIB_OPS)
TEST_F(GraphTransformationTests, FuseConvActivation) {
  std::unordered_map<PathString, std::string> model_to_op_name{{ORT_TSTR("fusion/conv_relu.onnx"), "Relu"},
                                                               {ORT_TSTR("fusion/conv_relu_opset12.onnx"), "Relu"},
                                                               {ORT_TSTR("fusion/conv_clip.onnx"), "Clip"},
                                                               {ORT_TSTR("fusion/conv_sigmoid.onnx"), "Sigmoid"},
                                                               {ORT_TSTR("fusion/conv_tanh.onnx"), "Tanh"},
                                                               {ORT_TSTR("fusion/conv_leakyrelu.onnx"), "LeakyRelu"},
                                                               {ORT_TSTR("fusion/conv_hardsigmoid.onnx"), "HardSigmoid"}};
  for (const auto& model : model_to_op_name) {
    PathString model_uri = PathString(MODEL_FOLDER) + model.first;
    SCOPED_TRACE(ORT_TSTR("model file: ") + model_uri);
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();
#ifdef USE_CUDA
    for (auto& node : p_model->MainGraph().Nodes()) {
      node.SetExecutionProviderType(kCudaExecutionProvider);
    }
#elif defined(USE_ROCM)
    for (auto& node : p_model->MainGraph().Nodes()) {
      node.SetExecutionProviderType(kCudaExecutionProvider);
    }
#elif defined(USE_JSEP)
    for (auto& node : p_model->MainGraph().Nodes()) {
      node.SetExecutionProviderType(kJsExecutionProvider);
    }
#endif
    std::map<std::string, int> op_to_count_before_fusion = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count_before_fusion[model.second] >= 1);

    // Apply transformer
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

    std::map<std::string, int> op_to_count_after_fusion = CountOpsInGraph(graph);
#if defined(USE_CUDA) || defined(USE_ROCM)
    std::set<std::string> cuda_rocm_supported = {"Relu"};
    if (cuda_rocm_supported.find(model.second) == cuda_rocm_supported.end()) {
      ASSERT_EQ(op_to_count_before_fusion[model.second], op_to_count_after_fusion[model.second]);
    } else {
      ASSERT_EQ(op_to_count_after_fusion[model.second], 0);
    }
#elif defined(USE_JSEP)
    std::set<std::string> js_supported = {"Relu", "Clip", "Sigmoid", "Tanh", "LeakyRelu"};
    if (js_supported.find(model.second) == js_supported.end()) {
      ASSERT_EQ(op_to_count_before_fusion[model.second], op_to_count_after_fusion[model.second]);
    } else {
      ASSERT_TRUE(op_to_count_after_fusion[model.second] == 0);
    }
#else
    ASSERT_TRUE(op_to_count_after_fusion[model.second] == 0);
#endif
  }
}

TEST_F(GraphTransformationTests, FuseConvClip11Activation) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_clip11.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Clip"], 3);

  // Apply transformer
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Clip"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      EXPECT_TRUE(node.Name() == "Conv1") << "Conv1 should not have been fused as 'min' input to Clip was mutable.";
    }

    if (node.OpType() == "FusedConv") {
      const ONNX_NAMESPACE::AttributeProto& attr_proto = node.GetAttributes().at("activation_params");
      const auto& params = attr_proto.floats();
      // check expected values for each. Conv0 is explicitly specified. Conv2 are defaults
      if (node.Name() == "Conv0") {
        EXPECT_EQ(params.Get(0), -1.f);
        EXPECT_EQ(params.Get(1), 10.f);
      } else if (node.Name() == "Conv2") {
        EXPECT_EQ(params.Get(0), std::numeric_limits<float>::lowest());
        EXPECT_EQ(params.Get(1), std::numeric_limits<float>::max());
      } else {
        FAIL() << "Unexpected fused node name: '" << node.Name() << "'.";
      }
    }
  }
}

TEST_F(GraphTransformationTests, FuseConvActivationPreservingAttributes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_with_padding_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Relu"], 1);

  // Apply transformer
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Relu"], 0);

  ASSERT_EQ(graph.NumberOfNodes(), 1);
  const auto& fused_conv_node = *graph.Nodes().begin();
  ASSERT_EQ(fused_conv_node.OpType(), "FusedConv");

  auto check_ints_attr =
      [&fused_conv_node](const std::string& attr_name, gsl::span<const int64_t> expected_values) {
        const auto& attrs = fused_conv_node.GetAttributes();
        const auto attr_it = attrs.find(attr_name);
        ASSERT_NE(attr_it, attrs.end());
        EXPECT_THAT(attr_it->second.ints(), testing::ContainerEq(expected_values));
      };

  check_ints_attr("pads", AsSpan<int64_t>({1, 1, 1, 1}));
  check_ints_attr("kernel_shape", AsSpan<int64_t>({3, 3}));
}
#endif  // !defined(DISABLE_CONTRIB_OPS)

TEST_F(GraphTransformationTests, FuseConvMulNoBias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-mul-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvMulFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, FuseConvAddNoBias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

// if IR version is 4 or higher the weights can be overridden if there's a matching graph input.
// check that we don't fuse if that is the case
TEST_F(GraphTransformationTests, NegativeFuseConvAddNoBias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/negative-fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Nodes are not fused because the weights to conv/add are not constants (they appear in the graph inputs).
  // Unsqueeze is also not eliminated as the initializer that is its input is also not constant
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] != 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] != 0);
}

static void TestFuseConvAddMul(logging::Logger& logger, const PathChar* model_uri) {
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, logger));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvMulFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
}

TEST_F(GraphTransformationTests, FuseConvAddMul3D) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-3d.onnx";
  TestFuseConvAddMul(*logger_, model_uri);
}

TEST_F(GraphTransformationTests, FuseConvAddMul1D) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-1d.onnx";
  TestFuseConvAddMul(*logger_, model_uri);
}

TEST_F(GraphTransformationTests, FuseConvAddMul3D_2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-3d-2.onnx";
  TestFuseConvAddMul(*logger_, model_uri);
}

TEST_F(GraphTransformationTests, FuseConvAddMul1D_2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-1d-2.onnx";
  TestFuseConvAddMul(*logger_, model_uri);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_two_input) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/2Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_three_input) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

// Matmul+Add with shape [k]*[k,N]+[N], won't do the fusion
// We can do the fusion by changing shape to [1,k]*[k,N]+[1,N], then add a reshape [1,N]=>[N]
// This will bring extra cost. And there's only very limited gain to fuse Matmul+Add to Gemm
// Since the basic implementation is almost same
TEST_F(GraphTransformationTests, MatMulAddFusion_negitive_case) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/neg_model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
}

// Matmul+Add with shape [M,k]*[k,N]+[1,4], won't do the fusion
// 1,4 is not uni-directionally broadcast
TEST_F(GraphTransformationTests, MatMulAddFusion_NotBroadcastable) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/matmul_add_not_broadcastable.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_MissingShape) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/matmul_add_missing_shape.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["MatMul"], 1);
  ASSERT_EQ(op_to_count["Add"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 0);
}

#ifndef DISABLE_CONTRIB_OPS
TEST_F(GraphTransformationTests, Gemm_Relu_three_input) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/gemm_relu.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GemmActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);
}

TEST_F(GraphTransformationTests, TransposeMatmulFusion) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 1);
}

TEST_F(GraphTransformationTests, TransposeCastMatmulFusion) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion0.onnx",  // Test fusion from the right input
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion1.onnx",  // Test fusion from the left input
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion2.onnx",  // Test fusion both from the left and right inputs
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion3.onnx",  // Cast nodes feed multiple MatMul nodes.
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion4.onnx",  // Cast nodes feed one MatMul node and
                                                                    // the Transpose nodes feed another MatMul node.
      MODEL_FOLDER "fusion/transpose_cast_matmul_4d_fusion5.onnx"   // One Cast node and one Transpose node feed each
                                                                    // MatMul nodes.
  };
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();
    std::map<std::string, int> orig_op_to_count = CountOpsInGraph(graph);  // Original op count

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatmulTransposeFusion>(),
                                                       TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["Transpose"] == 0);
    ASSERT_TRUE(op_to_count["MatMul"] == 0);
    ASSERT_TRUE(op_to_count["Cast"] == orig_op_to_count["Cast"]);
    ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == orig_op_to_count["MatMul"]);
  }
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionOnTwoTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_2_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatmulTransposeFusion>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "FusedMatMul");
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionOnThreeTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_3_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatmulTransposeFusion>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "FusedMatMul");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
}

TEST_F(GraphTransformationTests, TransposeMatmulNoFusionOnInvalidInput) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_perm.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_default_perm.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_datatype_int32.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_datatype_int64.onnx",
  };
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(
        std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["Transpose"], 1);
    ASSERT_EQ(op_to_count["MatMul"], 1);
    ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], 0);
  }
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionFromTransposeMatMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_2d_fusion_from_transpose_scale_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  float expected_alpha;
  {
    auto transpose_scale_matmul_node =
        std::find_if(
            graph.Nodes().cbegin(), graph.Nodes().cend(),
            [](const Node& node) { return node.Name() == "FusedMatMul"; });
    ASSERT_NE(transpose_scale_matmul_node, graph.Nodes().cend());
    expected_alpha = transpose_scale_matmul_node->GetAttributes().at("alpha").f();
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], 1);

  auto& transpose_scale_matmul_node = *graph.Nodes().begin();
  ASSERT_EQ(transpose_scale_matmul_node.OpType(), "FusedMatMul");
  ASSERT_FALSE(static_cast<bool>(transpose_scale_matmul_node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(transpose_scale_matmul_node.GetAttributes().at("transB").i()));
  ASSERT_EQ(transpose_scale_matmul_node.GetAttributes().at("alpha").f(), expected_alpha);
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionWithPreservedTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/transpose_matmul_2d_fusion_with_preserved_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], 1);

  ASSERT_FALSE(graph.GraphResolveNeeded());
}

TEST_F(GraphTransformationTests, TransposeMatmulTransBatchFusion) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion1.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion2.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion3.onnx",
  };
  const std::vector<std::pair<int64_t, int64_t>> trans_batch_attrs = {
      {1, 0},
      {1, 1},
      {1, 1},
  };
  size_t index = 0;
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(
        std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["Transpose"], 0);
    ASSERT_EQ(op_to_count["MatMul"], 0);
    ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "FusedMatMul") {
        auto attrs = node.GetAttributes();
        int64_t trans_batch_a = 0;
        if (attrs.find("transBatchA") != attrs.end()) {
          trans_batch_a = attrs.at("transBatchA").i();
        }
        int64_t trans_batch_b = 0;
        if (attrs.find("transBatchB") != attrs.end()) {
          trans_batch_b = attrs.at("transBatchB").i();
        }
        ASSERT_EQ(trans_batch_a, trans_batch_attrs[index].first);
        ASSERT_EQ(trans_batch_b, trans_batch_attrs[index].second);
        break;
      }
    }
    ++index;
  }
}

TEST_F(GraphTransformationTests, TransposeMatmulTransBatchNoFusion) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion_invalid_case1.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion_invalid_case2.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_trans_batch_fusion_invalid_case3.onnx",
  };
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();
    std::map<std::string, int> orig_op_to_count = CountOpsInGraph(graph);

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(
        std::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["Transpose"], orig_op_to_count["Transpose"]);
    ASSERT_EQ(op_to_count["MatMul"], orig_op_to_count["MatMul"]);
    ASSERT_EQ(op_to_count["com.microsoft.FusedMatMul"], orig_op_to_count["com.microsoft.FusedMatMul"]);
  }
}

TEST_F(GraphTransformationTests, Gemm_LeakyRelu_Fusion) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "gemm_activation_fusion/gemm_activation_fusion.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GemmActivationFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["LeakyRelu"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedGemm"] == 1);
}
#endif

// (A')'B' = AB'
TEST_F(GraphTransformationTests, GemmTransposeFusion2Inputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_2inputs_transposed.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "A");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
}

// (A')'B' = AB' where transpose has multiple consumers
TEST_F(GraphTransformationTests, GemmTransposeFusion2OutputsFromTranspose) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_2outputs_from_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(op_to_count["Identity"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(op_to_count["Identity"], 1);

  auto gemm_node =
      std::find_if(
          graph.Nodes().cbegin(), graph.Nodes().cend(),
          [](const Node& node) { return node.Name() == "Gemm_transformed"; });

  auto& node = *gemm_node;
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "tp0");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
}

// (A')'B' = AB' and  (B')'C = BC where transpose has multiple consumers
TEST_F(GraphTransformationTests, GemmTransposeFusion2OutputsFromTransposeTo2Gemms) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_2outputs_from_transpose_to_2gemms.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2);
  ASSERT_EQ(op_to_count["Gemm"], 2);
  ASSERT_EQ(op_to_count["Identity"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 2);
  ASSERT_EQ(op_to_count["Identity"], 1);

  auto gemm1_node =
      std::find_if(
          graph.Nodes().cbegin(), graph.Nodes().cend(),
          [](const Node& node) { return node.Name() == "Gemm1_transformed"; });

  auto& node1 = *gemm1_node;
  ASSERT_TRUE(node1.OpType() == "Gemm");
  ASSERT_TRUE(static_cast<bool>(node1.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node1.GetAttributes().at("transB").i()));
  auto new_input_defs1 = node1.InputDefs();
  ASSERT_TRUE(new_input_defs1[0]->Name() == "tp0");
  ASSERT_TRUE(new_input_defs1[1]->Name() == "B");

  auto gemm2_node =
      std::find_if(
          graph.Nodes().cbegin(), graph.Nodes().cend(),
          [](const Node& node) { return node.Name() == "Gemm2_transformed"; });

  auto& node2 = *gemm2_node;
  ASSERT_TRUE(node2.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node2.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node2.GetAttributes().at("transB").i()));
  auto new_input_defs2 = node2.InputDefs();
  ASSERT_TRUE(new_input_defs2[0]->Name() == "B");
  ASSERT_TRUE(new_input_defs2[1]->Name() == "C");
}

// (A'B)' = B'A
TEST_F(GraphTransformationTests, GemmTransposeFusionOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_output_transposed.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "B");
  ASSERT_TRUE(new_input_defs[1]->Name() == "A");
}

// ((A')'B')' = BA'
TEST_F(GraphTransformationTests, GemmTransposeFusionInputOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_inputs_output_transposed.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 3);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "B");
  ASSERT_TRUE(new_input_defs[1]->Name() == "A");
}

// (A'(B'))' = BA
TEST_F(GraphTransformationTests, GemmTransposeFusionInputOutput2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_transpose_inputs_output_transposed_2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 2);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmTransposeFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  auto new_input_defs = node.InputDefs();
  ASSERT_TRUE(new_input_defs[0]->Name() == "B");
  ASSERT_TRUE(new_input_defs[1]->Name() == "A");
}

// Sum(Gemm(A, B, _), C) -> Gemm(A, B, C)
TEST_F(GraphTransformationTests, GemmSumFusionBasic) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_basic.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  ASSERT_EQ(node.GetAttributes().at("alpha").f(), 1.0);
  ASSERT_EQ(node.GetAttributes().at("beta").f(), 1.0);
  auto new_input_defs = node.InputDefs();
  ASSERT_EQ(new_input_defs.size(), 3u);
  ASSERT_TRUE(new_input_defs[0]->Name() == "A");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
  ASSERT_TRUE(new_input_defs[2]->Name() == "C");
  auto new_output_defs = node.OutputDefs();
  ASSERT_EQ(new_output_defs.size(), 1u);
  ASSERT_TRUE(new_output_defs[0]->Name() == "output");
}

// Sum(Gemm(A, B, _), C) -> Gemm(A, B, C), with attributes
TEST_F(GraphTransformationTests, GemmSumFusionAttributes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_attributes.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  ASSERT_EQ(node.GetAttributes().at("alpha").f(), 3.5);
  ASSERT_EQ(node.GetAttributes().at("beta").f(), 1.0);
  auto new_input_defs = node.InputDefs();
  ASSERT_EQ(new_input_defs.size(), 3u);
  ASSERT_TRUE(new_input_defs[0]->Name() == "A");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
  ASSERT_TRUE(new_input_defs[2]->Name() == "C");
  auto new_output_defs = node.OutputDefs();
  ASSERT_EQ(new_output_defs.size(), 1u);
  ASSERT_TRUE(new_output_defs[0]->Name() == "output");
}

// Identity(Sum(Gemm(Identity(A), Identity(B), _), Identity(C)) should still fuse Gemm/Sum internally.
TEST_F(GraphTransformationTests, GemmSumFusionInternalNodes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_internal_nodes.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(op_to_count["Identity"], 4);
  ASSERT_EQ(graph.NumberOfNodes(), 6);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(op_to_count["Identity"], 4);
  ASSERT_EQ(graph.NumberOfNodes(), 5);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
      ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
      ASSERT_EQ(node.GetAttributes().at("alpha").f(), 1.0);
      ASSERT_EQ(node.GetAttributes().at("beta").f(), 1.0);

      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 3u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "tp0");
      ASSERT_TRUE(new_input_defs[1]->Name() == "tp1");
      ASSERT_TRUE(new_input_defs[2]->Name() == "tp3");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "tp4");
    }
  }
}

// Sum(Gemm(A, B, C), D) does not perform transform.
TEST_F(GraphTransformationTests, GemmSumFusionNoFusionCUsed) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_no_fusion_c_used.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // Assert that the Sum still exists. Fusion should not occur with this pattern.
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 3u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "A");
      ASSERT_TRUE(new_input_defs[1]->Name() == "B");
      ASSERT_TRUE(new_input_defs[2]->Name() == "C");
    } else if (node.OpType() == "Sum") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[1]->Name() == "D");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "output");
    } else {
      FAIL();
    }
  }
}

// Sum(Gemm(A, B), C, D) does not perform transform.
TEST_F(GraphTransformationTests, GemmSumFusionNoFusionSumMultipleInputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_no_fusion_sum_multiple_inputs.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // Assert that the Sum still exists. Fusion should not occur with this pattern.
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "A");
      ASSERT_TRUE(new_input_defs[1]->Name() == "B");
    } else if (node.OpType() == "Sum") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 3u);
      ASSERT_TRUE(new_input_defs[1]->Name() == "C");
      ASSERT_TRUE(new_input_defs[2]->Name() == "D");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "output");
    } else {
      FAIL();
    }
  }
}

// Sum(Gemm(A, B, _), C) -> Gemm(A, B, C), with broadcast.
TEST_F(GraphTransformationTests, GemmSumFusionBroadcast) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_fusion_broadcast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 0);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "Gemm");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transB").i()));
  ASSERT_EQ(node.GetAttributes().at("alpha").f(), 1.0);
  ASSERT_EQ(node.GetAttributes().at("beta").f(), 1.0);
  auto new_input_defs = node.InputDefs();
  ASSERT_EQ(new_input_defs.size(), 3u);
  ASSERT_TRUE(new_input_defs[0]->Name() == "A");
  ASSERT_TRUE(new_input_defs[1]->Name() == "B");
  ASSERT_TRUE(new_input_defs[2]->Name() == "C");
  auto new_output_defs = node.OutputDefs();
  ASSERT_EQ(new_output_defs.size(), 1u);
  ASSERT_TRUE(new_output_defs[0]->Name() == "output");
}

// Sum(Gemm(A, B, _), C), with invalid broadcasting (no fusion performed).
TEST_F(GraphTransformationTests, GemmSumFusionNoFusionBroadcastFailure) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_no_fusion_broadcast_failure.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "A");
      ASSERT_TRUE(new_input_defs[1]->Name() == "B");
    } else if (node.OpType() == "Sum") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[1]->Name() == "C");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "output");
    } else {
      FAIL();
    }
  }
}

// Sum(Gemm(A, B, _), C) where intermediate Gemm output is used, so fusion cannot be performed.
TEST_F(GraphTransformationTests, GemmSumFusionNoFusionOriginalGemmOutputUsed) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gemm_sum_no_fusion_original_gemm_output_used.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<GemmSumFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Sum"], 1);
  ASSERT_EQ(op_to_count["Gemm"], 1);
  ASSERT_EQ(graph.NumberOfNodes(), 2);

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "Gemm") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[0]->Name() == "A");
      ASSERT_TRUE(new_input_defs[1]->Name() == "B");
    } else if (node.OpType() == "Sum") {
      auto new_input_defs = node.InputDefs();
      ASSERT_EQ(new_input_defs.size(), 2u);
      ASSERT_TRUE(new_input_defs[1]->Name() == "C");
      auto new_output_defs = node.OutputDefs();
      ASSERT_EQ(new_output_defs.size(), 1u);
      ASSERT_TRUE(new_output_defs[0]->Name() == "output");
    } else {
      FAIL();
    }
  }
}

TEST_F(GraphTransformationTests, FuseConvBnAddMulFloat16) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-add-mul-float16.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvBNFusion>()));
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvMulFusion>()));
  ASSERT_STATUS_OK(session_object.RegisterGraphTransformer(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(session_object.Initialize());

  NameMLValMap feeds;
  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  OrtValue ml_value_x;

  auto x_f = MLFloat16(1.0f);
  std::vector<int64_t> dims_x = {1, 1, 3, 3};
  std::vector<MLFloat16> values_x;
  for (int i = 0; i < 9; ++i) {
    values_x.push_back(x_f);
  }
  CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0],
                           dims_x, values_x, &ml_value_x);
  feeds.insert(std::make_pair("X", ml_value_x));

  std::vector<std::string> output_names;
  output_names.push_back("PROD");
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));

  auto prod_f = MLFloat16(6.0f);
  std::vector<int64_t> expected_dims_prod = {1, 1, 2, 2};
  std::vector<MLFloat16> expected_values_prod;
  for (int i = 0; i < 4; ++i) {
    expected_values_prod.push_back(prod_f);
  }

  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims_prod);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<MLFloat16> found(rtensor.Data<MLFloat16>(),
                                     rtensor.Data<MLFloat16>() + expected_dims_prod.size());
  ASSERT_EQ(expected_values_prod, found);
}

TEST_F(GraphTransformationTests, ReluClip6Fusion) {
  // Clip op schema changed for opset version 11. Until Clip op is updated in ORT hard coding this model to use
  // older opset.
  Model model("ReluClip6Fusion", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 10}},
              {}, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // 3 paths in the model, each with Relu followed by Clip
  // One has a Clip with min of 0  (remove Relu)
  // One have a Clip with a min > 1 (remove Relu)
  // One has a Clip with min < 0 (remove Relu and update Clip 'min' to 0)
  auto& input0 = graph.GetOrCreateNodeArg("input_0", &input_tensor_type);
  auto& input1 = graph.GetOrCreateNodeArg("input_1", &input_tensor_type);
  auto& input2 = graph.GetOrCreateNodeArg("input_2", &input_tensor_type);

  auto& relu0_output = graph.GetOrCreateNodeArg("relu0_output", &input_tensor_type);
  auto& relu1_output = graph.GetOrCreateNodeArg("relu1_output", &input_tensor_type);
  auto& relu2_output = graph.GetOrCreateNodeArg("relu2_output", &input_tensor_type);

  auto& clip0_output = graph.GetOrCreateNodeArg("clip0_output", &input_tensor_type);
  auto& clip1_output = graph.GetOrCreateNodeArg("clip1_output", &input_tensor_type);
  auto& clip2_output = graph.GetOrCreateNodeArg("clip2_output", &input_tensor_type);

  graph.AddNode("relu0", "Relu", "Relu to eliminate", {&input0}, {&relu0_output});
  graph.AddNode("relu1", "Relu", "Relu to not eliminate", {&input1}, {&relu1_output});
  graph.AddNode("relu2", "Relu", "Relu to eliminate and update 'min' of following Clip", {&input2}, {&relu2_output});

  auto& clip0 = graph.AddNode("clip0", "Clip", "Clip with min 0", {&relu0_output}, {&clip0_output});
  clip0.AddAttribute("min", 0.f);
  clip0.AddAttribute("max", 1.f);

  auto& clip1 = graph.AddNode("clip1", "Clip", "Clip with min 1", {&relu1_output}, {&clip1_output});
  clip1.AddAttribute("min", 1.f);
  clip1.AddAttribute("max", 1.f);

  auto& clip2 = graph.AddNode("clip2", "Clip", "Clip with min -1", {&relu2_output}, {&clip2_output});
  clip2.AddAttribute("min", -1.f);
  clip2.AddAttribute("max", 1.f);

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 3);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<FuseReluClip>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);

  // make sure the Clip nodes were updated to have a 'min' >= 0
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Clip") {
      auto* min = graph_utils::GetNodeAttribute(node, "min");
      ASSERT_TRUE(min->f() >= 0.f);
    }
  }
}

// test handling of Clip 11
TEST_F(GraphTransformationTests, ReluClip11Fusion) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 11;
  Model model("ReluClip6Fusion", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              *logger_);  //, true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 11}}, {});
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto float16_tensor_type;
  float16_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  float16_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // 4 paths in the model, each with Relu followed by Clip to test different aspects of Clip 11 handling
  // One has a Clip with mutable 'min' (don't fuse)
  // One has a Clip with constant 'min' < 0 (fuse and update 'min')
  // One has a Clip with constant 'min' > 0 (fuse and leave 'min')
  // One has a Clip with no 'min' (fuse and update to set min to 0 using type info from 'input')
  auto& input0 = graph.GetOrCreateNodeArg("input_0", &input_tensor_type);
  auto& input1 = graph.GetOrCreateNodeArg("input_1", &float16_tensor_type);
  auto& input2 = graph.GetOrCreateNodeArg("input_2", &input_tensor_type);
  auto& input3 = graph.GetOrCreateNodeArg("input_3", &input_tensor_type);

  auto& min_input_0 = graph.GetOrCreateNodeArg("min_input_0", &input_tensor_type);
  auto& min_input_1 = graph.GetOrCreateNodeArg("min_input_1", &float16_tensor_type);
  auto& min_input_2 = graph.GetOrCreateNodeArg("min_input_2", &input_tensor_type);

  // add initializer for min_input_1 so it's constant
  TensorProto const_min_1;
  Initializer i1(TensorProto_DataType_FLOAT16, "min_input_1", AsSpan<int64_t>({1}));
  i1.data<MLFloat16>()->val = math::floatToHalf(-1.f);
  i1.ToProto(const_min_1);
  graph.AddInitializedTensor(const_min_1);

  TensorProto const_min_2;
  Initializer i2(TensorProto_DataType_FLOAT, "min_input_2", AsSpan<int64_t>({1}));
  *i2.data<float>() = 1.f;
  i2.ToProto(const_min_2);
  graph.AddInitializedTensor(const_min_2);

  auto& relu0_output = graph.GetOrCreateNodeArg("relu0_output", &input_tensor_type);
  auto& relu1_output = graph.GetOrCreateNodeArg("relu1_output", &float16_tensor_type);
  auto& relu2_output = graph.GetOrCreateNodeArg("relu2_output", &input_tensor_type);
  auto& relu3_output = graph.GetOrCreateNodeArg("relu3_output", &input_tensor_type);

  auto& clip0_output = graph.GetOrCreateNodeArg("clip0_output", &input_tensor_type);
  auto& clip1_output = graph.GetOrCreateNodeArg("clip1_output", &float16_tensor_type);
  auto& clip2_output = graph.GetOrCreateNodeArg("clip2_output", &input_tensor_type);
  auto& clip3_output = graph.GetOrCreateNodeArg("clip3_output", &input_tensor_type);

  graph.AddNode("relu0", "Relu", "Relu0", {&input0}, {&relu0_output});
  graph.AddNode("relu1", "Relu", "Relu1", {&input1}, {&relu1_output});
  graph.AddNode("relu2", "Relu", "Relu2", {&input2}, {&relu2_output});
  graph.AddNode("relu3", "Relu", "Relu3", {&input3}, {&relu3_output});

  auto& clip0 = graph.AddNode("clip0", "Clip", "Clip with mutable min", {&relu0_output, &min_input_0}, {&clip0_output});
  auto& clip1 = graph.AddNode("clip1", "Clip", "Clip with constant min < 0", {&relu1_output, &min_input_1}, {&clip1_output});
  auto& clip2 = graph.AddNode("clip2", "Clip", "Clip with constant min > 0", {&relu2_output, &min_input_2}, {&clip2_output});
  auto& clip3 = graph.AddNode("clip3", "Clip", "Clip with no min", {&relu3_output}, {&clip3_output});

  graph.SetInputs({&input0, &input1, &input2, &input3, &min_input_0});
  ASSERT_STATUS_OK(graph.Resolve());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 4);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<FuseReluClip>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 1) << "All except the first Relu should have been fused";

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Relu") {
      EXPECT_TRUE(node.Name() == "relu0") << "relu0 should be the only Relu node left";
    }

    if (node.OpType() == "Clip") {
      auto* min_input = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());

      if (&node == &clip0) {
        EXPECT_TRUE(min_input == nullptr) << "clip0 should not have been fused as min_input_0 is not constant";
      } else {
        EXPECT_TRUE(min_input != nullptr)
            << node.Name() << " should have been fused and have a constant initializer for 'min'";

        auto type = min_input->data_type();

        if (&node == &clip1) {
          // fusion with float16 data and min set to 0
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT16);
          MLFloat16 value = *Initializer(*min_input, graph.ModelPath()).data<MLFloat16>();
          EXPECT_EQ(math::halfToFloat(value.val), 0.f) << "Min was not 0.f. Got:" << math::halfToFloat(value.val);
        } else if (&node == &clip2) {
          // fusion with float data and min untouched
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT);
          float value = *Initializer(*min_input, graph.ModelPath()).data<float>();
          EXPECT_EQ(value, 1.0) << "Min should have remained unchanged but is now " << value;
        } else if (&node == &clip3) {
          // fusion with no min so type comes from input
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT);
          float value = *Initializer(*min_input, graph.ModelPath()).data<float>();
          EXPECT_EQ(value, 0.f) << "Min was not 0.f. Got:" << value;

        } else {
          EXPECT_TRUE(false) << "Unexpected node " << node.Name();
        }
      }
    }
  }
}

TEST_F(GraphTransformationTests, ReluClip11FusionGHIssue9753) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/relu_clip_fusion_gh_issue_9753.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  // The model contains one Relu and one Clip
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  ASSERT_TRUE(op_to_count["Clip"] == 1);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<FuseReluClip>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  // After fusion, the model only contains Clip.
  ASSERT_TRUE(op_to_count["Relu"] == 0);
  ASSERT_TRUE(op_to_count["Clip"] == 1);
}

// Test Reshape Fusion with 2 constant initializers for Concat inputs.
TEST_F(GraphTransformationTests, ReshapeFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 4U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], 12);
      EXPECT_EQ(val[3], 64);
    }
  }
}

// Test Reshape Fusion with one constant initializer for Concat inputs.
TEST_F(GraphTransformationTests, ReshapeFusionOneConstTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_one_const.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], 768);
    }
  }
}

// Test Reshape Fusion with an internal node being the output of the graph.
TEST_F(GraphTransformationTests, ReshapeFusionInternalNodeIsOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_internal_node_is_graph_output.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

// Test Reshape Fusion where some of the internal nodes are reused:
// A Shape is used in two Gather's, and the third Gather is the graph output.
TEST_F(GraphTransformationTests, ReshapeFusionInternalReuseTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_internal_nodes_reused.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 5U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 128);
      EXPECT_EQ(val[2], 0);
      EXPECT_EQ(val[3], 0);
      EXPECT_EQ(val[4], -1);
    } else if (node.OpType() == "Shape") {
      EXPECT_EQ(node.Name(), "shape2");
    } else if (node.OpType() == "Gather") {
      EXPECT_EQ(node.Name(), "gather3");
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionGraphInputsTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_with_graph_inputs.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 1);
  ASSERT_EQ(op_to_count["Concat"], 1);
  ASSERT_EQ(op_to_count["Reshape"], 1);
}

TEST_F(GraphTransformationTests, ReshapeFusionMultipleValuesInInitializerSubgraphTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_multiple_values_in_initializer_tensor_1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count_orig = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The optimization does not apply.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 1);
      EXPECT_EQ(val[1], 200);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionMultipleValuesInInitializerAppliesTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_multiple_values_in_initializer_tensor_2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 1);
      EXPECT_EQ(val[1], 200);
      EXPECT_EQ(val[2], 0);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionAnotherGraphInput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_input_is_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The optimization does not apply.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
}

TEST_F(GraphTransformationTests, ReshapeFusionOverridableInitializer) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_overridable_initializer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count_orig = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The optimization does not apply.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count_orig, op_to_count);
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphMultipleOutputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_multiple_outputs.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // The optimization applies but certain paths with multiple outputs/graph outputs are not removed.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 3);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 1);
  ASSERT_EQ(op_to_count["Slice"], 1);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraph) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionWithSlice1) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_with_slice1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphNotTriggered) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_not_triggered.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Two of the branches leading to Concat are candidates to trigger the optimization
  // (Shape -> Gather -> Unsqueeze -> Concat).
  // But one of the subgraphs leading to the Concat node will not trigger the optimization
  // as an additional pad value of 1 is inserted thus making the inputs to the Concat -
  // [10], [20], and [1, 30]
  // Since the third branch will match the subgraph fusion, (it has more than 1 value in the tensor)
  // and hence the optimization will not be triggered eventually

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 3);
  ASSERT_EQ(op_to_count["Gather"], 2);
  ASSERT_EQ(op_to_count["Unsqueeze"], 2);
  ASSERT_EQ(op_to_count["Slice"], 1);
  ASSERT_EQ(op_to_count["Concat"], 1);
  ASSERT_EQ(op_to_count["Pad"], 1);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto == nullptr);  // No initializer as optimizer is not triggered
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphWithDiv) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_div.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Div"], 0);
  ASSERT_EQ(op_to_count["Squeeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphWithMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_mul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Mul"], 0);
  ASSERT_EQ(op_to_count["Squeeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionDistilBertTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/reshape_fusion_distillbert.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ReshapeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 4U);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], -1);
      EXPECT_EQ(val[2], 2);
      EXPECT_EQ(val[3], 4);
    }
  }
}

// Test eliminating redundant Concat-Slice pattern.
TEST_F(GraphTransformationTests, ConcatSliceEliminationTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "concat_slice_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConcatSliceElimination>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Slice"] == 0);
}

TEST_F(GraphTransformationTests, ExpandElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "expand_elimination.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Expand"] == 6);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ExpandElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Expand"] == 3);
}

TEST_F(GraphTransformationTests, CastElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "cast_elimination.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 7);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<CastElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 4);
}

TEST_F(GraphTransformationTests, PreShapeNodeElimination) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "pre_shape_node_elimination.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 3);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<PreShapeNodeElimination>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  ASSERT_TRUE(op_to_count["Cast"] == 2);

  // Assert that the remaining "Cast" nodes have different names than "cast2"
  bool names_are_different = true;
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Cast") {
      const std::string& node_name = node.Name();
      if (node_name == "cast") {
        names_are_different = false;
        break;
      }
    }
  }

  ASSERT_TRUE(names_are_different);

  auto pre_graph_checker = [](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 0);
    return Status::OK();
  };

  // cast is the first node.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* cast_out = builder.MakeIntermediate();
      auto* shape_out = builder.MakeIntermediate();
      auto* output = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Shape", {cast_out}, {shape_out});
      builder.AddNode("Identity", {shape_out}, {output});
    };

    auto rule_transformer = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer");
    ASSERT_STATUS_OK(rule_transformer->Register(std::make_unique<PreShapeNodeElimination>()));
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 13, *logger_, std::move(rule_transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

#ifndef DISABLE_CONTRIB_OPS

static void ValidateAttention(Graph& graph) {
  // Validate the merged weights (initializer) input for Attention node.
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Attention") {
      int64_t expected_heads = 2;
      ASSERT_TRUE(optimizer_utils::IsAttributeWithExpectedValue(node, "num_heads", expected_heads));

      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      auto initializer = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(initializer->size(), 192U);

      // Validate two rows (2x24 items) for sanity check.
      std::vector<double> expected_value = {
          -0.10791015625,
          -0.04193115234375,
          0.09051513671875,
          0.025787353515625,
          -0.11572265625,
          -0.126953125,
          -0.043304443359375,
          -0.02984619140625,
          0.022125244140625,
          -0.017730712890625,
          -0.03265380859375,
          -0.05108642578125,
          0.0423583984375,
          0.112060546875,
          0.080810546875,
          0.09375,
          -0.03643798828125,
          0.02862548828125,
          0.039764404296875,
          0.06097412109375,
          -0.002288818359375,
          -0.10797119140625,
          -0.01171875,
          0.041717529296875,

          0.033538818359375,
          -0.05755615234375,
          -0.04986572265625,
          -0.01558685302734375,
          -0.0352783203125,
          0.03546142578125,
          0.05218505859375,
          0.005565643310546875,
          -0.043182373046875,
          -0.05010986328125,
          -0.063720703125,
          -0.00824737548828125,
          0.1492919921875,
          0.048431396484375,
          -0.0482177734375,
          -0.1123046875,
          0.032196044921875,
          0.0135650634765625,
          0.020233154296875,
          -0.05084228515625,
          -0.011260986328125,
          -0.1241455078125,
          -0.0101165771484375,
          -0.00490570068359375};

      const float* data = initializer->data<float>();
      for (size_t i = 0; i < expected_value.size(); i++) {
        EXPECT_EQ(data[i], static_cast<float>(expected_value[i]));
      }

      tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[2]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      auto initializer2 = std::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(initializer2->size(), 24U);

      std::vector<double> expected_value2 = {
          -0.23681640625,
          -0.16552734375,
          0.2191162109375,
          -0.1756591796875,
          -0.03460693359375,
          -0.05316162109375,
          -0.336181640625,
          -0.253662109375,
          0.0246734619140625,
          0.011993408203125,
          0.0178375244140625,
          0.00998687744140625,
          0.0255126953125,
          0.076416015625,
          -0.040771484375,
          0.0107879638671875,
          -0.005893707275390625,
          -0.00916290283203125,
          0.04541015625,
          0.0159454345703125,
          -0.0029163360595703125,
          -0.03472900390625,
          0.0535888671875,
          0.0091094970703125};

      const float* data2 = initializer2->data<float>();
      for (size_t i = 0; i < expected_value2.size(); i++) {
        EXPECT_EQ(data2[i], static_cast<float>(expected_value2[i]));
      }
    }
  }
}

// Test Attention Fusion with int32 mask
TEST_F(GraphTransformationTests, AttentionFusionInt32Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_int32_mask.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  ValidateAttention(graph);
}

// Test Attention Fusion with int64 mask and symbolic batch dimension
TEST_F(GraphTransformationTests, AttentionFusionInt64Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_symbolic_batch.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Cast"], 1);  // Cast for int64 mask to int32
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  ValidateAttention(graph);
}

// Test Attention Fusion with float32 mask and no "cast" node in mask path
TEST_F(GraphTransformationTests, AttentionFusionFloat32Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_mask_no_cast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["Div"], 0);
  EXPECT_EQ(op_to_count["Sub"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  ValidateAttention(graph);
}

// Test GPT-2 Attention Fusion with past and unidirectional mask
TEST_F(GraphTransformationTests, AttentionFusionWithPastAndUnidirMaskTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_past_unidir.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    if (p_node->OpType().compare("Attention") == 0) {
      EXPECT_EQ(p_node->GetAttributes().at("unidirectional").i(), 1);
    }
  }
}

// Test Attention Fusion with past but no unidirectional mask
TEST_F(GraphTransformationTests, AttentionFusionWithPastAndNoUnidirMaskTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_past_no_unidir.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    if (p_node->OpType().compare("Attention") == 0) {
      EXPECT_EQ(p_node->GetAttributes().at("unidirectional").i(), 0);
    }
  }
}

// Test GPT-2 Attention Fusion with float32 mask
TEST_F(GraphTransformationTests, AttentionFusionGPTWithPastAndMaskTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gpt2_past_mask_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
}

// Test GPT-2 Attention Fusion without input mask
TEST_F(GraphTransformationTests, AttentionFusionGPTWithPastNoMaskTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gpt2_past_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
}

// Test GPT-2 Attention Fusion without input mask and past state
TEST_F(GraphTransformationTests, AttentionFusionGPTTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gpt2_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
}

TEST_F(GraphTransformationTests, AttentionFusionDistilBertTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/attention_distilbert.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<AttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["ReduceSum"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["Concat"], 0);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["Shape"], 0);
}

TEST_F(GraphTransformationTests, GeluFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestSwitchOrderFormat2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_format2_0.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_format2_1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2GraphInput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_format2_1_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2GraphOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/gelu_format2_0_with_bias_use_graph_output.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.BiasGelu"] == 0);
}

TEST_F(GraphTransformationTests, BiasGeluTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.BiasGelu"] == 1);
}

// BiasGelu allows input switching based on input dimensions.
// This test validates the input edges are plugged correct in the optimized graph.
TEST_F(GraphTransformationTests, BiasGeluSwitchedInputOrder) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion_format_2.onnx";

  // create inputs and outputs
  RandomValueGenerator random{};
  NameMLValMap feeds;

  OrtValue mlvalue_b_i;
  std::vector<int64_t> dims_b_i = {3072};
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims_b_i,
                       random.Uniform<float>(dims_b_i, 0.0f, 1.0f), &mlvalue_b_i);
  feeds.insert(std::make_pair("B_I", mlvalue_b_i));

  OrtValue mlvalue_a_i;
  std::vector<int64_t> dims_a_i = {3, 512, 3072};
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims_a_i,
                       random.Uniform<float>(dims_a_i, 0.0f, 1.0f), &mlvalue_a_i);
  feeds.insert(std::make_pair("A_I", mlvalue_a_i));

  std::vector<std::string> output_names;
  output_names.push_back("C");

  auto run_model_test = [&](TransformerLevel level, std::vector<OrtValue>& fetches) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    session_options.session_logid = "OptimizerTests";
    InferenceSession session{session_options, GetEnvironment()};
    ASSERT_STATUS_OK(session.Load(model_uri));
    ASSERT_STATUS_OK(session.Initialize());

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options, feeds, output_names, &fetches));
  };

  // run model with and w/o optimizations and compare the results
  std::vector<OrtValue> unoptimized_fetches;
  run_model_test(TransformerLevel::Default, unoptimized_fetches);

  std::vector<OrtValue> optimized_fetches;
  run_model_test(TransformerLevel::MaxLevel, optimized_fetches);

  // Compare results
  double per_sample_tolerance = 1e-3;
  double relative_per_sample_tolerance = 0.0;
  auto ret = CompareOrtValue(optimized_fetches[0], unoptimized_fetches[0], per_sample_tolerance, relative_per_sample_tolerance, false);
  EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
}

static void VerifyGeluApproximation(bool is_enabled, SessionOptions& session_options) {
  std::unique_ptr<CPUExecutionProvider> e =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  bool has_gelu_approximation = false;
  auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, session_options, *e.get(), {});
  for (auto& transformer : transformers) {
    if (transformer->Name() == "GeluApproximation") {
      has_gelu_approximation = true;
    }
  }

  EXPECT_EQ(has_gelu_approximation, is_enabled);
}

// Test session option configuration for DoubleQDQPairsRemover
TEST_F(GraphTransformationTests, DoubleQDQRemover_SessionOptionConfig) {
  auto verify_session_config = [&](bool is_enabled, SessionOptions& session_option) {
    std::unique_ptr<CPUExecutionProvider> cpu_ep = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    bool has_double_qdq_remover = false;
    auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level1, session_option, *cpu_ep.get(), {});
    for (auto& transformer : transformers) {
      if (transformer->Name() == "DoubleQDQPairsRemover") {
        has_double_qdq_remover = true;
      }
    }
    EXPECT_EQ(has_double_qdq_remover, is_enabled);
  };
  SessionOptions session_options;
  // DoubleQDQPairsRemover is enabled by default.
  verify_session_config(true, session_options);

  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableDoubleQDQRemover, "1"));
  verify_session_config(false, session_options);

  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableDoubleQDQRemover, "0"));
  verify_session_config(true, session_options);
}

// Test session option configuration for GeluApproximation
TEST_F(GraphTransformationTests, GeluApproximation_SessionOptionConfig) {
  SessionOptions session_options;

  // GeluApproximation is not enabled by default.
  VerifyGeluApproximation(false, session_options);

  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsEnableGeluApproximation, "1"));
  VerifyGeluApproximation(true, session_options);

  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsEnableGeluApproximation, "0"));
  VerifyGeluApproximation(false, session_options);
}

// Test DoubleQDQPairsRemover to remove unnecessary DQ->Q nodes in the middle
TEST_F(GraphTransformationTests, DoublQDQRemover_RemoveDupQDQ) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "qdq_optimization/dup_qdq.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DoubleQDQPairsRemover>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["QuantizeLinear"], 3);
  EXPECT_EQ(op_to_count["DequantizeLinear"], 4);

  std::string dq_scale_name_before_reshape_node;
  std::string zp_name_before_reshape_node;
  std::string dq_scale_name_after_reshape_node;
  std::string zp_name_after_reshape_node;
  for (auto& node : graph.Nodes()) {
    if (node.Name() == "dq_2") {
      dq_scale_name_before_reshape_node = node.InputDefs()[QDQ::InputIndex::SCALE_ID]->Name();
      zp_name_before_reshape_node = node.InputDefs()[QDQ::InputIndex::ZERO_POINT_ID]->Name();
    }
    if (node.Name() == "q_3") {
      dq_scale_name_after_reshape_node = node.InputDefs()[QDQ::InputIndex::SCALE_ID]->Name();
      zp_name_after_reshape_node = node.InputDefs()[QDQ::InputIndex::ZERO_POINT_ID]->Name();
    }
  }
  EXPECT_EQ(dq_scale_name_before_reshape_node.empty(), false);
  EXPECT_EQ(zp_name_before_reshape_node.empty(), false);
  EXPECT_EQ(dq_scale_name_before_reshape_node, dq_scale_name_after_reshape_node);
  EXPECT_EQ(zp_name_before_reshape_node, zp_name_after_reshape_node);
}

// Test Gelu -> FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "approximation/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluApproximation>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.Gelu"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.FastGelu"], 1);
}

// Test AddGeluFusion -> FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu_Add_Bias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "approximation/gelu_add_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluApproximation>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.BiasGelu"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.FastGelu"], 1);
}

// Test MatMul & AddGeluFusion -> MatMul & FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu_Add_MatMul) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "approximation/gelu_add_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluApproximation>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["com.microsoft.BiasGelu"], 0);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.FastGelu"], 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 2);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluUseGraphInputFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasUseGraphInputFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionTest2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluUseGraphInputFusionTest2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasFusionTest2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasUseGraphInputFusionTest2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu2_with_bias_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionWithCastsTest3) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/fast_gelu3_with_casts.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  // ORTModule for gpt2 model has two casts fused into one before FastGeluFusion
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<CommonSubexpressionElimination>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 2);

  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<FastGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Cast"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, QuickGelu) {
  // Sigmoid(x*alpha)*x, float
  {
    constexpr float alpha = 1.702f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<float>({}, {alpha});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();

      builder.AddNode("Mul", {input_arg, alpha_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {sigmoid_out, input_arg}, {mul_out_1});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "QuickGelu") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("alpha") != attrs.end());
          TEST_RETURN_IF_NOT(alpha == attrs.at("alpha").f());
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x*Sigmoid(alpha*x), MLFloat16
  {
    constexpr float alpha = -1.f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<MLFloat16>({}, {static_cast<MLFloat16>(alpha)});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "QuickGelu") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("alpha") != attrs.end());
          TEST_RETURN_IF_NOT(alpha == attrs.at("alpha").f());
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Sigmoid's output is consumed by other node.
  {
    constexpr float alpha = 1.702f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<float>({}, {alpha});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
      builder.AddNode("Identity", {sigmoid_out}, {identity_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // First Mul's output is consumed by other node.
  {
    constexpr float alpha = -1.f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<MLFloat16>({}, {static_cast<MLFloat16>(alpha)});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
      builder.AddNode("Identity", {mul_out_0}, {identity_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Sigmoid's output is a graph output.
  {
    constexpr float alpha = 1.702f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<float>({}, {alpha});
      auto* mul_out_0 = builder.MakeIntermediate();
      auto* sigmoid_out = builder.MakeOutput();
      auto* mul_out_1 = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // First Mul's output is a graph output.
  {
    constexpr float alpha = 1.702f;
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* alpha_arg = builder.MakeInitializer<float>({}, {alpha});
      auto* mul_out_0 = builder.MakeOutput();
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out_1 = builder.MakeOutput();

      builder.AddNode("Mul", {alpha_arg, input_arg}, {mul_out_0});
      builder.AddNode("Sigmoid", {mul_out_0}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out_1});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 2);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Sigmoid(x)*x, float
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out = builder.MakeOutput();

      builder.AddNode("Sigmoid", {input_arg}, {sigmoid_out});
      builder.AddNode("Mul", {sigmoid_out, input_arg}, {mul_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "QuickGelu") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("alpha") != attrs.end());
          TEST_RETURN_IF_NOT(1.0f == attrs.at("alpha").f());
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // x*Sigmoid(x), MLFloat16
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* sigmoid_out = builder.MakeIntermediate();
      auto* mul_out = builder.MakeOutput();

      builder.AddNode("Sigmoid", {input_arg}, {sigmoid_out});
      builder.AddNode("Mul", {input_arg, sigmoid_out}, {mul_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sigmoid"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.QuickGelu"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "QuickGelu") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("alpha") != attrs.end());
          TEST_RETURN_IF_NOT(1.0f == attrs.at("alpha").f());
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<QuickGeluFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

struct BiasSoftmaxFusionTester {
  std::shared_ptr<Model> p_model_;
  Status model_load_;
  onnxruntime::logging::Logger* logger_;
  onnxruntime::GraphTransformerManager graph_transformation_mgr_;

  bool GetAxis(const std::string op_type, const std::string name, int* axis) {
    for (auto& node : p_model_->MainGraph().Nodes()) {
      if (node.OpType() == op_type) {
        auto& softmax_attr = node.GetAttributes();
        if (softmax_attr.find(name) != softmax_attr.end()) {
          // found axis attribute
          auto& axis_attr = softmax_attr.at(name);
          *axis = (int)axis_attr.i();
          return true;
        }
      }
    }
    // not found
    return false;
  }

  BiasSoftmaxFusionTester(
      const PathString& model_uri,
      onnxruntime::logging::Logger* logger,
      const char* execution_provider = kCudaExecutionProvider) : logger_(logger), graph_transformation_mgr_{5} {
    model_load_ = Model::Load(model_uri, p_model_, nullptr, *logger_);

    // move to cuda since fusion only takes place in that case
    SetExecutionProvider(execution_provider);

    ORT_THROW_IF_ERROR(graph_transformation_mgr_.Register(
        std::make_unique<BiasSoftmaxFusion>(), TransformerLevel::Level2));
  }

  void SetExecutionProvider(const char* ep) {
    for (auto& node : p_model_->MainGraph().Nodes()) {
      node.SetExecutionProviderType(ep);
    }
  }

  void TestFusionOccurs(int expected_axis, bool expected_is_inner_broadcast) {
    ASSERT_STATUS_OK(model_load_);

    ASSERT_STATUS_OK(graph_transformation_mgr_.ApplyTransformers(p_model_->MainGraph(), TransformerLevel::Level2, *logger_));
    std::map<std::string, int> op_to_count = CountOpsInGraph(p_model_->MainGraph());

    ASSERT_EQ(op_to_count["Add"], 0);
    ASSERT_EQ(op_to_count["Softmax"], 0);
    ASSERT_EQ(op_to_count["com.microsoft.BiasSoftmax"], 1);

    int actual_axis = 1, actual_broadcast_type = 1;
    ASSERT_TRUE(GetAxis("BiasSoftmax", "axis", &actual_axis));
    ASSERT_EQ(actual_axis, expected_axis);

    ASSERT_TRUE(GetAxis("BiasSoftmax", "is_inner_broadcast", &actual_broadcast_type));
    ASSERT_EQ(actual_broadcast_type, expected_is_inner_broadcast ? 1 : 0);
  }

  void TestNoFusionOccurs() {
    ASSERT_STATUS_OK(model_load_);

    ASSERT_STATUS_OK(graph_transformation_mgr_.ApplyTransformers(p_model_->MainGraph(), TransformerLevel::Level2, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(p_model_->MainGraph());
    ASSERT_EQ(op_to_count["Add"], 1);
    ASSERT_EQ(op_to_count["Softmax"], 1);
    ASSERT_EQ(op_to_count["com.microsoft.BiasSoftmax"], 0);
  }
};

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_GpuOnly) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_simple.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get(), kCpuExecutionProvider);
  tester.TestNoFusionOccurs();
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_Simple_Rocm) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_simple.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get(), kRocmExecutionProvider);
  tester.TestFusionOccurs(1, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_Simple_Cuda) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_simple.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(1, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_Simple_Opset13_DefaultAxis) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_simple_no_axis_opset13.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(1, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_BFloat16_Input) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_bfloat16.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestNoFusionOccurs();
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_MiddleOnes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_middleones.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_ReversedInputs) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_middleones_reversed.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_BadAxis) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_middleones_badaxis.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestNoFusionOccurs();
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_AllLeadingOnes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_allleadingones.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, true);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_SomeLeadingOnes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_someleadingones.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, false);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_NoLeadingOnes) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/bias_softmax_fusion_noleadingones.onnx";
  BiasSoftmaxFusionTester tester(model_uri, logger_.get());
  tester.TestFusionOccurs(6, false);
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_OuterBroadcast) {
  auto pre_graph_checker = [&](Graph& graph) {
    for (auto& node : graph.Nodes()) node.SetExecutionProviderType(kCudaExecutionProvider);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Softmax"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.BiasSoftmax"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "BiasSoftmax") {
        auto& attrs = node.GetAttributes();
        TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
        TEST_RETURN_IF_NOT(attrs.find("is_inner_broadcast") != attrs.end());
        TEST_RETURN_IF_NOT(6 == static_cast<int>(attrs.at("axis").i()));
        TEST_RETURN_IF_NOT(static_cast<int>(attrs.at("is_inner_broadcast").i()) == 0);
      }
    }
    return Status::OK();
  };

  // Input and bias have different ranks.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3, 2, 3, 3, 3}});
      auto* bias_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
      auto* add_out = builder.MakeIntermediate();
      auto* softmax_out = builder.MakeOutput();

      builder.AddNode("Add", {input_arg, bias_arg}, {add_out});
      builder.AddNode("Softmax", {add_out}, {softmax_out}).AddAttribute("axis", static_cast<int64_t>(6));
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<BiasSoftmaxFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Input and bias have same rank.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3, 2, 3, 3, 3}});
      auto* bias_arg = builder.MakeInput<float>({{1, 1, 1, 1, 2, 3, 3, 3}});
      auto* add_out = builder.MakeIntermediate();
      auto* softmax_out = builder.MakeOutput();

      builder.AddNode("Add", {input_arg, bias_arg}, {add_out});
      builder.AddNode("Softmax", {add_out}, {softmax_out}).AddAttribute("axis", static_cast<int64_t>(6));
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<BiasSoftmaxFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, BiasSoftmaxFusionTest_OpSet13InValidAxis) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({{2, 3, 3, 3, 2, 3, 3, 3}});
    auto* bias_arg = builder.MakeInput<float>({{2, 3, 3, 3}});
    auto* add_out = builder.MakeIntermediate();
    auto* softmax_out = builder.MakeOutput();

    builder.AddNode("Add", {input_arg, bias_arg}, {add_out});
    builder.AddNode("Softmax", {add_out}, {softmax_out}).AddAttribute("axis", static_cast<int64_t>(6));
  };

  auto pre_graph_checker = [&](Graph& graph) {
    for (auto& node : graph.Nodes()) node.SetExecutionProviderType(kCudaExecutionProvider);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Softmax"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Softmax"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.BiasSoftmax"] == 0);
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<BiasSoftmaxFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                        pre_graph_checker, post_graph_checker));
}

static void TestBiasDropoutFusion(const PathString& file_path, const logging::Logger& logger, const int add_count = 0) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasDropoutFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Add"], add_count);
  ASSERT_EQ(op_to_count["Dropout"], 0);
  ASSERT_EQ(op_to_count["com.microsoft.BiasDropout"], 1);
}

TEST_F(GraphTransformationTests, BiasDropoutFusionTest) {
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_fusion1.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_fusion2.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion1.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion2.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_mismatch.onnx", *logger_, 1);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_multiple_consumers1.onnx", *logger_, 1);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_multiple_consumers2.onnx", *logger_, 1);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_same_shape_fusion.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_same_shape_fusion.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_fusion_dim_is_param.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_dim_is_param.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_same_shape_fusion_dim_is_param.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_same_shape_fusion_dim_is_param.onnx", *logger_);
}

#ifdef ENABLE_TRAINING
static void TestBitmaskDropoutFusion(const PathString& file_path, bool is_bias_dropout, const logging::Logger& logger,
                                     const int add_count, const int dropout_count, const int bitmask_dropout_count,
                                     const int bias_dropout_count, const int bitmask_bias_dropout_count,
                                     const int dropout_grad_count, const int bitmask_dropout_grad_count) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  if (is_bias_dropout) {
    ASSERT_STATUS_OK(
        graph_transformation_mgr.Register(std::make_unique<BiasDropoutFusion>(), TransformerLevel::Level2));
  } else {
    ASSERT_STATUS_OK(
        graph_transformation_mgr.Register(std::make_unique<BitmaskDropoutReplacement>(), TransformerLevel::Level2));
  }
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Add"], add_count);
  ASSERT_EQ(op_to_count["Dropout"], dropout_count);
  ASSERT_EQ(op_to_count["com.microsoft.BitmaskDropout"], bitmask_dropout_count);
  ASSERT_EQ(op_to_count["com.microsoft.BiasDropout"], bias_dropout_count);
  ASSERT_EQ(op_to_count["com.microsoft.BitmaskBiasDropout"], bitmask_bias_dropout_count);
  ASSERT_EQ(op_to_count["com.microsoft.DropoutGrad"], dropout_grad_count);
  ASSERT_EQ(op_to_count["com.microsoft.BitmaskDropoutGrad"], bitmask_dropout_grad_count);
}

TEST_F(GraphTransformationTests, BitmaskDropoutFusionTest) {
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_dropout_replacement_basic.onnx", false, *logger_, 0, 0, 1, 0, 0,
                           0, 1);
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_dropout_replacement_multiple_mask_uses.onnx", false, *logger_,
                           0, 1, 0, 0, 0, 1, 0);
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_bias_dropout_replacement_basic.onnx", false, *logger_, 0, 0, 0,
                           0, 1, 0, 1);
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_bias_dropout_fusion_basic.onnx", true, *logger_, 0, 0, 0, 0, 1,
                           0, 1);
  TestBitmaskDropoutFusion(MODEL_FOLDER "fusion/bitmask_bias_dropout_fusion_residual.onnx", true, *logger_, 0, 0, 0, 0,
                           1, 0, 1);
}

/*
This test build a graph like:
             input0  input1
                  \ /
                  Add
 -----------------|
 |                |
 |             Shape
 |             /   \
 |        Gather0  Gather1
 |           /       \
 |     Unsqueeze0  Unsqueeze1  (Constant Initializer) (Constant Initializer)
 |          \        /               /                  /
 |     	     \      /               /                  /
 |         ConcatTraining   -------       ------------
  \	       /
   \	      /
     Reshape


After fusion, the graph become:
             input0  input1
                  \ /
                  Add     (Constant Initializer)
                   \       /
                    Reshape

*/
TEST_F(GraphTransformationTests, ReshapeFusionOpsetTest) {
  constexpr const int batch_size = 64;
  constexpr const int seq_lenth = 1024;
  constexpr const int hidden_size = 1024;

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.ConcatTraining"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Shape"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Unsqueeze"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.ConcatTraining"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Reshape"] == 1);
    return Status::OK();
  };

  const std::vector<int> opsets{11, 12, 13, 14, 15, 18};
  bool shape_test_for_opset15 = false;

  for (auto& opset : opsets) {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto opset_version = builder.DomainToVersionMap().find(kOnnxDomain)->second;
      auto* input_arg0 = builder.MakeInput<float>({{batch_size, seq_lenth, hidden_size}});
      auto* input_arg1 = builder.MakeInput<float>({{hidden_size}});
      auto* scalar_int_0 = builder.MakeInitializer<int64_t>({}, {0});
      auto* scalar_int_1 = builder.MakeInitializer<int64_t>({}, {1});
      auto* single_value_1d_int_0 = builder.MakeInitializer<int64_t>({1}, {0});
      auto* single_value_1d_int_16 = builder.MakeInitializer<int64_t>({1}, {16});
      auto* single_value_1d_int_64 = builder.MakeInitializer<int64_t>({1}, {64});
      auto* add_out = builder.MakeIntermediate();
      auto* shape_out = builder.MakeIntermediate();
      auto* gather_out_0 = builder.MakeIntermediate();
      auto* gather_out_1 = builder.MakeIntermediate();
      auto* unsqueeze_out_0 = builder.MakeIntermediate();

      auto* unsqueeze_out_1 = builder.MakeIntermediate();
      auto* concattraining1_out = builder.MakeIntermediate();
      auto* concattraining1_length = builder.MakeIntermediate();
      auto* out = builder.MakeOutput();

      builder.AddNode("Add", {input_arg0, input_arg1}, {add_out});
      if (opset_version >= 15) {
        if (shape_test_for_opset15) {
          auto& shape_1 = builder.AddNode("Shape", {add_out}, {shape_out});
          shape_1.AddAttribute("start", (int64_t)1);
          shape_1.AddAttribute("end", (int64_t)2);
        } else {
          builder.AddNode("Shape", {add_out}, {shape_out}).AddAttribute("start", (int64_t)0);
          shape_test_for_opset15 = true;
        }
      } else {
        builder.AddNode("Shape", {add_out}, {shape_out});
      }
      builder.AddNode("Gather", {shape_out, scalar_int_0}, {gather_out_0});
      builder.AddNode("Gather", {shape_out, scalar_int_1}, {gather_out_1});
      if (opset_version >= 13) {
        builder.AddNode("Unsqueeze", {gather_out_0, single_value_1d_int_0}, {unsqueeze_out_0});
        builder.AddNode("Unsqueeze", {gather_out_1, single_value_1d_int_0}, {unsqueeze_out_1});
      } else {
        builder.AddNode("Unsqueeze", {gather_out_0}, {unsqueeze_out_0}).AddAttribute("axes", std::vector<int64_t>{0});
        builder.AddNode("Unsqueeze", {gather_out_1}, {unsqueeze_out_1}).AddAttribute("axes", std::vector<int64_t>{0});
      }
      builder.AddNode("ConcatTraining", {unsqueeze_out_0, unsqueeze_out_1, single_value_1d_int_16, single_value_1d_int_64},
                      {concattraining1_out, concattraining1_length}, "com.microsoft")
          .AddAttribute("axis", static_cast<int64_t>(0));
      builder.AddNode("Reshape", {add_out, concattraining1_out}, {out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ReshapeFusion>();
    if (opset >= 15 && shape_test_for_opset15) {
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                            pre_graph_checker, pre_graph_checker));
    } else {
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                            pre_graph_checker, post_graph_checker));
    }
  }
}
#endif

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_Bias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_ND_bias) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias_ND.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_Bias_No_B_ZP) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias_b_no_zp.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, MatMulIntegerToFloatTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/matmul_integer_to_float.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MatMulIntegerToFloatFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 1);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.MatMulIntegerToFloat"], 3);
  EXPECT_EQ(op_to_count["Add"], 1);
}

#endif

#ifndef DISABLE_CONTRIB_OPS
template <typename GraphTransformationCheckFn, typename GraphPreprocessFn>
static void TestMatMulScaleFusion(
    const PathString& model_path, const Logger& logger,
    GraphPreprocessFn graph_preprocess_fn,
    GraphTransformationCheckFn graph_transformation_check_fn,
    const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
    const InlinedHashSet<std::string>& excluded_initializer_names = {}) {
  SCOPED_TRACE(ORT_TSTR("model path: ") + model_path);

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_path, model, nullptr, logger));
  Graph& graph = model->MainGraph();

  graph_preprocess_fn(graph);

  auto original_op_counts = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformer_manager{5};
  ASSERT_STATUS_OK(graph_transformer_manager.Register(
      make_unique<MatMulScaleFusion>(compatible_execution_providers, excluded_initializer_names),
      TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformer_manager.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  auto transformed_op_counts = CountOpsInGraph(graph);

  graph_transformation_check_fn(graph, original_op_counts, transformed_op_counts);
}

template <typename GraphTransformationCheckFn>
static void TestMatMulScaleFusion(
    const PathString& model_path, const Logger& logger,
    GraphTransformationCheckFn graph_transformation_check,
    const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
    const InlinedHashSet<std::string>& excluded_initializer_names = {}) {
  TestMatMulScaleFusion(
      model_path, logger,
      [](Graph&) {}, graph_transformation_check,
      compatible_execution_providers, excluded_initializer_names);
}

TEST_F(GraphTransformationTests, MatMulScaleFusionFusableModels) {
  const std::vector<PathString> one_fusion_model_paths{
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx",
      MODEL_FOLDER "fusion/matmul_scale_in0_in1.onnx",
      MODEL_FOLDER "fusion/matmul_scale_in0_in1_out.onnx",
      MODEL_FOLDER "fusion/matmul_scale_transposescalematmul_in0_in1_out.onnx",
  };

  for (const auto& path : one_fusion_model_paths) {
    TestMatMulScaleFusion(
        path, *logger_,
        [](const Graph& graph,
           std::map<std::string, int> original_op_counts,
           std::map<std::string, int> transformed_op_counts) {
          EXPECT_EQ(transformed_op_counts["Mul"], 0);
          EXPECT_EQ(transformed_op_counts["Div"], 0);
          EXPECT_EQ(transformed_op_counts["MatMul"], 0);
          EXPECT_EQ(transformed_op_counts["com.microsoft.FusedMatMul"], 1);

          // check combined scale, individual scales should all have the same value
          constexpr float scale_value = 3.0f;

          const int num_scales =
              original_op_counts["Mul"] + original_op_counts["Div"] + original_op_counts["com.microsoft.FusedMatMul"];

          auto fused_node = std::find_if(
              graph.Nodes().cbegin(), graph.Nodes().cend(),
              [](const Node& node) { return node.OpType() == "FusedMatMul"; });
          ASSERT_NE(fused_node, graph.Nodes().cend());

          auto alpha_attr = fused_node->GetAttributes().find("alpha");
          ASSERT_NE(alpha_attr, fused_node->GetAttributes().end());

          EXPECT_EQ(alpha_attr->second.f(), pow(scale_value, num_scales));
        });
  }
}

TEST_F(GraphTransformationTests, MatMulScaleFusionUnfusableModels) {
  const std::vector<PathString> unfusable_model_paths{
      MODEL_FOLDER "fusion/matmul_scale_unfusable_div_not_scale.onnx",
      MODEL_FOLDER "fusion/matmul_scale_unfusable_scale_not_scalar.onnx",
      MODEL_FOLDER "fusion/matmul_scale_unfusable_scale_not_constant.onnx",
  };

  for (const auto& path : unfusable_model_paths) {
    TestMatMulScaleFusion(
        path, *logger_,
        [](const Graph&,
           const std::map<std::string, int>& original_op_counts,
           const std::map<std::string, int>& transformed_op_counts) {
          EXPECT_EQ(original_op_counts, transformed_op_counts);
        });
  }
}

TEST_F(GraphTransformationTests, MatMulScaleFusionReusedInputScale) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_reused_input_scale.onnx", *logger_,
      [](const Graph&,
         const std::map<std::string, int>&,
         std::map<std::string, int> transformed_op_counts) {
        EXPECT_EQ(transformed_op_counts["Mul"], 0);
        EXPECT_EQ(transformed_op_counts["Div"], 0);
        EXPECT_EQ(transformed_op_counts["MatMul"], 0);
        EXPECT_EQ(transformed_op_counts["com.microsoft.FusedMatMul"], 2);
      });
}

TEST_F(GraphTransformationTests, MatMulScaleFusionExcludedInitializerName) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx", *logger_,
      [](const Graph&,
         const std::map<std::string, int>& original_op_counts,
         const std::map<std::string, int>& transformed_op_counts) {
        EXPECT_EQ(original_op_counts, transformed_op_counts);
      },
      {},
      {"scale"});
}

TEST_F(GraphTransformationTests, MatMulScaleFusionIncompatibleExecutionProvider) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx", *logger_,
      [](Graph& graph) {
        for (auto& node : graph.Nodes()) {
          node.SetExecutionProviderType(kCudaExecutionProvider);
        }
      },
      [](const Graph&,
         const std::map<std::string, int>& original_op_counts,
         const std::map<std::string, int>& transformed_op_counts) {
        EXPECT_EQ(original_op_counts, transformed_op_counts);
      },
      {kCpuExecutionProvider});
}

TEST_F(GraphTransformationTests, MatMulScaleFusionUnsupportedInputType) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_int32.onnx", *logger_,
      [](Graph& graph) {
        for (auto& node : graph.Nodes()) {
          node.SetExecutionProviderType(kCpuExecutionProvider);
        }
      },
      [](const Graph&,
         const std::map<std::string, int>& original_op_counts,
         const std::map<std::string, int>& transformed_op_counts) {
        EXPECT_EQ(original_op_counts, transformed_op_counts);
      },
      {kCpuExecutionProvider});
}

TEST_F(GraphTransformationTests, MatMulScaleFusionWithScaleInput) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_with_scale_input.onnx", *logger_,
      [](const Graph&,
         const std::map<std::string, int>&,
         std::map<std::string, int> transformed_op_counts) {
        EXPECT_EQ(transformed_op_counts["Mul"], 1);
        EXPECT_EQ(transformed_op_counts["MatMul"], 1);
        EXPECT_EQ(transformed_op_counts["com.microsoft.FusedMatMul"], 0);
      });
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST_F(GraphTransformationTests, IsInfReduceSum_Test) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/isinf_reducesum.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<IsInfReduceSumFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["IsInf"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.IsAllFinite"], 1);
  EXPECT_EQ(op_to_count["Not"], 1);
}
#endif
#endif

TEST_F(GraphTransformationTests, FilterEnabledOptimizers) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/constant_folding_with_scalar_shape_to_initializer.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.FilterEnabledOptimizers";
  InferenceSessionWrapper session_object{so, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.Load(model_uri));

  const auto& graph = session_object.GetGraph();

  // check the ops that should go away if the constant folding transformer runs
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);

  ASSERT_STATUS_OK(session_object.FilterEnabledOptimizers({"ConstantFolding"}));
  ASSERT_STATUS_OK(session_object.Initialize());  // Initialize runs the transformers

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

TEST_F(GraphTransformationTests, PropagateCastOpsTests) {
  using Strategy = GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy;
  struct PropagateCastOpsTestSpecs {
    PathString model_uri;
    // Expected number of casts after the transformation with different stratigies and optimization levels
    std::map<std::pair<Strategy, int>, int> casts_count_map;
    vector<std::string> allow_ops = {};  // Allowed ops for PropagateCastOps graph transformer
  };

  std::pair<Strategy, int> insertAndReduce0 = std::make_pair(Strategy::InsertAndReduce, 0);
  std::pair<Strategy, int> insertAndReduce1 = std::make_pair(Strategy::InsertAndReduce, 1);
  std::pair<Strategy, int> floodFill1 = std::make_pair(Strategy::FloodFill, 1);
  std::pair<Strategy, int> floodFill2 = std::make_pair(Strategy::FloodFill, 2);
  std::vector<std::string> allow_matmul = {"MatMul"};
  std::vector<std::string> allow_matmul_transpose = {"MatMul", "Transpose"};
  std::vector<std::string> allow_matmul_transpose_add = {"Add", "MatMul", "Transpose"};
  const std::vector<PropagateCastOpsTestSpecs> test_cases = {
      {MODEL_FOLDER "propagate_cast/squeeze_cast_propagation_test.onnx", {{insertAndReduce0, 2}, {insertAndReduce1, 0}, {floodFill1, 0}, {floodFill2, 0}}},
      {MODEL_FOLDER "propagate_cast/unsqueeze_cast_propagation_test.onnx", {{insertAndReduce0, 2}, {insertAndReduce1, 0}, {floodFill1, 0}, {floodFill2, 0}}},
      // Negative testcase to test that the transformer will not move cast bool to float/float16.
      {MODEL_FOLDER "propagate_cast/negative_test_case_bool_fp_cast.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, {"Add"}},
      {MODEL_FOLDER "propagate_cast/negative_test_case_bool_fp16_cast.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, {"Add"}},
      // Test fusing back to back casts functionality
      {MODEL_FOLDER "propagate_cast/fuse_back2back_casts_float16_float16.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      {MODEL_FOLDER "propagate_cast/fuse_back2back_casts_float16_float.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}},
      {MODEL_FOLDER "propagate_cast/fuse_back2back_casts_float_float16.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}},
      {MODEL_FOLDER "propagate_cast/fuse_back2back_casts_float_float.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      // Test fusing subgraph functionality
      {MODEL_FOLDER "propagate_cast/fuse_sibling_casts_float16_float16.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      {MODEL_FOLDER "propagate_cast/fuse_sibling_casts_float16_float.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      {MODEL_FOLDER "propagate_cast/fuse_sibling_casts_float_float16.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},
      {MODEL_FOLDER "propagate_cast/fuse_sibling_casts_float_float.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}},

      // Test constant propagation with various combinations
      // 1. Computation is float or float16
      // 2. The inputs and/or output may be casted
      // 3. The inputs and/or output may be transposed
      // These variations help testing the following functions.
      // PropagateForward, PropagateBackward, PropagateFP16FromInputsToOutput, and PropagateFP32FromOutputsToInputs

      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_product_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_product_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_transpose_product_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_transpose_product_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_transpose_inputs_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 2}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_product.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_sum.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_product_cast_sum.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 4}, {floodFill1, 4}, {floodFill2, 4}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_cast_product_cast_input2.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 4}, {floodFill1, 4}, {floodFill2, 4}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 4}, {floodFill1, 4}, {floodFill2, 4}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_inputs_transpose_product_cast_product_cast_input2.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_input2_cast_sum.onnx", {{insertAndReduce0, 0}, {floodFill1, 0}, {floodFill2, 0}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 2}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_inputs_cast_product_cast_input2.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_product_cast_input2_cast_sum.onnx", {{insertAndReduce0, 4}, {floodFill1, 4}, {floodFill2, 4}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_add_transpose_product_cast_product_cast_input2.onnx", {{insertAndReduce0, 3}, {floodFill1, 3}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_transpose.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_transpose.onnx", {{insertAndReduce0, 3}, {floodFill1, 1}, {floodFill2, 3}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_transpose_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_transpose_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_second_matmul.onnx", {{insertAndReduce0, 3}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 2}, {floodFill2, 4}}, allow_matmul},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 2}, {floodFill2, 4}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_transpose_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 1}, {floodFill2, 4}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_after_cast_transpose_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 1}, {floodFill2, 4}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_transpose_before_cast_second_matmul_add_products.onnx", {{insertAndReduce0, 5}, {floodFill1, 2}, {floodFill2, 4}}, allow_matmul_transpose},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs.onnx", {{insertAndReduce0, 1}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 4}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast.onnx", {{insertAndReduce0, 1}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 4}, {floodFill2, 3}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_transpose.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_transpose_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 3}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_after_cast_transpose_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast.onnx", {{insertAndReduce0, 1}, {floodFill1, 2}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 4}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_transpose.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_transpose_second_matmul_add_products.onnx", {{insertAndReduce0, 2}, {floodFill1, 3}, {floodFill2, 2}}, allow_matmul_transpose_add},
      {MODEL_FOLDER "propagate_cast/matmul_two_outputs_cast_inputs_transpose_before_cast_transpose_second_matmul.onnx", {{insertAndReduce0, 1}, {floodFill1, 1}, {floodFill2, 1}}, allow_matmul_transpose_add}};

  // Create a temporary directory, which will be deleted automatically, to save/load the transformed models.
  TemporaryDirectory temp_dir{ORT_TSTR("propagate_casts_test_output_dir")};
  for (PropagateCastOpsTestSpecs test_case : test_cases) {
    for (const auto& scenario : test_case.casts_count_map) {
      Strategy strategy = scenario.first.first;
      int level = scenario.first.second;
      int expected_casts_count = scenario.second;
      std::shared_ptr<Model> p_model;
      ASSERT_STATUS_OK(Model::Load(test_case.model_uri, p_model, nullptr, *logger_));
      Graph& graph = p_model->MainGraph();
      ASSERT_STATUS_OK(graph.Resolve());
      onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
      ASSERT_STATUS_OK(graph_transformation_mgr.Register(
          std::make_unique<PropagateCastOps>(strategy, level, test_case.allow_ops),
          TransformerLevel::Level1));
      ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
      Path p = Path::Parse(test_case.model_uri);
      ASSERT_FALSE(p.GetComponents().empty());
      PathString transformed_model_uri = temp_dir.Path() + GetPathSep<PathChar>() + ORT_TSTR("transformed_") + p.GetComponents().back();
      ASSERT_STATUS_OK(Model::Save(*p_model, transformed_model_uri));
      // Load the transformed model to validate
      ASSERT_STATUS_OK(Model::Load(transformed_model_uri, p_model, nullptr, *logger_));
      Graph& transformed_graph = p_model->MainGraph();
      ASSERT_STATUS_OK(transformed_graph.Resolve());
      std::map<std::string, int> op_to_count = CountOpsInGraph(transformed_graph);
      ASSERT_EQ(op_to_count["Cast"], expected_casts_count);
    }
  }
}

#ifdef ENABLE_TRAINING_CORE
TEST_F(GraphTransformationTests, PropagateCastOpsTests_Gelu) {
  using Strategy = GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy;
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* cast_out_0 = builder.MakeIntermediate();
      auto* gelu_out = builder.MakeIntermediate();
      auto* cast_out_1 = builder.MakeIntermediate();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out_0})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Gelu", {cast_out_0}, {gelu_out}, kMSDomain);
      builder.AddNode("Cast", {gelu_out}, {cast_out_1})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      builder.AddNode("Identity", {cast_out_1}, {identity_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<PropagateCastOps>(Strategy::FloodFill, 1);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<BFloat16>({{2, -1, 3, -1}});
      auto* cast_out_0 = builder.MakeIntermediate();
      auto* gelu_out = builder.MakeIntermediate();
      auto* cast_out_1 = builder.MakeIntermediate();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out_0})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Gelu", {cast_out_0}, {gelu_out}, kMSDomain);
      builder.AddNode("Cast", {gelu_out}, {cast_out_1})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16));
      builder.AddNode("Identity", {cast_out_1}, {identity_out});
    };

    auto pre_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<PropagateCastOps>(Strategy::FloodFill, 1);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, PropagateCastOpsTests_Softmax) {
  using Strategy = GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy;
  {
    auto build_test_case = [](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<MLFloat16>({{2, 3, 3, 3}});
      auto* cast_out_0 = builder.MakeIntermediate();
      auto* softmax_out = builder.MakeIntermediate();
      auto* cast_out_1 = builder.MakeIntermediate();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out_0})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Softmax", {cast_out_0}, {softmax_out});
      builder.AddNode("Cast", {softmax_out}, {cast_out_1})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
      builder.AddNode("Identity", {cast_out_1}, {identity_out});
    };

    auto pre_graph_checker = [](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    auto post_graph_checker = [](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 0);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<PropagateCastOps>(Strategy::FloodFill, 1);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  {
    auto build_test_case = [](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<BFloat16>({{2, -1, 3, -1}});
      auto* cast_out_0 = builder.MakeIntermediate();
      auto* softmax_out = builder.MakeIntermediate();
      auto* cast_out_1 = builder.MakeIntermediate();
      auto* identity_out = builder.MakeOutput();

      builder.AddNode("Cast", {input_arg}, {cast_out_0})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
      builder.AddNode("Softmax", {cast_out_0}, {softmax_out});
      builder.AddNode("Cast", {softmax_out}, {cast_out_1})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16));
      builder.AddNode("Identity", {cast_out_1}, {identity_out});
    };

    auto pre_graph_checker = [](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    auto post_graph_checker = [](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<PropagateCastOps>(Strategy::FloodFill, 1);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

#endif

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 256] (int64_t)
                 |
                Neg
            /    |  \______________________________________________________
           /     |  256 (int64_t)                                       Cast
          / ...  | /                                    _________/     /   \      \_____________
        Add      Add                               ____/    __________/     \________           \
         |       |  0 (int64_t)  511 (int64_t)    /        /                         \          |
         |       |  __/    _____/                /        | 128 (int32_t)            |  64 [1] |
        Clip ... Clip                           /         | /                        |  /       |
         |        |                            Sub   ...  Sub                        Mul  ...  Mul
                                                |         |                          |          |
graph out [1, 1, 256, 256] (int64_t)        graph out [1, 1, 256, 256] (int32_t)   graph out [1, 1, 256, 256] (int32_t)

Be noted:
 the Add's input initializer 256 is a scalar int64_t;
 the Sub's input initializer 128 is a scalar int32_t;
 the Mul's input initializer 64 is a 1-D int32_t.
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShareIntTypedInitializer) {
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 6U);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Clip"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Sub"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Neg"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Cast"] == 1);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 15U);
    return Status::OK();
  };

  std::vector<int64_t> adders{256, 512};
  std::vector<int32_t> subers{128, 512};
  std::vector<int32_t> mulers{64, 512};
  for (size_t test_data_index = 0; test_data_index < adders.size(); ++test_data_index) {
    int64_t adder = adders[test_data_index];
    int32_t suber = subers[test_data_index];
    int32_t muler = mulers[test_data_index];
    auto post_graph_checker = [adder, suber, muler](Graph& graph) {
      const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
      TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 5U);
      const NodeArg* add_initializer = nullptr;
      const NodeArg* clip_min_initializer = nullptr;
      const NodeArg* clip_max_initializer = nullptr;
      const NodeArg* sub_initializer = nullptr;
      const NodeArg* mul_initializer = nullptr;

      for (auto& node : graph.Nodes()) {
        if (node.OpType().compare("Add") == 0) {
          if (!add_initializer) {
            add_initializer = node.InputDefs()[1];
            TEST_RETURN_IF_NOT(add_initializer != nullptr);
            const TensorShapeProto* s = add_initializer->Shape();
            TEST_RETURN_IF_NOT(s->dim_size() == 0);
          } else {
            TEST_RETURN_IF_NOT(add_initializer == node.InputDefs()[1]);
            CheckShapeEquality(add_initializer->Shape(), node.InputDefs()[1]->Shape());
          }
        } else if (node.OpType().compare("Clip") == 0) {
          if (!clip_min_initializer && !clip_max_initializer) {
            clip_min_initializer = node.InputDefs()[1];
            clip_max_initializer = node.InputDefs()[2];
            TEST_RETURN_IF(clip_min_initializer == nullptr);
            TEST_RETURN_IF(clip_max_initializer == nullptr);
            const TensorShapeProto* s1 = clip_min_initializer->Shape();
            const TensorShapeProto* s2 = clip_max_initializer->Shape();
            TEST_RETURN_IF_NOT(s1->dim_size() == 0);
            TEST_RETURN_IF_NOT(s2->dim_size() == 0);
          } else {
            TEST_RETURN_IF_NOT(clip_min_initializer == node.InputDefs()[1]);
            TEST_RETURN_IF_NOT(clip_max_initializer == node.InputDefs()[2]);
            CheckShapeEquality(clip_min_initializer->Shape(), node.InputDefs()[1]->Shape());
            CheckShapeEquality(clip_max_initializer->Shape(), node.InputDefs()[2]->Shape());
          }
        } else if (node.OpType().compare("Sub") == 0) {
          if (!sub_initializer) {
            sub_initializer = node.InputDefs()[1];
            TEST_RETURN_IF(sub_initializer == nullptr);
            TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim_size() == 0);
          } else {
            TEST_RETURN_IF_NOT(sub_initializer == node.InputDefs()[1]);
            CheckShapeEquality(sub_initializer->Shape(), node.InputDefs()[1]->Shape());
          }
        } else if (node.OpType().compare("Mul") == 0) {
          if (!mul_initializer) {
            mul_initializer = node.InputDefs()[1];
            TEST_RETURN_IF(mul_initializer == nullptr);
            const TensorShapeProto* s = mul_initializer->Shape();
            TEST_RETURN_IF_NOT(s->dim_size() == 1);
            auto dim1 = s->dim(0);
            TEST_RETURN_IF_NOT(s->dim(0).has_dim_value());
            TEST_RETURN_IF_NOT(s->dim(0).dim_value() == 1);
          } else {
            TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
            CheckShapeEquality(mul_initializer->Shape(), node.InputDefs()[1]->Shape());
          }
        }
      }

      for (const auto& entry : initialized_tensor_set) {
        InlinedVector<int64_t> values;
        constexpr bool require_constant = true;
        NodeArg* initializer_node_arg = graph.GetNodeArg(entry.first);
        TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));

        if (add_initializer != nullptr && entry.first.compare(add_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == adder);
        } else if (clip_min_initializer != nullptr && entry.first.compare(clip_min_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == 0);
        } else if (clip_max_initializer != nullptr && entry.first.compare(clip_max_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == 511);
        } else if (sub_initializer != nullptr && entry.first.compare(sub_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == suber);
        } else if (mul_initializer != nullptr && entry.first.compare(mul_initializer->Name()) == 0) {
          TEST_RETURN_IF_NOT(values.size() == 1U);
          TEST_RETURN_IF_NOT(values[0] == muler);
        }
      }

      auto op_count = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count.size() == 6U);
      TEST_RETURN_IF_NOT(op_count["Add"] == 3);
      TEST_RETURN_IF_NOT(op_count["Clip"] == 3);
      TEST_RETURN_IF_NOT(op_count["Sub"] == 3);
      TEST_RETURN_IF_NOT(op_count["Mul"] == 3);
      TEST_RETURN_IF_NOT(op_count["Neg"] == 1);
      TEST_RETURN_IF_NOT(op_count["Cast"] == 1);
      return Status::OK();
    };

    auto build_test_case = [adder, suber, muler](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<int64_t>({{1, 1, 256, 256}});
      auto* neg_out = builder.MakeIntermediate();
      builder.AddNode("Neg", {input_arg}, {neg_out});

      // test scalar int64_t values.
      for (size_t i = 0; i < 3; ++i) {
        auto* add_initializer = builder.MakeScalarInitializer<int64_t>(adder);
        auto* add_out = builder.MakeIntermediate();
        auto* clip_out = builder.MakeOutput();
        auto* clip_min_initializer = builder.MakeScalarInitializer<int64_t>(0);
        auto* clip_max_initializer = builder.MakeScalarInitializer<int64_t>(511);
        builder.AddNode("Add", {neg_out, add_initializer}, {add_out});
        builder.AddNode("Clip", {add_out, clip_min_initializer, clip_max_initializer}, {clip_out});
      }
      auto* cast_out = builder.MakeIntermediate();
      builder.AddNode("Cast", {neg_out}, {cast_out})
          .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32));

      // test scalar int32_t values.
      for (size_t i = 0; i < 3; ++i) {
        auto* sub_initializer = builder.MakeScalarInitializer<int32_t>(suber);
        auto* sub_out = builder.MakeOutput();
        builder.AddNode("Sub", {cast_out, sub_initializer}, {sub_out});
      }

      // test 1-D int32_t values.
      for (size_t i = 0; i < 3; ++i) {
        auto* mul_initializer = builder.MakeInitializer<int32_t>({1}, {muler});
        auto* mul_out = builder.MakeOutput();
        builder.AddNode("Mul", {cast_out, mul_initializer}, {mul_out});
      }
    };

    const std::vector<int> opsets{12, 13, 14};  // Clip support int64_t since opset 12
    for (auto& opset_version : opsets) {
      std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger_, std::move(transformer), TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

template <typename T>
void BuildConstantSharingDivMulGraph(ModelTestBuilder& builder) {
  auto* input0_arg = builder.MakeInput<T>({{1, 1, 256, 256}});
  auto* input1_arg = builder.MakeInput<T>({{1, 1, 256, 256}});
  auto* div_out = builder.MakeIntermediate();
  builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

  for (size_t i = 0; i < 12; ++i) {
    NodeArg* mul_initializer = nullptr;
    if (std::is_same<T, MLFloat16>::value) {
      mul_initializer = builder.MakeScalarInitializer<MLFloat16>(MLFloat16(1.0f));
    } else if (std::is_same<T, float>::value) {
      mul_initializer = builder.MakeScalarInitializer<float>(1.0f);
    } else {
      ASSERT_TRUE(false);
    }
    auto* mul_out = builder.MakeOutput();
    builder.AddNode("Mul", {div_out, mul_initializer}, {mul_out});
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 256] (float|MLFloat16)
                 |
                Div
            /    |       \
           /     |  1.0   \
          / ...  |  / ...  \
        Mul      Mul      Mul
         |       |         |
 graph out [1, 1, 256, 256] (float|MLFloat16)

Be noted:
 the Mul's input initializer 1.0f is a scalar float/MLFloat16.
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShareFloatOrHalfTypedInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 2U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 12);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 12U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 1U);
    const NodeArg* mul_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        int32_t data_type = tensor_proto->data_type();
        onnxruntime::Initializer float_const{*tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(float_const.size() == 1U);
        float float_const_value;
        if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
          float_const_value = math::halfToFloat(float_const.data<MLFloat16>()->val);
        } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          float_const_value = *(float_const.data<float>());
        } else {
          return Status(common::ONNXRUNTIME, common::FAIL, "unexpected type");
        }

        TEST_RETURN_IF_NOT(float_const_value == 1.0f);
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 2U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 12);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14};  // Clip support int64_t since opset 12

  // Float data type tests.
  auto build_test_case_float = [&](ModelTestBuilder& builder) {
    BuildConstantSharingDivMulGraph<float>(builder);
  };
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // MLFloat16 data type tests.
  auto build_test_case_mlfloat16 = [&](ModelTestBuilder& builder) {
    BuildConstantSharingDivMulGraph<MLFloat16>(builder);
  };

  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_mlfloat16, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

template <typename T>
void BuildConstantSharingDivMulGraphFor2DInitializer(ModelTestBuilder& builder) {
  auto* input0_arg = builder.MakeInput<T>({{1, 1, 256, 8}});
  auto* input1_arg = builder.MakeInput<T>({{1, 1, 256, 8}});
  auto* div_out = builder.MakeIntermediate();
  builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

  std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<MLFloat16> values_float16;
  values_float16.reserve(values.size());
  if (std::is_same<T, MLFloat16>::value) {
    for (auto v : values) {
      values_float16.push_back(MLFloat16(v));
    }
  }

  for (size_t i = 0; i < 12; ++i) {
    NodeArg* mul_initializer = nullptr;
    if (std::is_same<T, MLFloat16>::value) {
      mul_initializer = builder.MakeInitializer<MLFloat16>({1, 8}, values_float16);
    } else if (std::is_same<T, float>::value) {
      mul_initializer = builder.MakeInitializer<float>({1, 8}, values);
    } else {
      ASSERT_TRUE(false);
    }
    auto* mul_out = builder.MakeOutput();
    builder.AddNode("Mul", {div_out, mul_initializer}, {mul_out});
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 8] (float|MLFloat16)
                 |
                Div
            /    |       \
           /     |        \
          / ...  |  / ...  \
        Mul      Mul      Mul
         |       |         |
 graph out [1, 1, 256, 8] (float|MLFloat16)

Be noted:
 the Mul's input initializer is a 2D float/MLFloat16.
*/
TEST_F(GraphTransformationTests, ConstantSharing_Share2DFloatOrHalfTypedInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 2U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 12);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 12U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 1U);
    const NodeArg* mul_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 2);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        int32_t data_type = tensor_proto->data_type();
        onnxruntime::Initializer float_const{*tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(float_const.size() == 8U);
        for (int i = 0; i < 8; ++i) {
          float float_const_value;
          if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
            float_const_value = math::halfToFloat((float_const.data<MLFloat16>() + i)->val);
          } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
            float_const_value = *(float_const.data<float>() + i);
          } else {
            return Status(common::ONNXRUNTIME, common::FAIL, "unexpected type");
          }
          TEST_RETURN_IF_NOT(float_const_value == i * 1.0f);
        }
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 2U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 12);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14};  // Clip support int64_t since opset 12

  // Float data type tests.
  auto build_test_case_float = [&](ModelTestBuilder& builder) {
    BuildConstantSharingDivMulGraphFor2DInitializer<float>(builder);
  };
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // MLFloat16 data type tests.
  auto build_test_case_mlfloat16 = [&](ModelTestBuilder& builder) {
    BuildConstantSharingDivMulGraphFor2DInitializer<MLFloat16>(builder);
  };

  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_mlfloat16, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 256] (float)
                 |
                Div ______________________________
            /    |                 |              |
           /     |  1.0float       |  1.0half     |  1.0half
          / ...  |  / ...          |  /   ...     |  /   ...
        Mul      Mul              Add            Add
         |       |                     \          /
 graph out [1, 1, 256, 256](float)   graph out [1, 1, 256, 256](MLFloat16)

Be noted:
 the Mul's input initializer 1.0f is a scalar float.
 the Add's input initializer 1.0f is a scalar MLFloat16.
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShareFloatAndHalfTypedInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 4U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 3);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 6U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 2U);
    const NodeArg* mul_initializer = nullptr;
    const NodeArg* add_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      } else if (node.OpType().compare("Add") == 0) {
        if (!add_initializer) {
          add_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(add_initializer == nullptr);
          TEST_RETURN_IF_NOT(add_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(add_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    TEST_RETURN_IF(add_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
      int32_t data_type = tensor_proto->data_type();
      onnxruntime::Initializer float_const{*tensor_proto, graph.ModelPath()};
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(float_const.size() == 1U);
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        float float_const_value = *(float_const.data<float>());
        TEST_RETURN_IF_NOT(float_const_value == 1.0f);
      } else if (entry.first.compare(add_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(float_const.size() == 1U);
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
        float float_const_value = math::halfToFloat(float_const.data<MLFloat16>()->val);
        TEST_RETURN_IF_NOT(float_const_value == 1.0f);
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 4U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count["Add"] == 3);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14};

  auto build_test_case_float = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* input1_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* div_out = builder.MakeIntermediate();
    builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

    for (size_t i = 0; i < 3; ++i) {
      NodeArg* mul_initializer = builder.MakeScalarInitializer<float>(1.0f);

      auto* mul_out = builder.MakeOutput();
      builder.AddNode("Mul", {div_out, mul_initializer}, {mul_out});
    }

    auto* cast_out = builder.MakeIntermediate();
    builder.AddNode("Cast", {div_out}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
    for (size_t i = 0; i < 3; ++i) {
      NodeArg* add_initializer = builder.MakeScalarInitializer<MLFloat16>(MLFloat16(1.0f));
      auto* add_out = builder.MakeOutput();
      builder.AddNode("Add", {cast_out, add_initializer}, {add_out});
    }
  };
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 8, 8] (float)
                 |
                Div ______________________________
            /    |    \_______     |              |
           /     |  float  |  |    |  half        |  half
          / ...  |  / ...  |  |    |  /   ...     |  /   ...
        Mul      Mul      Sub Sub Add            Add
         |       |         |  |        \          /
 graph out [1, 1, 8, 8](float)   graph out [1, 1, 8, 8](MLFloat16)

Be noted:
 the Mul's input initializer is a 2D float tensor.
 the Add's input initializer is a 2D MLFloat16 tensor.
*/
TEST_F(GraphTransformationTests, ConstantSharing_Share2DFloatAndHalfTypedInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 5U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Sub"] == 3);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 3);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 9U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 3U);
    const NodeArg* mul_initializer = nullptr;
    const NodeArg* sub_initializer = nullptr;
    const NodeArg* add_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 2);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim(0).dim_value() == 1);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim(1).dim_value() == 8);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      } else if (node.OpType().compare("Sub") == 0) {
        if (!sub_initializer) {
          sub_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(sub_initializer == nullptr);
          TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim_size() == 2);
          TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim(0).dim_value() == 8);
          TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim(1).dim_value() == 1);
        } else {
          TEST_RETURN_IF_NOT(sub_initializer == node.InputDefs()[1]);
        }
      } else if (node.OpType().compare("Add") == 0) {
        if (!add_initializer) {
          add_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(add_initializer == nullptr);
          TEST_RETURN_IF_NOT(add_initializer->Shape()->dim_size() == 2);
          TEST_RETURN_IF_NOT(add_initializer->Shape()->dim(0).dim_value() == 1);
          TEST_RETURN_IF_NOT(add_initializer->Shape()->dim(1).dim_value() == 8);
        } else {
          TEST_RETURN_IF_NOT(add_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    TEST_RETURN_IF(sub_initializer == nullptr);
    TEST_RETURN_IF(add_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
      int32_t data_type = tensor_proto->data_type();
      onnxruntime::Initializer float_const{*tensor_proto, graph.ModelPath()};
      TEST_RETURN_IF_NOT(float_const.size() == 8U);
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        for (int i = 0; i < 8; ++i) {
          float float_const_value = *(float_const.data<float>() + i);
          TEST_RETURN_IF_NOT(float_const_value == i * 1.0f);
        }
      } else if (entry.first.compare(sub_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
        for (int i = 0; i < 8; ++i) {
          float float_const_value = *(float_const.data<float>() + i);
          TEST_RETURN_IF_NOT(float_const_value == i * 1.0f);
        }
      } else if (entry.first.compare(add_initializer->Name()) == 0) {
        TEST_RETURN_IF_NOT(data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
        for (int i = 0; i < 8; ++i) {
          float float_const_value = math::halfToFloat((float_const.data<MLFloat16>() + i)->val);
          TEST_RETURN_IF_NOT(float_const_value == i * 1.0f);
        }
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 5U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 3);
    TEST_RETURN_IF_NOT(op_count["Sub"] == 3);
    TEST_RETURN_IF_NOT(op_count["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count["Add"] == 3);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14};

  std::vector<float> values{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<MLFloat16> values_float16;
  values_float16.reserve(values.size());
  for (auto v : values) {
    values_float16.push_back(MLFloat16(v));
  }

  auto build_test_case_float = [&values, &values_float16](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({{1, 1, 8, 8}});
    auto* input1_arg = builder.MakeInput<float>({{1, 1, 8, 8}});
    auto* div_out = builder.MakeIntermediate();
    builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

    for (size_t i = 0; i < 3; ++i) {
      NodeArg* mul_initializer = builder.MakeInitializer<float>({1, 8}, values);
      auto* mul_out = builder.MakeOutput();
      builder.AddNode("Mul", {div_out, mul_initializer}, {mul_out});
    }

    for (size_t i = 0; i < 3; ++i) {
      NodeArg* sub_initializer = builder.MakeInitializer<float>({8, 1}, values);
      auto* sub_out = builder.MakeOutput();
      builder.AddNode("Sub", {div_out, sub_initializer}, {sub_out});
    }

    auto* cast_out = builder.MakeIntermediate();
    builder.AddNode("Cast", {div_out}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
    for (size_t i = 0; i < 3; ++i) {
      NodeArg* add_initializer = builder.MakeInitializer<MLFloat16>({1, 8}, values_float16);
      auto* add_out = builder.MakeOutput();
      builder.AddNode("Add", {cast_out, add_initializer}, {add_out});
    }
  };

  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [1, 1, 256, 256] (float)
                 |
                Div
            /    |  \______________________________________________________
           /     |  infinity (float)                                       Cast
          / ...  | /                                    _________/     /   \      \_____________
        Sub      Sub                               ____/    __________/     \________           |
         |       |                                /        /                         \          |
         |       |                               /        | int64_max (int64_t)      |          |
         |       |                              /         | /                        |          |
         |       |                             Mul   ...  Mul                        Mul  ...  Mul
                                                |         |                          |          |
graph out [1, 1, 256, 256] (float)        graph out [1, 1, 256, 256] (int64_t)   graph out [1, 1, 256, 256] (int64_t)

Be noted:
 the Sub's input initializer is a scalar std::numeric_limits<float>::infinity();
 the Mul's input initializer is a scalar std::numeric_limits<int64_t>::max().
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShareIntMaxOrFloatInfinityInitializer) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 4U);
    TEST_RETURN_IF_NOT(op_count_pre["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Mul"] == 12);
    TEST_RETURN_IF_NOT(op_count_pre["Sub"] == 12);
    TEST_RETURN_IF_NOT(graph.GetAllInitializedTensors().size() == 24U);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    TEST_RETURN_IF_NOT(initialized_tensor_set.size() == 2U);
    const NodeArg* mul_initializer = nullptr;
    const NodeArg* sub_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Mul") == 0) {
        if (!mul_initializer) {
          mul_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(mul_initializer == nullptr);
          TEST_RETURN_IF_NOT(mul_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(mul_initializer == node.InputDefs()[1]);
        }
      } else if (node.OpType().compare("Sub") == 0) {
        if (!sub_initializer) {
          sub_initializer = node.InputDefs()[1];
          TEST_RETURN_IF(sub_initializer == nullptr);
          TEST_RETURN_IF_NOT(sub_initializer->Shape()->dim_size() == 0);
        } else {
          TEST_RETURN_IF_NOT(sub_initializer == node.InputDefs()[1]);
        }
      }
    }
    TEST_RETURN_IF(mul_initializer == nullptr);
    TEST_RETURN_IF(sub_initializer == nullptr);
    for (const auto& entry : initialized_tensor_set) {
      if (entry.first.compare(mul_initializer->Name()) == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        onnxruntime::Initializer int64_const{*tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(int64_const.size() == 1U);
        int64_t int64_const_value = *(int64_const.data<int64_t>());
        TEST_RETURN_IF_NOT(int64_const_value == std::numeric_limits<int64_t>::max());
      } else if (entry.first.compare(sub_initializer->Name()) == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        onnxruntime::Initializer float_const{*tensor_proto, graph.ModelPath()};
        TEST_RETURN_IF_NOT(float_const.size() == 1U);
        float float_const_value = *(float_const.data<float>());
        TEST_RETURN_IF_NOT(float_const_value == std::numeric_limits<float>::infinity());
      }
    }

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 4U);
    TEST_RETURN_IF_NOT(op_count["Div"] == 1);
    TEST_RETURN_IF_NOT(op_count["Mul"] == 12);
    TEST_RETURN_IF_NOT(op_count["Sub"] == 12);
    return Status::OK();
  };

  const std::vector<int> opsets{12, 13, 14};

  // Float data type tests.
  auto build_test_case_float = [&](ModelTestBuilder& builder) {
    auto* input0_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* input1_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* div_out = builder.MakeIntermediate();
    builder.AddNode("Div", {input0_arg, input1_arg}, {div_out});

    auto* cast_out = builder.MakeIntermediate();
    builder.AddNode("Cast", {div_out}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT64));
    for (size_t i = 0; i < 12; ++i) {
      NodeArg* mul_initializer = nullptr;
      mul_initializer = builder.MakeScalarInitializer<int64_t>(std::numeric_limits<int64_t>::max());
      auto* mul_out = builder.MakeOutput();
      builder.AddNode("Mul", {cast_out, mul_initializer}, {mul_out});
    }

    for (size_t i = 0; i < 12; ++i) {
      NodeArg* sub_initializer = nullptr;
      sub_initializer = builder.MakeScalarInitializer<float>(std::numeric_limits<float>::infinity());
      auto* sub_out = builder.MakeOutput();
      builder.AddNode("Sub", {div_out, sub_initializer}, {sub_out});
    }
  };
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case_float, opset_version, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph as below.
           graph input [2] (float)                Constant (1.0float)                Constant (1.0uint8)
                  \_______________  ________________/        |                             |
                                  \/                         |                             |
                                 Add                         |                             |
                                  |                          |                             |
                            graph output [2](float)   graph output [](float)       graph output [](int8)

Be noted: expected result graph should maintain original graph outputs,
   both float and unin8 constant values are not shared.
*/
TEST_F(GraphTransformationTests, ConstantSharing_ShouldNotShareForGraphOutput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "scalar_const_not_share.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  {
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["Add"] == 1);
    // Be noted, constant nodes are converted to initialized already.
    ASSERT_TRUE(graph.GetAllInitializedTensors().size() == 2U);
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<GraphTransformer> transformer = std::make_unique<ConstantSharing>();
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(transformer), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  {
    const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
    ASSERT_TRUE(initialized_tensor_set.size() == 2U);
    const NodeArg* add_initializer = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("Add") == 0) {
        add_initializer = node.InputDefs()[1];
        ASSERT_TRUE(add_initializer->Shape()->dim_size() == 0);
        ASSERT_TRUE(add_initializer->Name().compare("y_scale") == 0);
      }
    }
    ASSERT_TRUE(add_initializer != nullptr);
    for (const auto& entry : initialized_tensor_set) {
      if (entry.first.compare("y_scale") == 0) {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        onnxruntime::Initializer int64_const{*tensor_proto, graph.ModelPath()};
        ASSERT_TRUE(int64_const.size() == 1U);
        float float_const_value = *(int64_const.data<float>());
        ASSERT_TRUE(float_const_value == 1);
      } else {
        const ONNX_NAMESPACE::TensorProto* tensor_proto = entry.second;
        onnxruntime::Initializer uint8_const{*tensor_proto, graph.ModelPath()};
        ASSERT_TRUE(uint8_const.size() == 1U);
        uint8_t uint8_const_value = *(uint8_const.data<uint8_t>());
        ASSERT_TRUE(uint8_const_value == static_cast<uint8_t>(1));
      }
    }

    auto op_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_count.size() == 1U);
    ASSERT_TRUE(op_count["Add"] == 1);
  }
}

TEST_F(GraphTransformationTests, GatherToSplitFusion) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* data_arg = builder.MakeInput<float>({{54}});
    auto* shape_arg = builder.MakeInput<int64_t>({{4}});
    auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 3, 3}});
    auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
    auto* gather_index_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
    auto* gather_index_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(2)});
    auto* gather_out_1 = builder.MakeIntermediate();
    auto* gather_out_2 = builder.MakeIntermediate();
    auto* gather_out_3 = builder.MakeIntermediate();
    auto* transpose_out_1 = builder.MakeOutput();
    auto* transpose_out_2 = builder.MakeOutput();
    auto* transpose_out_3 = builder.MakeOutput();

    builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
    builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
        .AddAttribute("axis", static_cast<int64_t>(2));
    builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
        .AddAttribute("axis", static_cast<int64_t>(-2));
    builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
        .AddAttribute("axis", static_cast<int64_t>(2));
    builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto pre_graph_checker = [&](Graph& graph) { TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 3); return Status::OK(); };

  // OpSet-12
  {
    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 3);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Split") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
          TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
        } else if (node.OpType() == "Squeeze") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("axes") != attrs.end());
          TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axes").ints().at(0)));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // OpSet-14
  {
    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 3);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Split") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
          TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
        } else if (node.OpType() == "Squeeze") {
          const NodeArg& input_arg = *(node.InputDefs()[1]);
          const ONNX_NAMESPACE::TensorProto* tensor_proto =
              graph_utils::GetConstantInitializer(graph, input_arg.Name());
          TEST_RETURN_IF_NOT(tensor_proto != nullptr);
          Initializer init_const{*tensor_proto, graph.ModelPath()};
          TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64);
          TEST_RETURN_IF_NOT(2 == static_cast<int>(*(init_const.data<int64_t>())));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // OpSet-18
  {
    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 3);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Split") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
          TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
        } else if (node.OpType() == "Squeeze") {
          const NodeArg& input_arg = *(node.InputDefs()[1]);
          const ONNX_NAMESPACE::TensorProto* tensor_proto =
              graph_utils::GetConstantInitializer(graph, input_arg.Name());
          TEST_RETURN_IF_NOT(tensor_proto != nullptr);
          Initializer init_const{*tensor_proto, graph.ModelPath()};
          TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64);
          TEST_RETURN_IF_NOT(2 == static_cast<int>(*(init_const.data<int64_t>())));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, GatherToSplitFusion_NoSqueeze) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* data_arg = builder.MakeInput<float>({{54}});
    auto* shape_arg = builder.MakeInput<int64_t>({{4}});
    auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 3, 3}});
    auto* gather_index_1 = builder.MakeInitializer<int64_t>({1}, {static_cast<int64_t>(0)});
    auto* gather_index_2 = builder.MakeInitializer<int64_t>({1}, {static_cast<int64_t>(1)});
    auto* gather_index_3 = builder.MakeInitializer<int64_t>({1}, {static_cast<int64_t>(2)});
    auto* gather_out_1 = builder.MakeIntermediate();
    auto* gather_out_2 = builder.MakeIntermediate();
    auto* gather_out_3 = builder.MakeIntermediate();
    auto* transpose_out_1 = builder.MakeOutput();
    auto* transpose_out_2 = builder.MakeOutput();
    auto* transpose_out_3 = builder.MakeOutput();

    builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
    builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
        .AddAttribute("axis", static_cast<int64_t>(2));
    builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
        .AddAttribute("axis", static_cast<int64_t>(-2));
    builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
        .AddAttribute("axis", static_cast<int64_t>(2));
    builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3}).AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
  };

  auto pre_graph_checker = [&](Graph& graph) { TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 3); return Status::OK(); };

  // OpSet-12
  {
    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 0);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Split") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
          TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // OpSet-14
  {
    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 0);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Split") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
          TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // OpSet-18
  {
    auto post_graph_checker = [&](Graph& graph) {
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 0);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 1);
      TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 0);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Split") {
          auto& attrs = node.GetAttributes();
          TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
          TEST_RETURN_IF_NOT(2 == static_cast<int>(attrs.at("axis").i()));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, GatherToSplitFusion_Invalid) {
  auto pre_graph_checker = [&](Graph& graph) { TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 3); return Status::OK(); };
  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Gather"] == 3);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Split"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Squeeze"] == 0);
    return Status::OK();
  };

  // Invalid shape.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{72}});
      auto* shape_arg = builder.MakeInput<int64_t>({{1}});
      auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 4, 3}});
      auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
      auto* gather_index_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
      auto* gather_index_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(2)});
      auto* gather_out_1 = builder.MakeIntermediate();
      auto* gather_out_2 = builder.MakeIntermediate();
      auto* gather_out_3 = builder.MakeIntermediate();
      auto* transpose_out_1 = builder.MakeOutput();
      auto* transpose_out_2 = builder.MakeOutput();
      auto* transpose_out_3 = builder.MakeOutput();

      builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
      builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Invalid Gather indices.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{54}});
      auto* shape_arg = builder.MakeInput<int64_t>({{1}});
      auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 3, 3}});
      auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
      auto* gather_index_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
      auto* gather_index_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
      auto* gather_out_1 = builder.MakeIntermediate();
      auto* gather_out_2 = builder.MakeIntermediate();
      auto* gather_out_3 = builder.MakeIntermediate();
      auto* transpose_out_1 = builder.MakeOutput();
      auto* transpose_out_2 = builder.MakeOutput();
      auto* transpose_out_3 = builder.MakeOutput();

      builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
      builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Invalid Gather axis.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{54}});
      auto* shape_arg = builder.MakeInput<int64_t>({{1}});
      auto* reshape_out = builder.MakeIntermediate<float>({{2, 3, 3, 3}});
      auto* gather_index_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
      auto* gather_index_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
      auto* gather_index_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(2)});
      auto* gather_out_1 = builder.MakeIntermediate();
      auto* gather_out_2 = builder.MakeIntermediate();
      auto* gather_out_3 = builder.MakeIntermediate();
      auto* transpose_out_1 = builder.MakeOutput();
      auto* transpose_out_2 = builder.MakeOutput();
      auto* transpose_out_3 = builder.MakeOutput();

      builder.AddNode("Reshape", {data_arg, shape_arg}, {reshape_out});
      builder.AddNode("Gather", {reshape_out, gather_index_1}, {gather_out_1})
          .AddAttribute("axis", static_cast<int64_t>(1));
      builder.AddNode("Gather", {reshape_out, gather_index_2}, {gather_out_2})
          .AddAttribute("axis", static_cast<int64_t>(2));
      builder.AddNode("Gather", {reshape_out, gather_index_3}, {gather_out_3})
          .AddAttribute("axis", static_cast<int64_t>(3));
      builder.AddNode("Transpose", {gather_out_1}, {transpose_out_1})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_2}, {transpose_out_2})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
      builder.AddNode("Transpose", {gather_out_3}, {transpose_out_3})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSplitFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

TEST_F(GraphTransformationTests, GatherToSliceFusion) {
  auto pre_graph_checker = [&](Graph& graph) {
    auto op_count_map = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_map["Range"] == 1);
    TEST_RETURN_IF_NOT(op_count_map["Gather"] == 1);
    return Status::OK();
  };

  // OpSet-12, Tind is int32.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{8, 8, 8, 8}});
      auto* range_input_1 = builder.MakeInitializer<int32_t>({}, {0});
      auto* range_input_2 = builder.MakeInitializer<int32_t>({}, {8});
      auto* range_input_3 = builder.MakeInitializer<int32_t>({}, {1});
      auto* range_output = builder.MakeIntermediate();
      auto* gather_output = builder.MakeOutput();

      builder.AddNode("Range", {range_input_1, range_input_2, range_input_3}, {range_output});
      builder.AddNode("Gather", {data_arg, range_output}, {gather_output})
          .AddAttribute("axis", static_cast<int64_t>(2));
    };

    auto post_graph_checker = [&](Graph& graph) {
      auto op_count_map = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_map["Range"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Gather"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Slice"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Slice") {
          const NodeArg& input_arg = *(node.InputDefs()[3]);
          const ONNX_NAMESPACE::TensorProto* tensor_proto =
              graph_utils::GetConstantInitializer(graph, input_arg.Name());
          TEST_RETURN_IF_NOT(tensor_proto != nullptr);
          Initializer init_const{*tensor_proto, graph.ModelPath()};
          TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32);
          TEST_RETURN_IF_NOT(2 == *(init_const.data<int32_t>()));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSliceFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 12, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // OpSet-14, Tind is int64.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* data_arg = builder.MakeInput<float>({{8, 8, 8, 8}});
      auto* range_input_1 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(0)});
      auto* range_input_2 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(8)});
      auto* range_input_3 = builder.MakeInitializer<int64_t>({}, {static_cast<int64_t>(1)});
      auto* range_output = builder.MakeIntermediate();
      auto* gather_output = builder.MakeOutput();

      builder.AddNode("Range", {range_input_1, range_input_2, range_input_3}, {range_output});
      builder.AddNode("Gather", {data_arg, range_output}, {gather_output})
          .AddAttribute("axis", static_cast<int64_t>(2));
    };

    auto post_graph_checker = [&](Graph& graph) {
      auto op_count_map = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_map["Range"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Gather"] == 0);
      TEST_RETURN_IF_NOT(op_count_map["Slice"] == 1);
      for (auto& node : graph.Nodes()) {
        if (node.OpType() == "Slice") {
          const NodeArg& input_arg = *(node.InputDefs()[3]);
          const ONNX_NAMESPACE::TensorProto* tensor_proto =
              graph_utils::GetConstantInitializer(graph, input_arg.Name());
          TEST_RETURN_IF_NOT(tensor_proto != nullptr);
          Initializer init_const{*tensor_proto, graph.ModelPath()};
          TEST_RETURN_IF_NOT(tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64);
          TEST_RETURN_IF_NOT(2 == static_cast<int32_t>(*(init_const.data<int64_t>())));
        }
      }
      return Status::OK();
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<GatherToSliceFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level1, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

}  // namespace test
}  // namespace onnxruntime
