// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/graph/model.h"

#include "test/framework/test_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
#include "orttraining/core/optimizer/megatron_transformer.h"
#include "orttraining/core/optimizer/concat_replacement.h"
#include "orttraining/core/optimizer/batchnorm_replacement.h"
#include "orttraining/core/optimizer/localized_recompute.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/asserts.h"
#include "orttraining/test/optimizer/horizontal_parallel_test_utils.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/optimizer/loss_rewriter.h"
#include "orttraining/core/optimizer/bias_softmax_dropout_fusion.h"
#include "orttraining/core/optimizer/sce_loss_grad_bias_fusion.h"
#include "orttraining/core/optimizer/qdq_fusion.h"

#include <random>

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

TEST_F(GraphTransformationTests, BatchNormReplacement) {
  Model model("BatchNormReplacement", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 14}, {"com.microsoft", 1}},
              {}, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // 1x3x3x3
  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  TypeProto scale_tensor_type;
  scale_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  scale_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto& input_X = graph.GetOrCreateNodeArg("X", &input_tensor_type);
  auto& input_scale = graph.GetOrCreateNodeArg("scale", &scale_tensor_type);
  auto& input_B = graph.GetOrCreateNodeArg("B", &scale_tensor_type);
  auto& input_mean = graph.GetOrCreateNodeArg("input_mean", &scale_tensor_type);
  auto& input_var = graph.GetOrCreateNodeArg("input_var", &scale_tensor_type);

  auto& output_Y = graph.GetOrCreateNodeArg("Y", &input_tensor_type);
  graph.AddNode("BN", "BatchNormalization", "", {&input_X, &input_scale, &input_B, &input_mean, &input_var}, {&output_Y});

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("BatchNormReplacement");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<BatchNormReplacement>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  ASSERT_TRUE(graph.NumberOfNodes() == 1);
  // Make sure that BN was updated to add outputs
  ASSERT_TRUE(graph.Nodes().begin()->MutableOutputDefs().size() == 5);
  ASSERT_TRUE(graph.Nodes().begin()->OpType().compare("BatchNormInternal") == 0);
}

TEST_F(GraphTransformationTests, BatchNormReplacementWithOptionalOutputPresentOpset14) {
  Model model("BatchNormReplacement", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 14}, {"com.microsoft", 1}},
              {}, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // 1x3x3x3
  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  TypeProto scale_tensor_type;
  scale_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  scale_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto& input_X = graph.GetOrCreateNodeArg("X", &input_tensor_type);
  auto& input_scale = graph.GetOrCreateNodeArg("scale", &scale_tensor_type);
  auto& input_B = graph.GetOrCreateNodeArg("B", &scale_tensor_type);
  auto& input_mean = graph.GetOrCreateNodeArg("input_mean", &scale_tensor_type);
  auto& input_var = graph.GetOrCreateNodeArg("input_var", &scale_tensor_type);

  auto& output_Y = graph.GetOrCreateNodeArg("Y", &input_tensor_type);
  auto& output_running_mean = graph.GetOrCreateNodeArg("running_mean", &scale_tensor_type);
  auto& output_running_var = graph.GetOrCreateNodeArg("running_var", &scale_tensor_type);
  auto& bn_node = graph.AddNode("BN", "BatchNormalization", "", {&input_X, &input_scale, &input_B, &input_mean, &input_var},
                                {&output_Y, &output_running_mean, &output_running_var});
  bn_node.AddAttribute("training_mode", static_cast<int64_t>(1));

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("BatchNormReplacement");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<BatchNormReplacement>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  ASSERT_TRUE(graph.NumberOfNodes() == 1);
  // Make sure that BN was updated to add outputs
  ASSERT_TRUE(graph.Nodes().begin()->MutableOutputDefs().size() == 5);
  ASSERT_TRUE(graph.Nodes().begin()->OpType().compare("BatchNormInternal") == 0);
}

TEST_F(GraphTransformationTests, BatchNormReplacementWithOptionalOutputPresentOpset9) {
  Model model("BatchNormReplacement", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 9}, {"com.microsoft", 1}},
              {}, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // 1x3x3x3
  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  TypeProto scale_tensor_type;
  scale_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  scale_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto& input_X = graph.GetOrCreateNodeArg("X", &input_tensor_type);
  auto& input_scale = graph.GetOrCreateNodeArg("scale", &scale_tensor_type);
  auto& input_B = graph.GetOrCreateNodeArg("B", &scale_tensor_type);
  auto& input_mean = graph.GetOrCreateNodeArg("input_mean", &scale_tensor_type);
  auto& input_var = graph.GetOrCreateNodeArg("input_var", &scale_tensor_type);

  auto& output_Y = graph.GetOrCreateNodeArg("Y", &input_tensor_type);
  auto& output_running_mean = graph.GetOrCreateNodeArg("running_mean", &scale_tensor_type);
  auto& output_running_var = graph.GetOrCreateNodeArg("running_var", &scale_tensor_type);
  auto& saved_mean = graph.GetOrCreateNodeArg("saved_mean", &scale_tensor_type);
  auto& saved_var = graph.GetOrCreateNodeArg("saved_var", &scale_tensor_type);
  graph.AddNode("BN", "BatchNormalization", "", {&input_X, &input_scale, &input_B, &input_mean, &input_var},
                {&output_Y, &output_running_mean, &output_running_var, &saved_mean, &saved_var});

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("BatchNormReplacement");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<BatchNormReplacement>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  ASSERT_TRUE(graph.NumberOfNodes() == 1);
  // Make sure that BN was updated to add outputs
  ASSERT_TRUE(graph.Nodes().begin()->MutableOutputDefs().size() == 5);
  ASSERT_TRUE(graph.Nodes().begin()->OpType().compare("BatchNormInternal") == 0);
}

TEST_F(GraphTransformationTests, DropoutWithZeroRatioElimination) {
  auto model_uri = MODEL_FOLDER "dropout_ratio.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 10);
  ASSERT_TRUE(op_to_count["Dropout"] == 5);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<EliminateDropout>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  ASSERT_TRUE(op_to_count["Identity"] == 10);
  ASSERT_TRUE(op_to_count["Dropout"] == 2);
}

template <typename T>
void RunBiasSoftmaxDropoutFusionTest(bool is_bitmask_dropout, bool is_softmax_grad_13, int opset_version,
                                     const logging::Logger& logger) {
  const std::string dropout_op_type = is_bitmask_dropout ? "BitmaskDropout" : "Dropout";
  const std::string dropout_grad_op_type = is_bitmask_dropout ? "BitmaskDropoutGrad" : "DropoutGrad";
  const std::string softmax_grad_op_typ = is_softmax_grad_13 ? "SoftmaxGrad_13" : "SoftmaxGrad";
  const std::string ms_domain_prefix = "com.microsoft.";

  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<T>({{2, 3, 3, 3, 2, 3, 3, 3}});
    auto* bias_arg = builder.MakeInput<T>({{2, 3, 3, 3}});
    auto* ratio_arg = builder.MakeInitializer<T>({}, {T(0.5f)});
    auto* training_mode_arg = builder.MakeInitializerBool({}, std::vector<bool>{true});
    auto* dy_arg = builder.MakeInput<T>({{2, 3, 3, 3, 2, 3, 3, 3}});
    auto* bias_softmax_out = builder.MakeIntermediate();
    auto* dropout_mask_out = builder.MakeIntermediate();
    auto* dropout_grad_out = builder.MakeIntermediate();
    auto* dropout_out = builder.MakeOutput();
    auto* dx_out = builder.MakeOutput();

    Node& node = builder.AddNode("BiasSoftmax", {input_arg, bias_arg}, {bias_softmax_out}, kMSDomain);
    node.AddAttribute("axis", static_cast<int64_t>(6));
    node.AddAttribute("is_inner_broadcast", static_cast<int64_t>(0));
    builder
        .AddNode(dropout_op_type, {bias_softmax_out, ratio_arg, training_mode_arg}, {dropout_out, dropout_mask_out},
                 is_bitmask_dropout ? kMSDomain : kOnnxDomain)
        .AddAttribute("seed", static_cast<int64_t>(42));
    builder.AddNode(dropout_grad_op_type, {dy_arg, dropout_mask_out, ratio_arg, training_mode_arg}, {dropout_grad_out},
                    kMSDomain);
    builder.AddNode(softmax_grad_op_typ, {dropout_grad_out, bias_softmax_out}, {dx_out}, kMSDomain)
        .AddAttribute("axis", static_cast<int64_t>(6));
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.BiasSoftmax"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)[(is_bitmask_dropout ? ms_domain_prefix : "") + dropout_op_type] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)[ms_domain_prefix + dropout_grad_op_type] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)[ms_domain_prefix + softmax_grad_op_typ] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.BiasSoftmax"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)[(is_bitmask_dropout ? ms_domain_prefix : "") + dropout_op_type] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)[ms_domain_prefix + dropout_grad_op_type] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)[ms_domain_prefix + softmax_grad_op_typ] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.BiasSoftmaxDropout"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.SoftmaxDropoutGrad"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "BiasSoftmaxDropout") {
        auto& attrs = node.GetAttributes();
        TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
        TEST_RETURN_IF_NOT(attrs.find("is_inner_broadcast") != attrs.end());
        TEST_RETURN_IF_NOT(attrs.find("seed") != attrs.end());
        TEST_RETURN_IF_NOT(6 == static_cast<int>(attrs.at("axis").i()));
        TEST_RETURN_IF_NOT(0 == static_cast<int>(attrs.at("is_inner_broadcast").i()));
        TEST_RETURN_IF_NOT(42 == static_cast<int>(attrs.at("seed").i()));
      } else if (node.OpType() == "SoftmaxDropoutGrad") {
        auto& attrs = node.GetAttributes();
        TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());
        TEST_RETURN_IF_NOT(6 == static_cast<int>(attrs.at("axis").i()));
      }
    }
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<BiasSoftmaxDropoutFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, logger, std::move(transformer), TransformerLevel::Level2, 1,
                                        pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, BiasSoftmaxDropoutFusion) {
  // Dropout.
  RunBiasSoftmaxDropoutFusionTest<float>(false, false, 12, *logger_);
  // BitmaskDropout.
  RunBiasSoftmaxDropoutFusionTest<MLFloat16>(true, false, 12, *logger_);
  // SoftmaxGrad_13.
  RunBiasSoftmaxDropoutFusionTest<float>(false, true, 14, *logger_);
  // BitmaskDropout and SoftmaxGrad_13.
  RunBiasSoftmaxDropoutFusionTest<MLFloat16>(true, true, 14, *logger_);
}

template <typename T>
void RunSceLossGradBiasFusionTest(bool has_reshape, bool is_add_op, bool is_bias_lhs_input, bool has_weight,
                                  bool has_ignore_index, const std::string& reduction, int opset_version,
                                  const logging::Logger& logger) {
  std::string bias_op_type = is_add_op ? "Add" : "Sum";
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* dY_arg = builder.MakeInput<T>(std::optional<std::vector<int64_t>>{std::vector<int64_t>{}});
    auto* log_prob_arg = builder.MakeInput<T>({{8, 2}});
    auto* index_arg = builder.MakeInput<int64_t>({{8}});
    std::vector<NodeArg*> scegrad_inputs{dY_arg, log_prob_arg, index_arg};
    if (has_weight || has_ignore_index) {
      auto* weight_arg = builder.MakeInput<T>({{2}});
      scegrad_inputs.emplace_back(weight_arg);
    }
    if (has_ignore_index) {
      auto* ignore_index_arg = builder.MakeInput<int64_t>(std::optional<std::vector<int64_t>>{std::vector<int64_t>{}});
      scegrad_inputs.emplace_back(ignore_index_arg);
    }
    auto* sce_grad_out = builder.MakeIntermediate();
    std::vector<NodeArg*> reshape_inputs;
    std::vector<NodeArg*> reshape_outputs;
    std::vector<NodeArg*> bias_op_inputs;
    if (has_reshape) {
      reshape_inputs.emplace_back(sce_grad_out);
      auto* shape_arg = builder.MakeInput<int64_t>({{1}});
      reshape_inputs.emplace_back(shape_arg);
      auto* reshape_out = builder.MakeIntermediate<T>({{16}});
      reshape_outputs.emplace_back(reshape_out);
      auto* bias_arg = builder.MakeInput<T>({{16}});
      if (is_bias_lhs_input) {
        bias_op_inputs.emplace_back(bias_arg);
        bias_op_inputs.emplace_back(reshape_out);
      } else {
        bias_op_inputs.emplace_back(reshape_out);
        bias_op_inputs.emplace_back(bias_arg);
      }
    } else {
      auto* bias_arg = builder.MakeInput<T>({{8, 2}});
      if (is_bias_lhs_input) {
        bias_op_inputs.emplace_back(bias_arg);
        bias_op_inputs.emplace_back(sce_grad_out);
      } else {
        bias_op_inputs.emplace_back(sce_grad_out);
        bias_op_inputs.emplace_back(bias_arg);
      }
    }
    auto* dx_out = builder.MakeOutput();

    builder.AddNode("SoftmaxCrossEntropyLossInternalGrad", scegrad_inputs, {sce_grad_out}, kMSDomain)
        .AddAttribute("reduction", reduction);
    if (has_reshape) {
      builder.AddNode("Reshape", reshape_inputs, reshape_outputs);
    }
    builder.AddNode(bias_op_type, bias_op_inputs, {dx_out});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.SoftmaxCrossEntropyLossInternalGrad"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)[bias_op_type] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.SoftmaxCrossEntropyLossInternalGrad"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)[bias_op_type] == 0);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SoftmaxCrossEntropyLossInternalGrad") {
        auto& attrs = node.GetAttributes();
        TEST_RETURN_IF_NOT(attrs.find("reduction") != attrs.end());
        TEST_RETURN_IF_NOT(reduction == attrs.at("reduction").s());
        TEST_RETURN_IF_NOT(6 == static_cast<int>(node.InputDefs().size()));
      }
    }
    return Status::OK();
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<SceLossGradBiasFusion>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, logger, std::move(transformer), TransformerLevel::Level2, 1,
                                        pre_graph_checker, post_graph_checker));
}

void RunSceLossGradBiasFusionTestWrapper(int opset_version, const logging::Logger& logger) {
  RunSceLossGradBiasFusionTest<float>(false, false, false, true, true, "none", opset_version, logger);
  RunSceLossGradBiasFusionTest<MLFloat16>(false, false, true, true, false, "mean", opset_version, logger);
  RunSceLossGradBiasFusionTest<float>(false, false, false, false, false, "sum", opset_version, logger);
  RunSceLossGradBiasFusionTest<MLFloat16>(false, true, true, true, true, "none", opset_version, logger);
  RunSceLossGradBiasFusionTest<float>(false, true, false, true, false, "mean", opset_version, logger);
  RunSceLossGradBiasFusionTest<MLFloat16>(false, true, true, false, false, "sum", opset_version, logger);
  RunSceLossGradBiasFusionTest<float>(true, false, false, true, true, "none", opset_version, logger);
  RunSceLossGradBiasFusionTest<MLFloat16>(true, false, true, true, false, "mean", opset_version, logger);
  RunSceLossGradBiasFusionTest<float>(true, false, false, false, false, "sum", opset_version, logger);
  RunSceLossGradBiasFusionTest<MLFloat16>(true, true, true, true, true, "none", opset_version, logger);
  RunSceLossGradBiasFusionTest<float>(true, true, false, true, false, "mean", opset_version, logger);
  RunSceLossGradBiasFusionTest<MLFloat16>(true, true, true, false, false, "sum", opset_version, logger);
}

TEST_F(GraphTransformationTests, SceLossGradBiasFusion) {
  RunSceLossGradBiasFusionTestWrapper(12, *logger_);
  RunSceLossGradBiasFusionTestWrapper(13, *logger_);
  RunSceLossGradBiasFusionTestWrapper(14, *logger_);
}

TEST_F(GraphTransformationTests, SceLossGradBiasFusion_Invalid) {
  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.SoftmaxCrossEntropyLossInternalGrad"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sum"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.SoftmaxCrossEntropyLossInternalGrad"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sum"] == 1);
    return Status::OK();
  };

  // Sum has more than 2 inputs.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* dY_arg = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{std::vector<int64_t>{}});
      auto* log_prob_arg = builder.MakeInput<float>({{8, 2}});
      auto* index_arg = builder.MakeInput<int64_t>({{8}});
      auto* sce_grad_out = builder.MakeIntermediate();
      auto* bias1_arg = builder.MakeInput<float>({{8, 2}});
      auto* bias2_arg = builder.MakeInput<float>({{8, 2}});
      auto* dx_out = builder.MakeOutput();
      builder
          .AddNode("SoftmaxCrossEntropyLossInternalGrad", {dY_arg, log_prob_arg, index_arg}, {sce_grad_out}, kMSDomain)
          .AddAttribute("reduction", "sum");
      builder.AddNode("Sum", {sce_grad_out, bias1_arg, bias2_arg}, {dx_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<SceLossGradBiasFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // SceGrad has more than 1 consumers.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* dY_arg = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{std::vector<int64_t>{}});
      auto* log_prob_arg = builder.MakeInput<float>({{8, 2}});
      auto* index_arg = builder.MakeInput<int64_t>({{8}});
      auto* sce_grad_out = builder.MakeIntermediate();
      auto* bias_arg = builder.MakeInput<float>({{8, 2}});
      auto* dx_out = builder.MakeOutput();
      auto* identity_out = builder.MakeOutput();
      builder
          .AddNode("SoftmaxCrossEntropyLossInternalGrad", {dY_arg, log_prob_arg, index_arg}, {sce_grad_out}, kMSDomain)
          .AddAttribute("reduction", "sum");
      builder.AddNode("Sum", {sce_grad_out, bias_arg}, {dx_out});
      builder.AddNode("Identity", {sce_grad_out}, {identity_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<SceLossGradBiasFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Sum inputs shape mismatch.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* dY_arg = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{std::vector<int64_t>{}});
      auto* log_prob_arg = builder.MakeInput<float>({{8, 2}});
      auto* index_arg = builder.MakeInput<int64_t>({{8}});
      auto* sce_grad_out = builder.MakeIntermediate();
      auto* bias_arg = builder.MakeInput<float>({{2}});
      auto* dx_out = builder.MakeOutput();
      builder
          .AddNode("SoftmaxCrossEntropyLossInternalGrad", {dY_arg, log_prob_arg, index_arg}, {sce_grad_out}, kMSDomain)
          .AddAttribute("reduction", "sum");
      builder.AddNode("Sum", {sce_grad_out, bias_arg}, {dx_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<SceLossGradBiasFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Sum inputs shape mismatch.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* dY_arg = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{std::vector<int64_t>{}});
      auto* log_prob_arg = builder.MakeInput<float>({{8, 1}});
      auto* index_arg = builder.MakeInput<int64_t>({{8}});
      auto* bias_arg = builder.MakeInput<float>({{8, 1}});
      auto* sce_grad_out = builder.MakeIntermediate();
      auto* shape_arg = builder.MakeInput<int64_t>({{2}});
      auto* reshape_out = builder.MakeIntermediate<float>({{1, 8}});
      auto* dx_out = builder.MakeOutput();
      builder
          .AddNode("SoftmaxCrossEntropyLossInternalGrad", {dY_arg, log_prob_arg, index_arg}, {sce_grad_out}, kMSDomain)
          .AddAttribute("reduction", "sum");
      builder.AddNode("Reshape", {sce_grad_out, shape_arg}, {reshape_out});
      builder.AddNode("Sum", {reshape_out, bias_arg}, {dx_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<SceLossGradBiasFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }

  // Reshape output has more than 1 consumers.
  {
    auto build_test_case = [&](ModelTestBuilder& builder) {
      auto* dY_arg = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{std::vector<int64_t>{}});
      auto* log_prob_arg = builder.MakeInput<float>({{8, 2}});
      auto* index_arg = builder.MakeInput<int64_t>({{8}});
      auto* bias_arg = builder.MakeInput<float>({{16}});
      auto* sce_grad_out = builder.MakeIntermediate();
      auto* shape_arg = builder.MakeInput<int64_t>({{1}});
      auto* reshape_out = builder.MakeIntermediate<float>({{16}});
      auto* dx_out = builder.MakeOutput();
      auto* identity_out = builder.MakeOutput();
      builder
          .AddNode("SoftmaxCrossEntropyLossInternalGrad", {dY_arg, log_prob_arg, index_arg}, {sce_grad_out}, kMSDomain)
          .AddAttribute("reduction", "sum");
      builder.AddNode("Reshape", {sce_grad_out, shape_arg}, {reshape_out});
      builder.AddNode("Sum", {reshape_out, bias_arg}, {dx_out});
      builder.AddNode("Identity", {reshape_out}, {identity_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<SceLossGradBiasFusion>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer), TransformerLevel::Level2, 1,
                                          pre_graph_checker, post_graph_checker));
  }
}

Node* GetNodeByName(Graph& graph, std::string node_name) {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node->Name().compare(node_name) == 0) {
      return p_node;
    }
  }

  return nullptr;
}

// MegatronF/G and ConcatTraining is defined only for training, and in msdomain.
#ifndef DISABLE_CONTRIB_OPS
TEST_F(GraphTransformationTests, ConcatReplacement) {
  auto model_uri = MODEL_FOLDER "concat_trainable.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("ConcatReplacement");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConcatReplacement>()));
  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["com.microsoft.ConcatTraining"], 1);
}

TEST_F(GraphTransformationTests, MegatronMLPPartitionRank0) {
  auto model_uri = MODEL_FOLDER "model_parallel/mlp_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  training::TrainingSession::OptimizerState init_optim_state;
  IExecutionProvider* e = TestCPUExecutionProvider();
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MegatronTransformer>(0, 2, updated_weight_names, weights_to_train, weight_partition_info,
                                            init_optim_state, *e),
      TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  auto model_uri2 = "mlp_megatron_basic_test_partition_rank0.onnx";
  ASSERT_STATUS_OK(Model::Save(*p_model, model_uri2));

  {
    std::vector<float> expected_value = {
        0.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f,
        0.16f, 0.17f, 0.18f, 0.19f, 0.20f, 0.21f, 0.22f, 0.23f,
        0.32f, 0.33f, 0.34f, 0.35f, 0.36f, 0.37f, 0.38f, 0.39f,
        0.48f, 0.49f, 0.50f, 0.51f, 0.52f, 0.53f, 0.54f, 0.55f};
    auto a1_weight_arg = GetNodeByName(graph, "matmul")->MutableInputDefs()[1];
    ORT_ENFORCE(a1_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {4, 8};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, a1_weight_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {0.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f};
    auto input_arg = GetNodeByName(graph, "add")->MutableInputDefs()[1];
    ORT_ENFORCE(input_arg != nullptr);
    std::vector<int64_t> expected_shape = {8};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, input_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {
        0.0f, 0.01f, 0.02f, 0.03f,
        0.04f, 0.05f, 0.06f, 0.07f,
        0.08f, 0.09f, 0.10f, 0.11f,
        0.12f, 0.13f, 0.14f, 0.15f,
        0.16f, 0.17f, 0.18f, 0.19f,
        0.20f, 0.21f, 0.22f, 0.23f,
        0.24f, 0.25f, 0.26f, 0.27f,
        0.28f, 0.29f, 0.30f, 0.31f};
    auto input_arg = GetNodeByName(graph, "matmul2")->MutableInputDefs()[1];
    ORT_ENFORCE(input_arg != nullptr);
    std::vector<int64_t> expected_shape = {8, 4};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, input_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
}

TEST_F(GraphTransformationTests, MegatronMLPPartitionRank1) {
  auto model_uri = MODEL_FOLDER "model_parallel/mlp_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  training::TrainingSession::OptimizerState init_optim_state;
  IExecutionProvider* e = TestCPUExecutionProvider();
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MegatronTransformer>(1, 2, updated_weight_names,
                                                                                           weights_to_train,
                                                                                           weight_partition_info,
                                                                                           init_optim_state, *e),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  auto model_uri2 = "mlp_megatron_basic_test_partition_rank1.onnx";
  ASSERT_STATUS_OK(Model::Save(*p_model, model_uri2));

  {
    std::vector<float> expected_value = {
        0.08f, 0.09f, 0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f,
        0.24f, 0.25f, 0.26f, 0.27f, 0.28f, 0.29f, 0.30f, 0.31f,
        0.40f, 0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f, 0.47f,
        0.56f, 0.57f, 0.58f, 0.59f, 0.60f, 0.61f, 0.62f, 0.63f};
    auto a1_weight_arg = GetNodeByName(graph, "matmul")->MutableInputDefs()[1];
    ORT_ENFORCE(a1_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {4, 8};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, a1_weight_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {0.08f, 0.09f, 0.1f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f};
    auto a1_bias_arg = GetNodeByName(graph, "add")->MutableInputDefs()[1];
    ORT_ENFORCE(a1_bias_arg != nullptr);
    std::vector<int64_t> expected_shape = {8};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, a1_bias_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {
        0.32f, 0.33f, 0.34f, 0.35f,
        0.36f, 0.37f, 0.38f, 0.39f,
        0.40f, 0.41f, 0.42f, 0.43f,
        0.44f, 0.45f, 0.46f, 0.47f,
        0.48f, 0.49f, 0.50f, 0.51f,
        0.52f, 0.53f, 0.54f, 0.55f,
        0.56f, 0.57f, 0.58f, 0.59f,
        0.60f, 0.61f, 0.62f, 0.63f};
    auto input_arg = GetNodeByName(graph, "matmul2")->MutableInputDefs()[1];
    ORT_ENFORCE(input_arg != nullptr);
    std::vector<int64_t> expected_shape = {8, 4};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, input_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
}

TEST_F(GraphTransformationTests, MegatronSelfAttentionPartitionRank0) {
  auto model_uri = MODEL_FOLDER "model_parallel/self_attention_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  training::TrainingSession::OptimizerState init_optim_state;
  IExecutionProvider* e = TestCPUExecutionProvider();
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MegatronTransformer>(0, 2, updated_weight_names, weights_to_train, weight_partition_info,
                                            init_optim_state, *e),
      TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  auto model_uri2 = "self_attention_megatron_basic_test_partition_rank0.onnx";
  ASSERT_STATUS_OK(Model::Save(*p_model, model_uri2));

  {
    std::vector<float> expected_value = {
        0.00f, 0.01f, 0.04f, 0.05f, 0.08f, 0.09f,
        0.12f, 0.13f, 0.16f, 0.17f, 0.20f, 0.21f,
        0.24f, 0.25f, 0.28f, 0.29f, 0.32f, 0.33f,
        0.36f, 0.37f, 0.40f, 0.41f, 0.44f, 0.45f};
    auto qkv_weight_arg = GetNodeByName(graph, "matmul1")->MutableInputDefs()[1];
    ORT_ENFORCE(qkv_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {4, 6};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, qkv_weight_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {0.00f, 0.01f, 0.04f, 0.05f, 0.08f, 0.09f};
    auto qkv_bias_arg = GetNodeByName(graph, "add1")->MutableInputDefs()[1];
    ORT_ENFORCE(qkv_bias_arg != nullptr);
    std::vector<int64_t> expected_shape = {6};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, qkv_bias_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {
        0.00f, 0.01f, 0.02f, 0.03f,
        0.04f, 0.05f, 0.06f, 0.07f};
    auto dense_weight_arg = GetNodeByName(graph, "matmul4")->MutableInputDefs()[1];
    ORT_ENFORCE(dense_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {2, 4};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, dense_weight_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    Node* node = graph.GetNode(GetNodeByName(graph, "matmul1")->InputNodesBegin()->Index());
    ORT_ENFORCE(node->OpType().compare("MegatronF") == 0);
    node = graph.GetNode(GetNodeByName(graph, "add2")->InputNodesBegin()->Index());
    ORT_ENFORCE(node->OpType().compare("MegatronG") == 0);
  }
}

TEST_F(GraphTransformationTests, MegatronSelfAttentionPartitionRank1) {
  auto model_uri = MODEL_FOLDER "model_parallel/self_attention_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  training::TrainingSession::OptimizerState init_optim_state;
  IExecutionProvider* e = TestCPUExecutionProvider();
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<MegatronTransformer>(1, 2, updated_weight_names,
                                                                                           weights_to_train,
                                                                                           weight_partition_info,
                                                                                           init_optim_state, *e),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  auto model_uri2 = "self_attention_megatron_basic_test_partition_rank1.onnx";
  ASSERT_STATUS_OK(Model::Save(*p_model, model_uri2));

  {
    std::vector<float> expected_value = {
        0.02f, 0.03f, 0.06f, 0.07f, 0.10f, 0.11f,
        0.14f, 0.15f, 0.18f, 0.19f, 0.22f, 0.23f,
        0.26f, 0.27f, 0.30f, 0.31f, 0.34f, 0.35f,
        0.38f, 0.39f, 0.42f, 0.43f, 0.46f, 0.47f};
    auto qkv_weight_arg = GetNodeByName(graph, "matmul1")->MutableInputDefs()[1];
    ORT_ENFORCE(qkv_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {4, 6};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, qkv_weight_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {0.02f, 0.03f, 0.06f, 0.07f, 0.10f, 0.11f};
    auto qkv_bias_arg = GetNodeByName(graph, "add1")->MutableInputDefs()[1];
    ORT_ENFORCE(qkv_bias_arg != nullptr);
    std::vector<int64_t> expected_shape = {6};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, qkv_bias_arg, actual_val,
                                                                                    actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {
        0.08f, 0.09f, 0.10f, 0.11f,
        0.12f, 0.13f, 0.14f, 0.15f};
    auto dense_weight_arg = GetNodeByName(graph, "matmul4")->MutableInputDefs()[1];
    ORT_ENFORCE(dense_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {2, 4};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    ASSERT_STATUS_OK(horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, dense_weight_arg,
                                                                                    actual_val, actual_shape));
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    Node* node = graph.GetNode(GetNodeByName(graph, "matmul1")->InputNodesBegin()->Index());
    ORT_ENFORCE(node->OpType().compare("MegatronF") == 0);
    node = graph.GetNode(GetNodeByName(graph, "add2")->InputNodesBegin()->Index());
    ORT_ENFORCE(node->OpType().compare("MegatronG") == 0);
  }
}

TEST_F(GraphTransformationTests, BiasGeluRecomputeTest) {
  auto model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion_recompute.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GeluRecompute>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.BiasGelu"] == 2);
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "com.microsoft.BiasGelu") {
      ASSERT_TRUE(node.InputDefs().size() == 2);
    }
  }
}

TEST_F(GraphTransformationTests, SoftmaxCrossEntropyLossInternalFusionWithoutCast) {
  Model model("SoftmaxCrossEntropyLossInternalFusion", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{"", 12}, {"com.microsoft", 1}}, {}, *logger_);
  auto& graph = model.MainGraph();

  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto tensor_int;
  tensor_int.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  onnxruntime::NodeArg x_def("X", &tensor_float);
  onnxruntime::NodeArg ls_out_def("ls_out", &tensor_float);
  onnxruntime::NodeArg target_def("target", &tensor_int);
  onnxruntime::NodeArg weight_def("weight", &tensor_float);
  onnxruntime::NodeArg ignore_index_def("ignore_index", &tensor_int);
  onnxruntime::NodeArg y_def("Y", &tensor_float);

  graph.AddNode("ls", "LogSoftmax", "LogSoftmax operator", {&x_def}, {&ls_out_def});
  Node& nll_loss_node = graph.AddNode(
      "nl_loss_internal", "NegativeLogLikelihoodLossInternal", "NegativeLogLikelihoodLossInternal operator",
      {&ls_out_def, &target_def, &weight_def, &ignore_index_def}, {&y_def}, nullptr, onnxruntime::kMSDomain);
  nll_loss_node.AddAttribute("reduction", "none");

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<SoftmaxCrossEntropyLossInternalFusion>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["LogSoftmax"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.NegativeLogLikelihoodLossInternal"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
}

TEST_F(GraphTransformationTests, SoftmaxCrossEntropyLossInternalFusionWithCast) {
  Model model("SoftmaxCrossEntropyLossInternalFusion", true, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{"", 12}, {"com.microsoft", 1}}, {}, *logger_);
  auto& graph = model.MainGraph();

  TypeProto tensor_half;
  tensor_half.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  TypeProto tensor_int;
  tensor_int.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  onnxruntime::NodeArg x_def("X", &tensor_half);
  onnxruntime::NodeArg ls_out_def("ls_out", &tensor_half);
  onnxruntime::NodeArg ct_out_def("ct_out", &tensor_float);
  onnxruntime::NodeArg target_def("target", &tensor_int);
  onnxruntime::NodeArg weight_def("weight", &tensor_float);
  onnxruntime::NodeArg ignore_index_def("ignore_index", &tensor_int);
  onnxruntime::NodeArg y_def("Y", &tensor_float);

  graph.AddNode("ls", "LogSoftmax", "LogSoftmax operator", {&x_def}, {&ls_out_def});
  Node& cast_node = graph.AddNode("ct", "Cast", "Cast operator", {&ls_out_def}, {&ct_out_def});
  cast_node.AddAttribute("to", static_cast<int64_t>(1));
  Node& nll_loss_node = graph.AddNode(
      "nl_loss_internal", "NegativeLogLikelihoodLossInternal", "NegativeLogLikelihoodLossInternal operator",
      {&ct_out_def, &target_def, &weight_def, &ignore_index_def}, {&y_def}, nullptr, onnxruntime::kMSDomain);
  nll_loss_node.AddAttribute("reduction", "mean");

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<SoftmaxCrossEntropyLossInternalFusion>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 1);
  ASSERT_TRUE(op_to_count["LogSoftmax"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.NegativeLogLikelihoodLossInternal"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
}

class QDQFusionTestsParameterized : public GraphTransformationTests,
                                    public ::testing::WithParamInterface<std::tuple<PathString>> {
};

TEST_P(QDQFusionTestsParameterized, CheckModelComposition) {
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(std::get<0>(GetParam()), p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::map<std::string, int> op_to_count_pre_fusion = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count_pre_fusion["QuantizeLinear"], 1);
  ASSERT_EQ(op_to_count_pre_fusion["DequantizeLinear"], 1);
  ASSERT_EQ(op_to_count_pre_fusion["com.microsoft.FakeQuant"], 0);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<QDQFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count_post_fusion = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count_post_fusion["QuantizeLinear"], 0);
  ASSERT_EQ(op_to_count_post_fusion["DequantizeLinear"], 0);
  ASSERT_EQ(op_to_count_post_fusion["com.microsoft.FakeQuant"], 1);
}

INSTANTIATE_TEST_SUITE_P(
    QDQFusionTests,
    QDQFusionTestsParameterized,
    ::testing::Values(
        std::make_tuple(MODEL_FOLDER "fusion/qdq_fusion_int8.onnx"),
        std::make_tuple(MODEL_FOLDER "fusion/qdq_fusion_uint8.onnx"),
        std::make_tuple(MODEL_FOLDER "fusion/qdq_fusion_zp_not_provided.onnx")));

// We only tested on CUDA run.
#if defined(USE_CUDA)
static void RunPartitionCorrectnessTest(std::string model_path,
                                        const logging::Logger& logger,
                                        const int total_rank,
                                        std::vector<std::string> input_names,
                                        std::vector<std::vector<int64_t>> input_dims) {
  const PathString model_uri = ToPathString(model_path) + ORT_TSTR(".onnx");
  // const int total_rank = 4;
  std::vector<Graph*> graphs;
  std::vector<std::shared_ptr<Model>> p_models(total_rank);
  for (auto i = 0; i < total_rank; i++) {
    Status ret = Model::Load(model_uri, p_models[i], nullptr, logger);
    ORT_ENFORCE(ret.IsOK());
    Graph& graph = p_models[i]->MainGraph();
    onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
    std::unordered_map<std::string, std::string> updated_weight_names;
    std::unordered_set<std::string> weights_to_train;
    std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
    training::TrainingSession::OptimizerState init_optim_state;
    IExecutionProvider* e = TestCPUExecutionProvider();
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(
        std::make_unique<MegatronTransformer>(i, total_rank, updated_weight_names, weights_to_train,
                                              weight_partition_info, init_optim_state, *e),
        TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));
    graphs.push_back(&graph);
    auto model_uri2 = ToPathString(model_path) + ORT_TSTR("_partition_rank_") + ToPathString(std::to_string(i)) + ORT_TSTR(".onnx");
    ASSERT_STATUS_OK(Model::Save(*p_models[i], model_uri2));
  }

  onnxruntime::Model combine_model("combine_graph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}, {kMSDomain, 1}}, {}, logger);
  auto& combine_graph = combine_model.MainGraph();
  auto ret = horizontal_parallel_test_utils::MergeGraphsOnAllWorkers(graphs, combine_graph);
  ORT_ENFORCE(ret.IsOK());
  auto model_uri2 = ToPathString(model_path) + ORT_TSTR("_partition_combine.onnx");
  ASSERT_STATUS_OK(Model::Save(combine_model, model_uri2));

  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};

  ORT_ENFORCE(input_names.size() == input_dims.size());
  NameMLValMap feeds;
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::vector<int64_t> dims_X = input_dims[i];
    std::vector<float> values_X(TensorShape(dims_X).Size());
    std::for_each(values_X.begin(), values_X.end(),
                  [&generator, &distribution](float& value) { value = distribution(generator); });

    OrtValue ml_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_X, values_X, &ml_value);
    feeds.insert(std::make_pair(input_names[i], ml_value));
  }

  std::vector<OrtValue> expected_ort_values;
  {
    SessionOptions so;
    so.session_logid = "RawGraphRun";

    InferenceSession session_object{so, GetEnvironment()};
    std::unique_ptr<IExecutionProvider> execution_provider = DefaultCudaExecutionProvider();
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

    Status st;
    ASSERT_TRUE((st = session_object.Load(model_uri)).IsOK()) << st;
    ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st;

    // prepare outputs
    std::vector<std::string> output_names;
    output_names.push_back("output");

    // Now run
    RunOptions run_options;
    run_options.training_mode = true;
    st = session_object.Run(run_options, feeds, output_names, &expected_ort_values);

    EXPECT_TRUE(st.IsOK());
  }

  std::vector<OrtValue> actual_ort_values;
  {
    SessionOptions so;
    so.session_logid = "SplitThenCombineRun";

    InferenceSession session_object{so, GetEnvironment()};
    std::unique_ptr<IExecutionProvider> execution_provider = DefaultCudaExecutionProvider();
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

    Status st;
    ASSERT_TRUE((st = session_object.Load(model_uri2)).IsOK()) << st;
    ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st;

    // prepare outputs
    std::vector<std::string> output_names;
    for (auto i = 0; i < total_rank; i++) {
      output_names.push_back("output_rank_" + std::to_string(i));
    }

    // Now run
    RunOptions run_options;
    run_options.training_mode = true;
    st = session_object.Run(run_options, feeds, output_names, &actual_ort_values);

    EXPECT_TRUE(st.IsOK());
  }

  auto& expected_val = expected_ort_values[0].Get<Tensor>();
  for (auto i = 0; i < total_rank; i++) {
    auto& actual_val = actual_ort_values[i].Get<Tensor>();
    horizontal_parallel_test_utils::VerifyOutputs(expected_val, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_val, actual_val, false);
  }
}

TEST_F(GraphTransformationTests, MegatronMLPPartitionCorrectnessTest) {
  RunPartitionCorrectnessTest("testdata/transform/model_parallel/mlp_megatron_basic_test", *logger_, 4, {"input"}, {{8, 16, 4}});
}

TEST_F(GraphTransformationTests, MegatronBARTMLPPartitionCorrectnessTest) {
  RunPartitionCorrectnessTest("testdata/transform/model_parallel/bart_mlp_megatron_basic_test", *logger_, 4, {"input"}, {{8, 16, 4}});
}

TEST_F(GraphTransformationTests, MegatronSelfAttentionPartitionCorrectnessTest) {
  RunPartitionCorrectnessTest("testdata/transform/model_parallel/self_attention_megatron_basic_test", *logger_, 2, {"input", "mask"}, {{8, 16, 4}, {8, 1, 16, 16}});
}

TEST_F(GraphTransformationTests, MegatronBARTSelfAttentionPartitionCorrectnessTest) {
  RunPartitionCorrectnessTest("testdata/transform/model_parallel/bart_self_attention_megatron_basic_test", *logger_, 2, {"input"}, {{6, 8, 4}});
}
// end of USE_CUDA
#endif

// end of DISABLE_CONTRIB_OPS
#endif

}  // namespace test
}  // namespace onnxruntime
