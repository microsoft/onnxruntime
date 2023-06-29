// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Only enabled in training full build, not in on device training build.
#ifdef ENABLE_TRAINING
#include <random>
#include "core/graph/onnx_protobuf.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "asserts.h"
#include "core/common/span_utils.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/utils.h"
#include "core/util/math.h"
#include "orttraining/core/optimizer/compute_optimizer/sceloss_compute_optimization.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/compare_ortvalue.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/optimizer/test_optimizer_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/temp_dir.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

namespace {
const InlinedHashSet<std::string_view> compatible_eps = {};
}

/*
Test graph includes multiple equivalent subgraphs as below.
           graph input [32, 256] (float)                graph input [32] (int64_t)
                            |                                   |
                             \_____________             _______/     graph input -1, scalar (int64_t)
                                           \           /        _______/
                                            \         /        /
                                  SCE Node, reduction = 'mean', output_type=1
                                            |
                                            |
                                    graph output, loss, scalar (float)
*/
TEST(ComputeOptimizerTests, InsertGatherBeforeSceLoss_Allowed) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();

  for (const bool is_sce_internal : {true, false}) {
    auto pre_graph_checker = [is_sce_internal](Graph& graph) -> Status {
      auto op_count_pre = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_pre.size() == 1U);

      if (is_sce_internal)
        TEST_RETURN_IF_NOT(op_count_pre["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
      else
        TEST_RETURN_IF_NOT(op_count_pre["SoftmaxCrossEntropyLoss"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [is_sce_internal](Graph& graph) {
      auto op_count_post = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_post.size() == 5U);
      TEST_RETURN_IF_NOT(op_count_post["Sub"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["NonZero"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["Squeeze"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["com.microsoft.ShrunkenGather"] == 2);

      if (is_sce_internal)
        TEST_RETURN_IF_NOT(op_count_post["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
      else
        TEST_RETURN_IF_NOT(op_count_post["SoftmaxCrossEntropyLoss"] == 1);

      const NodeArg* squeeze_output_arg = nullptr;
      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Squeeze") {
          squeeze_output_arg = node.OutputDefs()[0];
          break;
        }
      }

      TEST_RETURN_IF_NOT(squeeze_output_arg != nullptr);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "SoftmaxCrossEntropyLossInternal" || node.OpType() == "SoftmaxCrossEntropyLoss") {
          const auto& input_defs = node.InputDefs();

          {
            auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
            TEST_RETURN_IF_NOT(producer_node != nullptr);
            TEST_RETURN_IF_NOT(producer_node->OpType() == "ShrunkenGather");
            TEST_RETURN_IF_NOT(producer_node->InputDefs()[1] == squeeze_output_arg);
          }

          {
            auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
            TEST_RETURN_IF_NOT(producer_node != nullptr);
            TEST_RETURN_IF_NOT(producer_node->OpType() == "ShrunkenGather");
            TEST_RETURN_IF_NOT(producer_node->InputDefs()[1] == squeeze_output_arg);
          }
        }
      }
      return Status::OK();
    };

    auto build_test_case = [is_sce_internal](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{32, 256}});
      auto* input2_arg = builder.MakeInput<int64_t>({{32}}, "label");
      auto* sce_out1 = builder.MakeOutput();
      NodeArg* empty = builder.MakeEmptyInput();
      auto* sce_out2 = builder.MakeIntermediate();

      if (is_sce_internal) {
        auto* ignore_index_arg = builder.MakeScalarInitializer<int64_t>(-100);
        Node& sce = builder.AddNode("SoftmaxCrossEntropyLossInternal",
                                    {input1_arg, input2_arg, empty, ignore_index_arg},
                                    {sce_out1, sce_out2}, kMSDomain);
        sce.AddAttribute("reduction", "mean");
        sce.AddAttribute("output_type", static_cast<int64_t>(1));
      } else {
        Node& sce = builder.AddNode("SoftmaxCrossEntropyLoss",
                                    {input1_arg, input2_arg, empty},
                                    {sce_out1, sce_out2});
        sce.AddAttribute("reduction", "mean");
        sce.AddAttribute("ignore_index", static_cast<int64_t>(-100));
      }
    };

    std::vector<int> opsets{12, 13, 14, 15};
    for (auto opset : opsets) {
      std::unique_ptr<GraphTransformer> transformer =
          std::make_unique<InsertGatherBeforeSceLoss>(compatible_eps, std::vector<std::string>{"label"});
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                            TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

/*
Test graph includes multiple equivalent subgraphs as below.
           graph input [32, 256] (float)                graph input [32] (int64_t)
                            |                                   |
                             \_____________             _______/     graph input -1, scalar (int64_t)
                                           \           /        _______/
                                            \         /        /
                                  SCE Node, reduction = 'mean', output_type=1
                                            |
                                            |
                                    graph output, loss, scalar (float)
*/
TEST(ComputeOptimizerTests, InsertGatherBeforeSceLoss_NotAllowed_LabelNameNotMatch) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();

  for (const bool is_sce_internal : {true, false}) {
    auto pre_graph_checker = [is_sce_internal](Graph& graph) -> Status {
      auto op_count_pre = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_pre.size() == 1U);
      if (is_sce_internal)
        TEST_RETURN_IF_NOT(op_count_pre["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
      else
        TEST_RETURN_IF_NOT(op_count_pre["SoftmaxCrossEntropyLoss"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [is_sce_internal](Graph& graph) {
      auto op_count_post = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_post.size() == 1U);
      if (is_sce_internal)
        TEST_RETURN_IF_NOT(op_count_post["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
      else
        TEST_RETURN_IF_NOT(op_count_post["SoftmaxCrossEntropyLoss"] == 1);
      return Status::OK();
    };

    auto build_test_case = [is_sce_internal](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{32, 256}});
      auto* input2_arg = builder.MakeInput<int64_t>({{32}}, "label111");
      auto* sce_out1 = builder.MakeOutput();

      NodeArg* empty = builder.MakeEmptyInput();
      auto* sce_out2 = builder.MakeIntermediate();

      if (is_sce_internal) {
        auto* ignore_index_arg = builder.MakeScalarInitializer<int64_t>(-100);
        Node& sce = builder.AddNode("SoftmaxCrossEntropyLossInternal",
                                    {input1_arg, input2_arg, empty, ignore_index_arg},
                                    {sce_out1, sce_out2}, kMSDomain);
        sce.AddAttribute("reduction", "mean");
        sce.AddAttribute("output_type", static_cast<int64_t>(1));
      } else {
        Node& sce = builder.AddNode("SoftmaxCrossEntropyLoss",
                                    {input1_arg, input2_arg, empty},
                                    {sce_out1, sce_out2});
        sce.AddAttribute("reduction", "mean");
        sce.AddAttribute("ignore_index", static_cast<int64_t>(-100));
      }
    };

    std::vector<int> opsets{12, 13, 14, 15};
    for (auto opset : opsets) {
      std::unique_ptr<GraphTransformer> transformer =
          std::make_unique<InsertGatherBeforeSceLoss>(compatible_eps, std::vector<std::string>{"label"});
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                            TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

/*
Test graph includes multiple equivalent subgraphs as below.
           graph input [32, 256] (float)                graph input [32] (int64_t)
                            |                                   |
                             \_____________             _______/     graph input -1, scalar (int64_t)
                                           \           /        _______/
                                            \         /        /
                             SCE Node, reduction = 'none', output_type=1
                                            |
                                            |
                            graph output, loss, [32] (float)
*/
TEST(ComputeOptimizerTests, InsertGatherBeforeSceLoss_NotAllowed_ReduceNone) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();

  for (const bool is_sce_internal : {true, false}) {
    auto pre_graph_checker = [is_sce_internal](Graph& graph) -> Status {
      auto op_count_pre = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_pre.size() == 1U);
      if (is_sce_internal)
        TEST_RETURN_IF_NOT(op_count_pre["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
      else
        TEST_RETURN_IF_NOT(op_count_pre["SoftmaxCrossEntropyLoss"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [is_sce_internal](Graph& graph) {
      auto op_count_post = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_post.size() == 1U);
      if (is_sce_internal)
        TEST_RETURN_IF_NOT(op_count_post["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
      else
        TEST_RETURN_IF_NOT(op_count_post["SoftmaxCrossEntropyLoss"] == 1);
      return Status::OK();
    };

    auto build_test_case = [is_sce_internal](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{32, 256}});
      auto* input2_arg = builder.MakeInput<int64_t>({{32}}, "label");
      auto* sce_out1 = builder.MakeOutput();

      NodeArg* empty = builder.MakeEmptyInput();
      auto* sce_out2 = builder.MakeIntermediate();

      if (is_sce_internal) {
        auto* ignore_index_arg = builder.MakeScalarInitializer<int64_t>(-100);
        Node& sce = builder.AddNode("SoftmaxCrossEntropyLossInternal",
                                    {input1_arg, input2_arg, empty, ignore_index_arg},
                                    {sce_out1, sce_out2}, kMSDomain);
        sce.AddAttribute("reduction", "none");
        sce.AddAttribute("output_type", static_cast<int64_t>(1));
      } else {
        Node& sce = builder.AddNode("SoftmaxCrossEntropyLoss",
                                    {input1_arg, input2_arg, empty},
                                    {sce_out1, sce_out2});
        sce.AddAttribute("reduction", "none");
        sce.AddAttribute("ignore_index", static_cast<int64_t>(-100));
      }
    };

    std::vector<int> opsets{12, 13, 14, 15};
    for (auto opset : opsets) {
      std::unique_ptr<GraphTransformer> transformer =
          std::make_unique<InsertGatherBeforeSceLoss>(compatible_eps, std::vector<std::string>{"label"});
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                            TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [32, 256] (float)                graph input [32] (int64_t)
                            |                                   |
                             \_____________             _______/     graph input -1, scalar (int64_t)
                                           \           /        _______/
                                            \         /        /
                            SCE Node, reduction = 'none', output_type=1
                                            |
                                            |
               graph output, loss, scalar (float)
*/
TEST(ComputeOptimizerTests, InsertGatherBeforeSceLoss_NotAllowed_NoIgnoreIndex) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();

  for (const bool is_sce_internal : {true, false}) {
    auto pre_graph_checker = [is_sce_internal](Graph& graph) -> Status {
      auto op_count_pre = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_pre.size() == 1U);
      if (is_sce_internal)
        TEST_RETURN_IF_NOT(op_count_pre["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
      else
        TEST_RETURN_IF_NOT(op_count_pre["SoftmaxCrossEntropyLoss"] == 1);
      return Status::OK();
    };

    auto post_graph_checker = [is_sce_internal](Graph& graph) {
      auto op_count_post = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_post.size() == 1U);
      if (is_sce_internal)
        TEST_RETURN_IF_NOT(op_count_post["com.microsoft.SoftmaxCrossEntropyLossInternal"] == 1);
      else
        TEST_RETURN_IF_NOT(op_count_post["SoftmaxCrossEntropyLoss"] == 1);
      return Status::OK();
    };

    auto build_test_case = [is_sce_internal](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{32, 256}});
      auto* input2_arg = builder.MakeInput<int64_t>({{32}}, "label");
      auto* sce_out1 = builder.MakeOutput();
      auto* sce_out2 = builder.MakeIntermediate();

      if (is_sce_internal) {
        Node& sce = builder.AddNode("SoftmaxCrossEntropyLossInternal",
                                    {input1_arg, input2_arg},
                                    {sce_out1, sce_out2}, kMSDomain);
        sce.AddAttribute("reduction", "sum");
        sce.AddAttribute("output_type", static_cast<int64_t>(1));
      } else {
        Node& sce = builder.AddNode("SoftmaxCrossEntropyLoss",
                                    {input1_arg, input2_arg},
                                    {sce_out1, sce_out2});
        sce.AddAttribute("reduction", "sum");
      }
    };

    std::vector<int> opsets{12, 13, 14, 15};
    for (auto opset : opsets) {
      std::unique_ptr<GraphTransformer> transformer =
          std::make_unique<InsertGatherBeforeSceLoss>(compatible_eps, std::vector<std::string>{"label"});
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset, *logger, std::move(transformer),
                                            TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

TEST(ComputeOptimizerTests, InsertGatherBeforeSceLoss_MlmBertE2E) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  // Be noted all dropout have a ratio be 0.0, to make it easier to compare when running with the session.
  // This did not affect the transformer tests, because we did not remove the Dropout of ratio 0. in the middle.
  auto model_uri = MODEL_FOLDER "computation_reduction/reshape/mlm_bert_e2e.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{3};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<InsertGatherBeforeSceLoss>(compatible_eps, std::vector<std::string>{"labels"}),
      TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  {
    const NodeArg* squeeze_output_arg = nullptr;
    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "NonZero") {
        const std::vector<const Node*>& consumers = graph.GetConsumerNodes(node.OutputDefs()[0]->Name());
        ASSERT_TRUE(consumers.size() == 1);
        ASSERT_TRUE(consumers[0]->OpType() == "Squeeze");
        squeeze_output_arg = consumers[0]->OutputDefs()[0];
        break;
      }
    }

    ASSERT_TRUE(squeeze_output_arg != nullptr);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "SoftmaxCrossEntropyLossInternal") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          ASSERT_TRUE(producer_node != nullptr);
          ASSERT_TRUE(producer_node->OpType() == "ShrunkenGather");
          ASSERT_TRUE(producer_node->InputDefs()[1] == squeeze_output_arg);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          ASSERT_TRUE(producer_node != nullptr);
          ASSERT_TRUE(producer_node->OpType() == "ShrunkenGather");
          ASSERT_TRUE(producer_node->InputDefs()[1] == squeeze_output_arg);
        }
      }
    }
  }

  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent<PathChar>(
      tmp_dir.Path(),
      ORT_TSTR("insert_gather_before_sceloss_bert_e2e_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  const TensorShapeVector dims_mask = {batch_size, sequence_length};
  std::vector<int64_t> attention_mask(TensorShape(dims_mask).Size(), 0);
  RandomMasks(batch_size, sequence_length, attention_mask);
  input_container.AddInput<int64_t>("attention_mask", dims_mask, attention_mask);

  input_container.AddInput<MLFloat16>("matmul1.weight", {hidden_size, 1024}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add1.bias", {1024}, RandomFillHalfVector);

  input_container.AddInput<MLFloat16>("matmul2.weight", {hidden_size, 1024}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add2.bias", {1024}, RandomFillHalfVector);

  input_container.AddInput<MLFloat16>("matmul3.weight", {hidden_size, 1024}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add3.bias", {1024}, RandomFillHalfVector);

  input_container.AddInput<MLFloat16>("matmul4.weight", {hidden_size, 1024}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add4.bias", {1024}, RandomFillHalfVector);

  input_container.AddInput<float>("layer_norm1.weight", {hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("layer_norm1.bias", {hidden_size}, RandomFillFloatVector);

  input_container.AddInput<MLFloat16>("matmul7.weight", {hidden_size, hidden_size * 4}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add7.bias", {hidden_size * 4}, RandomFillHalfVector);

  input_container.AddInput<MLFloat16>("matmul8.weight", {hidden_size * 4, hidden_size}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add8.bias", {hidden_size}, RandomFillHalfVector);

  input_container.AddInput<float>("layer_norm2.weight", {hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("layer_norm2.bias", {hidden_size}, RandomFillFloatVector);

  input_container.AddInput<MLFloat16>("matmul9.weight", {hidden_size, hidden_size}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add9.bias", {hidden_size}, RandomFillHalfVector);

  input_container.AddInput<float>("layer_norm3.weight", {hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("layer_norm3.bias", {hidden_size}, RandomFillFloatVector);

  input_container.AddInput<MLFloat16>("matmul10.weight", {hidden_size, 30522}, RandomFillHalfVector);
  input_container.AddInput<MLFloat16>("add10.bias", {30522}, RandomFillHalfVector);

  const TensorShapeVector dims_labels = {batch_size * sequence_length};
  static RandomValueGenerator random{8910};
  std::vector<int64_t> labels = random.Uniform<int64_t>(dims_labels, 0, 30522);
  const std::vector<int64_t> num_count_to_random{batch_size};
  std::vector<int64_t> random_seq_lens = random.Uniform<int64_t>(num_count_to_random, 0, sequence_length);
  for (int64_t i = 0; i < batch_size; ++i) {
    for (int64_t j = 0; j < sequence_length; ++j) {
      if (j > random_seq_lens[i]) {
        labels[i * sequence_length + j] = -100;
      }
    }
  }

  input_container.AddInput<int64_t>("labels", dims_labels, labels);

  const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"output-1"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());

    constexpr double per_sample_tolerance = 1e-4;
    constexpr double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
