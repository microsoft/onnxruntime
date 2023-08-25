// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <random>
#include "core/graph/onnx_protobuf.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "asserts.h"
#include "core/common/span_utils.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/common_subexpression_elimination.h"
#include "core/optimizer/compute_optimizer/upstream_gather.h"
#include "core/optimizer/compute_optimizer/upstream_reshape.h"
#include "core/optimizer/utils.h"
#include "core/util/math.h"

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

// LayerNormalization/Gelu implementation are in contrib namespace (OnnxDomain 1), so
// Without contib_ops enabled, we cannot parse the graph correctly.
#ifndef DISABLE_CONTRIB_OPS
static void GatherNDComputationReductionTest(const std::string& op_type,
                                             const logging::Logger& logger,
                                             std::function<void(Graph&, std::string op_type)> validation_func) {
  std::string op_type_lower = op_type;
  std::transform(op_type_lower.begin(), op_type_lower.end(), op_type_lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::string file_path = std::string("testdata/transform/computation_reduction/gathernd/gathernd_") + op_type_lower +
                          std::string(".onnx");
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(file_path), model, nullptr, logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

  validation_func(graph, op_type);
}

void SingleOpDefaultValidationFunc(Graph& graph, std::string op_type) {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  Node* gathernd_node = nullptr;
  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    ASSERT_FALSE(p_node == nullptr);
    if (p_node->OpType().compare("GatherND") == 0) {
      gathernd_node = p_node;
      EXPECT_EQ(gathernd_node->MutableInputDefs()[0]->Name(), "input");
      const auto& consumers = graph.GetConsumerNodes(gathernd_node->MutableOutputDefs()[0]->Name());
      EXPECT_EQ(consumers[0]->OpType(), op_type);
    }
  }

  ASSERT_FALSE(gathernd_node == nullptr);
}

TEST(ComputeOptimizerTests, GatherND_Gelu) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  GatherNDComputationReductionTest("Gelu", *logger, SingleOpDefaultValidationFunc);
}

TEST(ComputeOptimizerTests, GatherND_Add) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  GatherNDComputationReductionTest("Add", *logger, [](Graph& graph, std::string op_type) -> void {
    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    Node* gathernd_node = nullptr;
    bool found_gathernd_around_graph_output = false;
    for (auto node_index : node_topology_list) {
      Node* p_node = graph.GetNode(node_index);
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("GatherND") == 0) {
        if (p_node->OutputDefs()[0]->Name().compare("output") != 0) {
          gathernd_node = p_node;
          EXPECT_EQ(gathernd_node->MutableInputDefs()[0]->Name(), "input");
          const auto& consumers = graph.GetConsumerNodes(gathernd_node->MutableOutputDefs()[0]->Name());
          EXPECT_EQ(consumers[0]->OpType(), op_type);
        } else {
          found_gathernd_around_graph_output = true;
        }
      }
    }
    ASSERT_FALSE(gathernd_node == nullptr);
    EXPECT_TRUE(found_gathernd_around_graph_output); });
}

TEST(ComputeOptimizerTests, GatherND_LayerNormalization) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  GatherNDComputationReductionTest("LayerNormalization", *logger, SingleOpDefaultValidationFunc);
}

TEST(ComputeOptimizerTests, GatherND_MatMul) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  GatherNDComputationReductionTest("MatMul", *logger, SingleOpDefaultValidationFunc);
}

TEST(ComputeOptimizerTests, GatherND_E2E) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gathernd/e2e.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  // check the expected node orders.
  {
    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    Node* gathernd_node = nullptr;
    for (auto node_index : node_topology_list) {
      Node* p_node = graph.GetNode(node_index);
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("GatherND") == 0) {
        gathernd_node = p_node;
        const Node* layer_norm_node = graph.GetProducerNode(gathernd_node->MutableInputDefs()[0]->Name());
        EXPECT_EQ(layer_norm_node->OpType(), "LayerNormalization");
        EXPECT_EQ(layer_norm_node->Name(), "layer_norm_1");
        const auto& consumers = graph.GetConsumerNodes(gathernd_node->MutableOutputDefs()[0]->Name());
        EXPECT_EQ(consumers[0]->OpType(), "MatMul");
        EXPECT_EQ(consumers[0]->Name(), "matmul_1");
        break;
      }
    }

    ASSERT_FALSE(gathernd_node == nullptr);
  }

  // check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("computation_reduction_transformer_after.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  InputContainer input_container;

  int batch_size = 8;
  int sequence = 128;
  int hidden_size = 128;
  int dynamic_predict_count = 20;
  input_container.AddInput<float>("input", {batch_size, sequence, hidden_size}, RandomFillFloatVector);

  const TensorShapeVector dims_unsqueezed_masked_lm_positions{batch_size, dynamic_predict_count, 1};
  std::vector<int64_t> values_unsqueezed_masked_lm_positions(TensorShape(dims_unsqueezed_masked_lm_positions).Size());

  std::random_device rd;                                   // obtain a random number from hardware
  std::mt19937 eng(rd());                                  // seed the generator
  std::uniform_int_distribution<> distr(0, sequence - 1);  // define the range
  std::for_each(values_unsqueezed_masked_lm_positions.begin(), values_unsqueezed_masked_lm_positions.end(),
                [&distr, &eng](int64_t& value) { value = distr(eng); });

  input_container.AddInput<int64_t>("unsqueezed_masked_lm_positions",
                                    dims_unsqueezed_masked_lm_positions,
                                    values_unsqueezed_masked_lm_positions);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };
  const std::vector<std::string> output_names{"output", "gather_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"), provider_type,
                     input_container, output_names, actual_ort_values);

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

TEST(ComputeOptimizerTests, GatherMatMul_ScalarSlicingOnBatchDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_scalar_batch_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Unsqueeze");

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Unsqueeze");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_matmul_scalar_batch_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherMatMul_SlicingOnBatchDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_batch_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Gather");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_matmul_batch_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherMatMul_ScalarSlicingOnLastDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_scalar_last_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first branch.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "MatMul");
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 2);
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_TRUE(lhs_input == nullptr);

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Unsqueeze");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_matmul_scalar_last_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherMatMul_SlicingOnLastDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_last_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first branch.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "MatMul");
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 2);
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_TRUE(lhs_input == nullptr);

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Gather");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_matmul_last_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherMatMul_ScalarSlicingOnSecondLastDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_scalar_second_last_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  // Check the second branch.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "MatMul");
  }

  // Check MatMul(who gathers on the second last dim)'s input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Unsqueeze");

    ASSERT_TRUE(rhs_input == nullptr);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(
      tmp_dir.Path(),
      ORT_TSTR("gather_matmul_scalar_second_last_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherMatMul_SlicingOnSecondLastDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_matmul_second_last_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  // Check the second branch.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input2");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "MatMul");
  }

  // Check MatMul's input and output.
  {
    const Node* m5 = graph.GetProducerNode("m1_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m1");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_matmul_second_last_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);
  input_container.AddInput<float>("input2", {batch_size, hidden_size, sequence_length}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

/*
Test graph includes multiple equivalent subgraphs as below.
             graph input [2, 32, 256] (float)
                            |
                  LayerNormalization[axis=-1 (as example)]
                            |
                      [2, 32, 256]
                            |
                            |     0 (scalar)
                            |    /
                       Gather[axis=1]
                            |
                        Identity
                            |
              graph output [2, 256] (float)

Add an Identity node because currently, we don't allow Gather generates graph output.
*/
TEST(ComputeOptimizerTests, GatherLayerNormalization) {
  std::vector<std::tuple<int, int64_t, int64_t, bool>> test_config_pairs{
      // {
      //  is_scalar_slice,
      //  ln_axis_before_propagation,
      //  expected_ln_axis_after_propagation,
      //  expected to propagate
      // }
      {true, 0, 0, false},
      {true, 1, 1, false},
      {true, 2, 1, true},
      {true, -3, -3, false},
      {true, -2, -2, false},
      {true, -1, 1, true},
      {false, 0, 0, false},
      {false, 1, 1, false},
      {false, 2, 2, true},
      {false, -3, -3, false},
      {false, -2, -2, false},
      {false, -1, -1, true},
  };

  constexpr static int64_t gather_axis = 1;
  constexpr static int64_t slice_data_value = 0;

  for (auto p : test_config_pairs) {
    bool is_scalar_slice = std::get<0>(p);
    int64_t ln_axis_before = std::get<1>(p);
    int64_t ln_axis_after = std::get<2>(p);
    bool expected_to_propagate = std::get<3>(p);

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();

    InlinedVector<int64_t> indices;
    auto pre_graph_checker = [&indices](Graph& graph) -> Status {
      auto op_count_pre = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
      TEST_RETURN_IF_NOT(op_count_pre["LayerNormalization"] == 1);
      TEST_RETURN_IF_NOT(op_count_pre["Gather"] == 1);
      TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Gather") {
          TEST_RETURN_IF_NOT(indices.empty());
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(node.InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg,
                                                                          indices, require_constant));
        }
      }
      return Status::OK();
    };

    auto post_graph_checker = [is_scalar_slice, ln_axis_after,
                               &indices, expected_to_propagate](Graph& graph) {
      auto op_count_post = CountOpsInGraph(graph);

      TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
      TEST_RETURN_IF_NOT(op_count_post["LayerNormalization"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["Gather"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "LayerNormalization") {
          const auto& input_defs = node.InputDefs();

          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          if (expected_to_propagate) {
            TEST_RETURN_IF_NOT(producer_node != nullptr);
            TEST_RETURN_IF_NOT(producer_node->OpType() == "Gather");

            InlinedVector<int64_t> values;
            constexpr bool require_constant = true;
            NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
            TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg,
                                                                            values, require_constant));
            for (size_t i = 0; i < values.size(); i++) {
              TEST_RETURN_IF_NOT(values[i] == indices[i]);
            }

            const ONNX_NAMESPACE::TensorShapeProto* slice_out_shape = producer_node->OutputDefs()[0]->Shape();
            TEST_RETURN_IF_NOT(slice_out_shape != nullptr);

            auto& attrs = node.GetAttributes();
            TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());

            auto& axis_attr = attrs.at("axis");
            auto axis_value = (int)axis_attr.i();
            TEST_RETURN_IF_NOT(axis_value == ln_axis_after);

            if (is_scalar_slice) {
              TEST_RETURN_IF_NOT(slice_out_shape->dim_size() == 2);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(0)) &&
                                 slice_out_shape->dim(0).dim_value() == 2);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(1)) &&
                                 slice_out_shape->dim(1).dim_value() == 256);
            } else {
              TEST_RETURN_IF_NOT(slice_out_shape->dim_size() == 3);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(0)) &&
                                 slice_out_shape->dim(0).dim_value() == 2);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(1)) &&
                                 slice_out_shape->dim(1).dim_value() == 1);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(2)) &&
                                 slice_out_shape->dim(2).dim_value() == 256);
            }

          } else {
            TEST_RETURN_IF_NOT(producer_node == nullptr);
          }
        }
      }

      return Status::OK();
    };

    auto build_test_case = [is_scalar_slice, ln_axis_before](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 32, 256}});
      auto* input2_arg = builder.MakeInput<float>({{256}});
      auto* input3_arg = builder.MakeInput<float>({{256}});
      auto* ln_out = builder.MakeIntermediate();
      builder.AddNode("LayerNormalization", {input1_arg, input2_arg, input3_arg}, {ln_out})
          .AddAttribute("axis", ln_axis_before);

      std::vector<NodeArg*> slice_inputs;
      NodeArg* indices_initializer = nullptr;

      if (is_scalar_slice) {
        indices_initializer = builder.MakeScalarInitializer<int64_t>(slice_data_value);
      } else {
        indices_initializer = builder.MakeInitializer<int64_t>({1}, {slice_data_value});
      }

      slice_inputs = {ln_out, indices_initializer};

      auto* gather_out = builder.MakeIntermediate();
      builder.AddNode("Gather", slice_inputs,
                      {gather_out})
          .AddAttribute("axis", gather_axis);

      auto* identity_out = builder.MakeOutput();
      builder.AddNode("Identity", {gather_out}, {identity_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamGatherGraphTransformer>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger, std::move(transformer),
                                          TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph includes multiple equivalent subgraphs as below.
             graph input [2, 4, 32, 256] (float)
                            |
                   Softmax[axis=3 (as example)]
                            |
                      [2, 4, 32, 256]
                            |
                            |     0 (scalar)
                            |    /
                       Gather[axis=1]
                            |
                        Identity
                            |
              graph output [2, 32, 256] (float)

Add an Identity node because currently, we don't allow Gather generates graph output.
*/
TEST(ComputeOptimizerTests, GatherSoftmax) {
  std::vector<std::tuple<int, int64_t, int64_t, bool>> test_config_pairs{
      // {is_scalar_slice, softmax_axis_before_propagation,
      //  expected_softmax_axis_after_propagation, expected to propagate}
      {true, 0, 0, false},
      {true, 1, 1, false},
      {true, 2, 1, true},
      {true, 3, 2, true},
      {true, -4, -4, false},
      {true, -3, -3, false},
      {true, -2, 1, true},
      {true, -1, 2, true},
      {false, 0, 0, false},
      {false, 1, 1, false},
      {false, 2, 2, true},
      {false, 3, 3, true},
      {false, -4, -4, false},
      {false, -3, -3, false},
      {false, -2, -2, true},
      {false, -1, -1, true},
  };

  constexpr static int64_t gather_axis = 1;
  constexpr static int64_t slice_data_value = 0;

  for (auto p : test_config_pairs) {
    bool is_scalar_slice = std::get<0>(p);
    int64_t softmax_axis_before = std::get<1>(p);
    int64_t softmax_axis_after = std::get<2>(p);
    bool expected_to_propagate = std::get<3>(p);

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();

    InlinedVector<int64_t> indices;
    auto pre_graph_checker = [&indices](Graph& graph) -> Status {
      auto op_count_pre = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
      TEST_RETURN_IF_NOT(op_count_pre["Softmax"] == 1);
      TEST_RETURN_IF_NOT(op_count_pre["Gather"] == 1);
      TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Gather") {
          TEST_RETURN_IF_NOT(indices.empty());
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(node.InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg,
                                                                          indices, require_constant));
        }
      }
      return Status::OK();
    };

    auto post_graph_checker = [is_scalar_slice, softmax_axis_after,
                               &indices, expected_to_propagate](Graph& graph) {
      auto op_count_post = CountOpsInGraph(graph);

      TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
      TEST_RETURN_IF_NOT(op_count_post["Softmax"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["Gather"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Softmax") {
          const auto& input_defs = node.InputDefs();

          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          if (expected_to_propagate) {
            TEST_RETURN_IF_NOT(producer_node != nullptr);
            TEST_RETURN_IF_NOT(producer_node->OpType() == "Gather");

            InlinedVector<int64_t> values;
            constexpr bool require_constant = true;
            NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
            TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values,
                                                                            require_constant));
            for (size_t i = 0; i < values.size(); i++) {
              TEST_RETURN_IF_NOT(values[i] == indices[i]);
            }

            const ONNX_NAMESPACE::TensorShapeProto* slice_out_shape = producer_node->OutputDefs()[0]->Shape();
            TEST_RETURN_IF_NOT(slice_out_shape != nullptr);

            auto& attrs = node.GetAttributes();
            TEST_RETURN_IF_NOT(attrs.find("axis") != attrs.end());

            auto& axis_attr = attrs.at("axis");
            auto axis_value = (int)axis_attr.i();
            TEST_RETURN_IF_NOT(axis_value == softmax_axis_after);

            if (is_scalar_slice) {
              TEST_RETURN_IF_NOT(slice_out_shape->dim_size() == 3);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(0)) &&
                                 slice_out_shape->dim(0).dim_value() == 2);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(1)) &&
                                 slice_out_shape->dim(1).dim_value() == 32);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(2)) &&
                                 slice_out_shape->dim(2).dim_value() == 256);
            } else {
              TEST_RETURN_IF_NOT(slice_out_shape->dim_size() == 4);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(0)) &&
                                 slice_out_shape->dim(0).dim_value() == 2);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(1)) &&
                                 slice_out_shape->dim(1).dim_value() == 1);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(2)) &&
                                 slice_out_shape->dim(2).dim_value() == 32);
              TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(3)) &&
                                 slice_out_shape->dim(3).dim_value() == 256);
            }

          } else {
            TEST_RETURN_IF_NOT(producer_node == nullptr);
          }
        }
      }

      return Status::OK();
    };

    auto build_test_case = [is_scalar_slice, softmax_axis_before](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{2, 4, 32, 256}});
      auto* softmax_out = builder.MakeIntermediate();
      builder.AddNode("Softmax", {input1_arg}, {softmax_out})
          .AddAttribute("axis", softmax_axis_before);

      std::vector<NodeArg*> slice_inputs;

      NodeArg* indices_initializer = nullptr;

      if (is_scalar_slice) {
        indices_initializer = builder.MakeScalarInitializer<int64_t>(slice_data_value);
      } else {
        indices_initializer = builder.MakeInitializer<int64_t>({1}, {slice_data_value});
      }

      slice_inputs = {softmax_out, indices_initializer};

      auto* gather_out = builder.MakeIntermediate();
      builder.AddNode("Gather", slice_inputs,
                      {gather_out})
          .AddAttribute("axis", gather_axis);

      auto* identity_out = builder.MakeOutput();
      builder.AddNode("Identity", {gather_out}, {identity_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamGatherGraphTransformer>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger, std::move(transformer),
                                          TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

TEST(ComputeOptimizerTests, GatherReshape_ScalarSlicingOnBatchDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_scalar_batch_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 3U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 16);
    ASSERT_EQ(new_shape_const_values[2], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_reshape_scalar_batch_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherReshape_SlicingOnBatchDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_batch_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 0);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 4U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 0);
    ASSERT_EQ(new_shape_const_values[2], 16);
    ASSERT_EQ(new_shape_const_values[3], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_reshape_batch_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherReshape_ScalarSlicingOnSeqlenDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_scalar_seqlen_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 3U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 16);
    ASSERT_EQ(new_shape_const_values[2], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_reshape_scalar_seqlen_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherReshape_SlicingOnSeqlenDim) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_seqlen_dim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 4U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 0);
    ASSERT_EQ(new_shape_const_values[2], 16);
    ASSERT_EQ(new_shape_const_values[3], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_reshape_seqlen_dim_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 16;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherReshape_SlicingOnSeqlenDim2) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_reshape_seqlen_dim2.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("input1");
    ASSERT_EQ(consumers.size(), 1U);
    const Node* gather_node = consumers[0];
    ASSERT_EQ(gather_node->OpType(), "Gather");

    auto& attrs = gather_node->GetAttributes();
    ASSERT_TRUE(attrs.find("axis") != attrs.end());

    auto& axis_attr = attrs.at("axis");
    auto axis_value = (int)axis_attr.i();
    ASSERT_EQ(axis_value, 1);
  }

  {
    const Node* m5 = graph.GetProducerNode("reshape_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "Reshape");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Gather");

    ASSERT_TRUE(rhs_input == nullptr);
    InlinedVector<int64_t> new_shape_const_values;
    optimizer_utils::AppendTensorFromInitializer(graph, *m5->InputDefs()[1], new_shape_const_values, true);
    ASSERT_EQ(new_shape_const_values.size(), 4U);
    ASSERT_EQ(new_shape_const_values[0], 0);
    ASSERT_EQ(new_shape_const_values[1], 31);
    ASSERT_EQ(new_shape_const_values[2], 16);
    ASSERT_EQ(new_shape_const_values[3], 64);
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_reshape_seqlen_dim2_optimized.onnx"))};
  ASSERT_STATUS_OK(Model::Save(*model, new_model_uri));

  int64_t batch_size = 8;
  int64_t sequence_length = 128;
  int64_t hidden_size = 1024;

  InputContainer input_container;

  input_container.AddInput<float>("input1", {batch_size, sequence_length, hidden_size}, RandomFillFloatVector);

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

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

TEST(ComputeOptimizerTests, GatherRobertaE2E) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  // Be noted, all dropouts have ratio be 0.0, to make it easier to compare when running with the session.
  // This did not affect the transformer tests, because we did not remove the Dropout of ratio 0. in the middle.
  auto model_uri = MODEL_FOLDER "computation_reduction/gather/gather_roberta_e2e.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{4};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamGatherGraphTransformer>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<CommonSubexpressionElimination>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("c1_out");
    const Node* gather_node = nullptr;
    for (auto p_node : consumers) {
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("Gather") == 0) {
        gather_node = p_node;
        const Node* cast_node = graph.GetProducerNode(gather_node->InputDefs()[0]->Name());
        EXPECT_EQ(cast_node->OpType(), "Cast");
        EXPECT_EQ(cast_node->Name(), "c1");
        const auto& gather_consumers = graph.GetConsumerNodes(gather_node->OutputDefs()[0]->Name());
        EXPECT_EQ(gather_consumers[0]->OpType(), "Unsqueeze");
        break;
      }
    }

    ASSERT_FALSE(gather_node == nullptr);
  }

  // Check the second Gather.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("d1_out");
    const Node* gather_node = nullptr;
    for (auto p_node : consumers) {
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("Gather") == 0) {
        gather_node = p_node;
        const Node* dropout_node = graph.GetProducerNode(gather_node->InputDefs()[0]->Name());
        EXPECT_EQ(dropout_node->OpType(), "Dropout");
        EXPECT_EQ(dropout_node->Name(), "d1");
        const auto& gather_consumers = graph.GetConsumerNodes(gather_node->OutputDefs()[0]->Name());
        EXPECT_EQ(gather_consumers[0]->OpType(), "Add");
        EXPECT_EQ(gather_consumers[0]->Name(), "a6");
        break;
      }
    }

    ASSERT_FALSE(gather_node == nullptr);
  }

  // Check the input/output of the original Gather node.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("layernorm2_out");
    ASSERT_TRUE(consumers.size() == 1);
    ASSERT_FALSE(consumers[0] == nullptr);
    EXPECT_EQ(consumers[0]->OpType(), "Dropout");
    EXPECT_EQ(consumers[0]->Name(), "d6");
  }

  // Check MatMul(who gathers on the second last dim)'s input and output.
  {
    const Node* m5 = graph.GetProducerNode("m5_out");
    ASSERT_FALSE(m5 == nullptr);
    EXPECT_EQ(m5->OpType(), "MatMul");
    EXPECT_EQ(m5->Name(), "m5");

    const Node* lhs_input = graph.GetProducerNode(m5->InputDefs()[0]->Name());
    const Node* rhs_input = graph.GetProducerNode(m5->InputDefs()[1]->Name());

    ASSERT_FALSE(lhs_input == nullptr);
    EXPECT_EQ(lhs_input->OpType(), "Unsqueeze");

    ASSERT_FALSE(rhs_input == nullptr);
    EXPECT_EQ(rhs_input->OpType(), "Transpose");
    EXPECT_EQ(rhs_input->Name(), "transpose1");
  }

  // Check Add(who has broadcastable dim on gather axis)'s input and output.
  {
    const Node* a4 = graph.GetProducerNode("a4_out");
    ASSERT_FALSE(a4 == nullptr);
    EXPECT_EQ(a4->OpType(), "Add");
    EXPECT_EQ(a4->Name(), "a4");

    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("a4_out");
    ASSERT_TRUE(consumers.size() == 1);
    ASSERT_FALSE(consumers[0] == nullptr);
    EXPECT_EQ(consumers[0]->OpType(), "Squeeze");
  }

  // Check the result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("gather_roberta_e2e_optimized.onnx"))};
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

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#elif USE_ROCM
      onnxruntime::kRocmExecutionProvider,
#endif
  };

  const std::vector<std::string> output_names = {"final_output"};

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunModelWithData(model_uri, std::string("RawGraphRun"), provider_type,
                     input_container, output_names, expected_ort_values);

    std::vector<OrtValue> actual_ort_values;
    RunModelWithData(ToPathString(new_model_uri), std::string("OptimizedGraphRun"),
                     provider_type, input_container, output_names, actual_ort_values);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());

    // "expected 0.793675 (3f4b2e44), got 0.79232 (3f4ad584), diff: 0.00135422, tol=0.000179367 idx=4276.
    // 1713 of 8192 differ"
    // Loose the atol a bit because we see the MatMuls results differ once we move Gather before it.
    constexpr double per_sample_tolerance = 2e-3;
    constexpr double relative_per_sample_tolerance = 2e-3;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [4, 32, 256] (float)            graph input [4, 32, 256] (float)
                            |                                |
                             \_____________   ______________/
                                           \ /
                                           Add    ______ [16]
                                            |    /
                                      ShrunkenGather, axis = 1
                                            |
                                         Identity
                                            |
                                    graph output [4, 16, 256] (float)

Add an Identity node because currently we don't allow ShrunkenGather generates graph output.
*/
TEST(ComputeOptimizerTests, ShrunkenGatherElementwiseOps_PropagationOnTwoBranches) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  InlinedVector<int64_t> gather_indices;
  auto pre_graph_checker = [&gather_indices](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["com.microsoft.ShrunkenGather"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "ShrunkenGather") {
        TEST_RETURN_IF_NOT(gather_indices.empty());
        constexpr bool require_constant = true;
        NodeArg* initializer_node_arg = graph.GetNodeArg(node.InputDefs()[1]->Name());
        TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, gather_indices,
                                                                        require_constant));
      }
    }
    return Status::OK();
  };

  auto post_graph_checker = [&gather_indices](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["com.microsoft.ShrunkenGather"] == 2);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "Add") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          TEST_RETURN_IF_NOT(producer_node != nullptr);
          TEST_RETURN_IF_NOT(producer_node->OpType() == "ShrunkenGather");

          InlinedVector<int64_t> values;
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values,
                                                                          require_constant));
          for (size_t i = 0; i < values.size(); i++) {
            TEST_RETURN_IF_NOT(values[i] == gather_indices[i]);
          }
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          TEST_RETURN_IF_NOT(producer_node != nullptr);
          TEST_RETURN_IF_NOT(producer_node->OpType() == "ShrunkenGather");

          InlinedVector<int64_t> values;
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));
          for (size_t i = 0; i < values.size(); i++) {
            TEST_RETURN_IF_NOT(values[i] == gather_indices[i]);
          }
        }
      }
    }
    return Status::OK();
  };

  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<int64_t>({{4, 32, 256}});
    auto* input2_arg = builder.MakeInput<int64_t>({{4, 32, 256}});
    auto* add_out = builder.MakeIntermediate();
    builder.AddNode("Add", {input1_arg, input2_arg}, {add_out});

    const std::vector<int64_t> slice_shape{16};
    static RandomValueGenerator random{8888};
    std::vector<int64_t> random_slices = random.Uniform<int64_t>(slice_shape, 0, 32);
    auto* slice_initializer = builder.MakeInitializer<int64_t>(slice_shape, random_slices);
    auto* gather_out = builder.MakeIntermediate();
    builder.AddNode("ShrunkenGather", {add_out, slice_initializer}, {gather_out}, kMSDomain)
        .AddAttribute("axis", static_cast<int64_t>(1));

    auto* identity_out = builder.MakeOutput();
    builder.AddNode("Identity", {gather_out}, {identity_out});
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamGatherGraphTransformer>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger, std::move(transformer),
                                        TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

/*
Test graph includes multiple equivalent subgraphs as below.
           graph input [4, 32, 256] (float)            graph input [4, 32, 256] (float)
                            |                                |
                             \_____________   ______________/
                                           \ /
                                           Add  starts:(0)  ends: (-1)  axes: (1) steps: (1)
                                            \       \       |          /         /
                                               \       \     |        /       /
                                                  \      \   |     /      /
                                                    \     \  |   /     /
                                                       \   \ |  /   /
                                                            Slice
                                                             |
                                                          Identity
                                                             |
                                                graph output [4, 31, 256] (float)

Add an Identity node because currently we don't allow Slice generates graph output.
*/
TEST(ComputeOptimizerTests, SliceElementwiseOps_PropagationOnTwoBranches) {
  // 0: no input, 1: has input, 2: empty input
  std::vector<std::tuple<std::optional<int>, std::vector<int64_t>, int, int, bool>> has_axes_and_has_steps_pairs{
      {std::nullopt, {4, 32, 256}, 0, 0, false},  // {axis, data_shape, has_axes, has_steps, expected to propagate}
      {1, {4, 32, 256}, 1, 0, true},
      {1, {4, 32, 256}, 1, 1, true},
      {1, {4, 32, 256}, 1, 2, true},
      {std::nullopt, {4, 32, 256}, 2, 0, false},
      {std::nullopt, {4, 32, 256}, 2, 1, false},
      {std::nullopt, {4, 32, 256}, 2, 2, false},

      {std::nullopt, {256}, 0, 0, true},
      {0, {256}, 1, 0, true},
      {0, {256}, 1, 1, true},
      {0, {256}, 1, 2, true},
      {std::nullopt, {256}, 2, 0, true},
      {std::nullopt, {256}, 2, 1, true},
      {std::nullopt, {256}, 2, 2, true},
  };

  for (auto p : has_axes_and_has_steps_pairs) {
    std::optional<int> axis = std::get<0>(p);
    std::vector<int64_t> data_shape = std::get<1>(p);
    int has_axes = std::get<2>(p);
    int has_steps = std::get<3>(p);
    bool expected_to_propagate = std::get<4>(p);

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedVector<int64_t> starts_indices;
    auto pre_graph_checker = [&starts_indices](Graph& graph) -> Status {
      auto op_count_pre = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
      TEST_RETURN_IF_NOT(op_count_pre["Add"] == 1);
      TEST_RETURN_IF_NOT(op_count_pre["Slice"] == 1);
      TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Slice") {
          TEST_RETURN_IF_NOT(starts_indices.empty());
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(node.InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, starts_indices,
                                                                          require_constant));
        }
      }
      return Status::OK();
    };

    auto post_graph_checker = [&starts_indices, expected_to_propagate](Graph& graph) {
      auto op_count_post = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
      TEST_RETURN_IF_NOT(op_count_post["Add"] == 1);
      if (expected_to_propagate) {
        TEST_RETURN_IF_NOT(op_count_post["Slice"] == 2);
      } else {
        TEST_RETURN_IF_NOT(op_count_post["Slice"] == 1);
      }
      TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Add") {
          const auto& input_defs = node.InputDefs();

          {
            auto producer_node = graph.GetProducerNode(input_defs[0]->Name());

            if (expected_to_propagate) {
              TEST_RETURN_IF_NOT(producer_node != nullptr);
              TEST_RETURN_IF_NOT(producer_node->OpType() == "Slice");

              InlinedVector<int64_t> values;
              constexpr bool require_constant = true;
              NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
              TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values,
                                                                              require_constant));
              for (size_t i = 0; i < values.size(); i++) {
                TEST_RETURN_IF_NOT(values[i] == starts_indices[i]);
              }
            } else {
              TEST_RETURN_IF_NOT(producer_node == nullptr);
            }
          }

          {
            auto producer_node = graph.GetProducerNode(input_defs[1]->Name());

            if (expected_to_propagate) {
              TEST_RETURN_IF_NOT(producer_node != nullptr);
              TEST_RETURN_IF_NOT(producer_node->OpType() == "Slice");

              InlinedVector<int64_t> values;
              constexpr bool require_constant = true;
              NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
              TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));
              for (size_t i = 0; i < values.size(); i++) {
                TEST_RETURN_IF_NOT(values[i] == starts_indices[i]);
              }
            } else {
              TEST_RETURN_IF_NOT(producer_node == nullptr);
            }
          }
        }
      }
      return Status::OK();
    };

    auto build_test_case = [has_axes, has_steps, &data_shape, axis](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int64_t>(data_shape);
      auto* input2_arg = builder.MakeInput<int64_t>(data_shape);
      auto* add_out = builder.MakeIntermediate();
      builder.AddNode("Add", {input1_arg, input2_arg}, {add_out});

      auto* starts_initializer = builder.MakeInitializer<int64_t>({1}, {0});
      auto* ends_initializer = builder.MakeInitializer<int64_t>({1}, {-1});

      std::vector<NodeArg*> slice_inputs;
      slice_inputs = {add_out, starts_initializer, ends_initializer};

      NodeArg* axes_initializer = nullptr;
      NodeArg* steps_initializer = nullptr;
      if (has_axes == 0 && has_steps == 0) {
        // nothing
      } else if (has_axes == 1 && has_steps == 0) {
        axes_initializer = builder.MakeInitializer<int64_t>({1}, {axis.value()});
        slice_inputs.push_back(axes_initializer);
      } else if (has_axes == 1 && has_steps == 1) {
        axes_initializer = builder.MakeInitializer<int64_t>({1}, {axis.value()});
        slice_inputs.push_back(axes_initializer);
        steps_initializer = builder.MakeInitializer<int64_t>({1}, {1});
        slice_inputs.push_back(steps_initializer);
      } else if (has_axes == 1 && has_steps == 2) {
        axes_initializer = builder.MakeInitializer<int64_t>({1}, {axis.value()});
        slice_inputs.push_back(axes_initializer);
        steps_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(steps_initializer);
      } else if (has_axes == 2 && has_steps == 0) {
        axes_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(axes_initializer);
      } else if (has_axes == 2 && has_steps == 1) {
        axes_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(axes_initializer);
        steps_initializer = builder.MakeInitializer<int64_t>({1}, {1});
        slice_inputs.push_back(steps_initializer);
      } else if (has_axes == 2 && has_steps == 2) {
        axes_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(axes_initializer);
        steps_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(steps_initializer);
      }

      auto* slice_out = builder.MakeIntermediate();
      builder.AddNode("Slice", slice_inputs,
                      {slice_out});

      auto* identity_out = builder.MakeOutput();
      builder.AddNode("Identity", {slice_out}, {identity_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamGatherGraphTransformer>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger, std::move(transformer),
                                          TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph includes multiple equivalent subgraphs as below.
             graph input [2, 4, 32, 256] (float)
                            |
                        Transpose[perms=[0, 2, 1, 3]]
                            |
                      [2, 32, 4, 256]
                            |   starts:(0)  ends: (-1)  axes: (1) steps: (1)
                            \       \       |          /         /
                                \       \     |        /       /
                                  \      \   |     /      /
                                    \     \  |   /     /
                                        \   \ |  /   /
                                            Slice
                                              |
                                          Identity
                                              |
                                graph output [2, 31, 4, 256] (float)

Add an Identity node because currently, we don't allow Slice generates graph output.
*/
TEST(ComputeOptimizerTests, SliceTranspose_Propagation) {
  // 0: no input, 1: has input, 2: empty input
  std::vector<std::tuple<int, int, bool>> has_axes_and_has_steps_pairs{
      {0, 0, false},  // {has_axes, has_steps, expected to propagate}
      {1, 0, true},
      {1, 1, true},
      {1, 2, true},
      {2, 0, false},
      {2, 1, false},
      {2, 2, false},
  };

  for (auto p : has_axes_and_has_steps_pairs) {
    int has_axes = std::get<0>(p);
    int has_steps = std::get<1>(p);
    bool expected_to_propagate = std::get<2>(p);

    const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
    InlinedVector<int64_t> starts_indices;
    auto pre_graph_checker = [&starts_indices](Graph& graph) -> Status {
      auto op_count_pre = CountOpsInGraph(graph);
      TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
      TEST_RETURN_IF_NOT(op_count_pre["Transpose"] == 1);
      TEST_RETURN_IF_NOT(op_count_pre["Slice"] == 1);
      TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Slice") {
          TEST_RETURN_IF_NOT(starts_indices.empty());
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(node.InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, starts_indices,
                                                                          require_constant));
        }
      }
      return Status::OK();
    };

    auto post_graph_checker = [&starts_indices, expected_to_propagate](Graph& graph) {
      auto op_count_post = CountOpsInGraph(graph);

      TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
      TEST_RETURN_IF_NOT(op_count_post["Transpose"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["Slice"] == 1);
      TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

      for (Node& node : graph.Nodes()) {
        if (node.OpType() == "Transpose") {
          const auto& input_defs = node.InputDefs();

          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          if (expected_to_propagate) {
            TEST_RETURN_IF_NOT(producer_node != nullptr);
            TEST_RETURN_IF_NOT(producer_node->OpType() == "Slice");

            InlinedVector<int64_t> values;
            constexpr bool require_constant = true;
            NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
            TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values,
                                                                            require_constant));
            for (size_t i = 0; i < values.size(); i++) {
              TEST_RETURN_IF_NOT(values[i] == starts_indices[i]);
            }

            const ONNX_NAMESPACE::TensorShapeProto* slice_out_shape = producer_node->OutputDefs()[0]->Shape();
            TEST_RETURN_IF_NOT(slice_out_shape != nullptr);
            TEST_RETURN_IF_NOT(slice_out_shape->dim_size() == 4);
            TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(0)) && slice_out_shape->dim(0).dim_value() == 2);
            TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(1)) && slice_out_shape->dim(1).dim_value() == 4);
            TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(2)) && slice_out_shape->dim(2).dim_value() == 31);
            TEST_RETURN_IF_NOT(utils::HasDimValue(slice_out_shape->dim(3)) && slice_out_shape->dim(3).dim_value() == 256);
          } else {
            TEST_RETURN_IF_NOT(producer_node == nullptr);
          }
        }
      }

      return Status::OK();
    };

    auto build_test_case = [has_axes, has_steps](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int64_t>({{2, 4, 32, 256}});
      auto* trans_out = builder.MakeIntermediate();
      builder.AddNode("Transpose", {input1_arg}, {trans_out})
          .AddAttribute("perm", std::vector<int64_t>{0, 2, 1, 3});

      std::vector<NodeArg*> slice_inputs;

      auto* starts_initializer = builder.MakeInitializer<int64_t>({1}, {0});
      auto* ends_initializer = builder.MakeInitializer<int64_t>({1}, {-1});

      slice_inputs = {trans_out, starts_initializer, ends_initializer};

      NodeArg* axes_initializer = nullptr;
      NodeArg* steps_initializer = nullptr;
      if (has_axes == 0 && has_steps == 0) {
        // nothing
      } else if (has_axes == 1 && has_steps == 0) {
        axes_initializer = builder.MakeInitializer<int64_t>({1}, {1});
        slice_inputs.push_back(axes_initializer);
      } else if (has_axes == 1 && has_steps == 1) {
        axes_initializer = builder.MakeInitializer<int64_t>({1}, {1});
        slice_inputs.push_back(axes_initializer);
        steps_initializer = builder.MakeInitializer<int64_t>({1}, {1});
        slice_inputs.push_back(steps_initializer);
      } else if (has_axes == 1 && has_steps == 2) {
        axes_initializer = builder.MakeInitializer<int64_t>({1}, {1});
        slice_inputs.push_back(axes_initializer);
        steps_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(steps_initializer);
      } else if (has_axes == 2 && has_steps == 0) {
        axes_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(axes_initializer);
      } else if (has_axes == 2 && has_steps == 1) {
        axes_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(axes_initializer);
        steps_initializer = builder.MakeInitializer<int64_t>({1}, {1});
        slice_inputs.push_back(steps_initializer);
      } else if (has_axes == 2 && has_steps == 2) {
        axes_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(axes_initializer);
        steps_initializer = builder.MakeEmptyInput();
        slice_inputs.push_back(steps_initializer);
      }

      auto* slice_out = builder.MakeIntermediate();
      builder.AddNode("Slice", slice_inputs,
                      {slice_out});

      auto* identity_out = builder.MakeOutput();
      builder.AddNode("Identity", {slice_out}, {identity_out});
    };

    std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamGatherGraphTransformer>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger, std::move(transformer),
                                          TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph includes multiple equivalent subgraphs as below.
           graph input [4, 32, 256] (float)            graph input [4, 32, 256] (float)
                            |                                |
                             \_____________   ______________/
                                           \ /
                                           Add  starts:(0,0)  ends: (-1,-1)  axes: (0,1) steps: (1,1)
                                            \       \        |              /           /
                                               \       \     |           /          /
                                                  \      \   |        /         /
                                                    \     \  |     /        /
                                                       \   \ |   /     /
                                                            Slice
                                                             |
                                                          Identity
                                                             |
                                                graph output [3, 31, 256] (float)

Add an Identity node because currently we don't allow Slice generates graph output.
*/
TEST(ComputeOptimizerTests, SliceElementwiseOps_NoPropagationForMutipleAxesSlice) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Slice"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);

    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Slice"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    return Status::OK();
  };

  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<int64_t>({{4, 32, 256}});
    auto* input2_arg = builder.MakeInput<int64_t>({{4, 32, 256}});
    auto* add_out = builder.MakeIntermediate();
    builder.AddNode("Add", {input1_arg, input2_arg}, {add_out});

    auto* starts_initializer = builder.MakeInitializer<int64_t>({2}, {0, 0});
    auto* ends_initializer = builder.MakeInitializer<int64_t>({2}, {-1, -1});
    auto* axes_initializer = builder.MakeInitializer<int64_t>({2}, {0, 1});
    auto* steps_initializer = builder.MakeInitializer<int64_t>({2}, {1, 1});
    auto* slice_out = builder.MakeIntermediate();
    builder.AddNode("Slice", {add_out, starts_initializer, ends_initializer, axes_initializer, steps_initializer},
                    {slice_out});

    auto* identity_out = builder.MakeOutput();
    builder.AddNode("Identity", {slice_out}, {identity_out});
  };

  std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamGatherGraphTransformer>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger, std::move(transformer),
                                        TransformerLevel::Level1,
                                        1, pre_graph_checker, post_graph_checker));
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [4, 32, 256] (int64_t)            graph input [4, 32, 256] (int64_t)
                            |                                |
                             \_____________   ______________/
                                           \ /
                                           Add
                                            |
                                         Reshape
                                            |
                                         Identity
                                            |
                                    graph out [128, 256] (int64_t)

Add an Identity node because currently we don't allow Reshape generate graph output.
*/
TEST(ComputeOptimizerTests, ReshapeElementwiseOps_PropagationOnTwoBranches) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Reshape"] == 2);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "Add") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          TEST_RETURN_IF_NOT(producer_node != nullptr);
          TEST_RETURN_IF_NOT(producer_node->OpType() == "Reshape");

          InlinedVector<int64_t> values;
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));
          TEST_RETURN_IF_NOT(values.size() == 2);
          TEST_RETURN_IF_NOT(values[0] == -1);
          TEST_RETURN_IF_NOT(values[1] == 256);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          TEST_RETURN_IF_NOT(producer_node != nullptr);
          TEST_RETURN_IF_NOT(producer_node->OpType() == "Reshape");

          InlinedVector<int64_t> values;
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));
          TEST_RETURN_IF_NOT(values.size() == 2);
          TEST_RETURN_IF_NOT(values[0] == -1);
          TEST_RETURN_IF_NOT(values[1] == 256);
        }
      }
    }
    return Status::OK();
  };

  std::vector<int> fist_dim_values = {-1, 128};
  for (auto first_dim_value : fist_dim_values) {
    auto build_test_case = [&first_dim_value](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<int64_t>({{4, 32, 256}});
      auto* input2_arg = builder.MakeInput<int64_t>({{4, 32, 256}});
      auto* add_out = builder.MakeIntermediate();
      builder.AddNode("Add", {input1_arg, input2_arg}, {add_out});

      auto* shape_initializer = builder.MakeInitializer<int64_t>({2}, {first_dim_value, 256});
      auto* reshape_out = builder.MakeIntermediate();
      builder.AddNode("Reshape", {add_out, shape_initializer}, {reshape_out});

      auto* identity_out = builder.MakeOutput();
      builder.AddNode("Identity", {reshape_out}, {identity_out});
    };

    const std::vector<int> opsets{12, 13, 14};
    for (auto& opset_version : opsets) {
      std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamReshapeGraphTransformer>();
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer),
                                            TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [4, 32, 256] (int64_t)            graph input [256] (int64_t)
                            |                                |
                             \_____________   ______________/
                                           \ /
                                           Add
                                            |
                                         Reshape
                                            |
                                         Identity
                                            |
                                    graph out [128, 256] (int64_t)

Add an Identity node because currently we don't allow Reshape generate graph output.

*/
TEST(ComputeOptimizerTests, ReshapeElementwiseOps_PropagationOnOneBranch_1DBroadcast) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "Add") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          TEST_RETURN_IF_NOT(producer_node != nullptr);
          TEST_RETURN_IF_NOT(producer_node->OpType() == "Reshape");

          InlinedVector<int64_t> values;
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));
          TEST_RETURN_IF_NOT(values.size() == 2);
          TEST_RETURN_IF_NOT(values[0] == -1);
          TEST_RETURN_IF_NOT(values[1] == 256);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }
      }
    }
    return Status::OK();
  };

  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<int64_t>({{4, 32, 256}});
    auto* input2_arg = builder.MakeInput<int64_t>({{256}});
    auto* add_out = builder.MakeIntermediate();
    builder.AddNode("Add", {input1_arg, input2_arg}, {add_out});

    auto* shape_initializer = builder.MakeInitializer<int64_t>({2}, {-1, 256});
    auto* reshape_out = builder.MakeIntermediate();
    builder.AddNode("Reshape", {add_out, shape_initializer}, {reshape_out});

    auto* identity_out = builder.MakeOutput();
    builder.AddNode("Identity", {reshape_out}, {identity_out});
  };

  const std::vector<int> opsets{12, 13, 14};
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamReshapeGraphTransformer>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer), TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [4, 1, 256] (int64_t)            graph input [32, 256] (int64_t)
                            |                                |
                             \_____________   ______________/
                                           \ /
                                           Add
                                            |
                                         Reshape
                                            |
                                         Identity
                                            |
                                    graph out [128, 256] (int64_t)

Add an Identity node because currently we don't allow Reshape generate graph output.

*/
TEST(ComputeOptimizerTests, ReshapeElementwiseOps_NoPropagation1) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "Add") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }
      }
    }
    return Status::OK();
  };

  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<int64_t>({{4, 1, 256}});
    auto* input2_arg = builder.MakeInput<int64_t>({{32, 256}});
    auto* add_out = builder.MakeIntermediate();
    builder.AddNode("Add", {input1_arg, input2_arg}, {add_out});

    auto* shape_initializer = builder.MakeInitializer<int64_t>({2}, {-1, 256});
    auto* reshape_out = builder.MakeIntermediate();
    builder.AddNode("Reshape", {add_out, shape_initializer}, {reshape_out});

    auto* identity_out = builder.MakeOutput();
    builder.AddNode("Identity", {reshape_out}, {identity_out});
  };

  const std::vector<int> opsets{12, 13, 14};
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamReshapeGraphTransformer>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer), TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
          graph input [128, 4, 32] (int64_t)
                          |
                        Cast    initializer value: (-1, 128)
                          |    /
                        Reshape
                          |
                        Identity
                          |
                  graph out [128, 128] (int64_t)

Add an Identity node because currently we don't allow Reshape generate graph output.
*/
TEST(ComputeOptimizerTests, ReshapeElementwiseOps_NoPropagation2) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["Cast"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "Reshape") {
        const auto& input_defs = node.InputDefs();
        auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
        TEST_RETURN_IF_NOT(producer_node != nullptr);
        TEST_RETURN_IF_NOT(producer_node->OpType() == "Cast");
      }
    }
    return Status::OK();
  };

  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<int64_t>({{128, 4, 32}});
    auto* cast_out = builder.MakeIntermediate();
    builder.AddNode("Cast", {input1_arg}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT64));

    auto* shape_initializer = builder.MakeInitializer<int64_t>({2}, {-1, 128});
    auto* reshape_out = builder.MakeIntermediate();
    builder.AddNode("Reshape", {cast_out, shape_initializer}, {reshape_out});

    auto* identity_out = builder.MakeOutput();
    builder.AddNode("Identity", {reshape_out}, {identity_out});
  };

  const std::vector<int> opsets{12, 13, 14};
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamReshapeGraphTransformer>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer),
                                          TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [4, 32, 256] (int64_t)            graph input () (scalar, int64_t)
                            |                                |
                             \_____________   ______________/
                                           \ /
                                           Add
                                            |
                                         Reshape
                                            |
                                         Identity
                                            |
                                    graph out [128, 256] (int64_t)

Add an Identity node because currently we don't allow Reshape generate graph output.

*/
TEST(ComputeOptimizerTests, ReshapeElementwiseOps_PropagationOnOneBranch_ScalarBroadcast) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "Add") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          TEST_RETURN_IF_NOT(producer_node != nullptr);
          TEST_RETURN_IF_NOT(producer_node->OpType() == "Reshape");

          InlinedVector<int64_t> values;
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));
          TEST_RETURN_IF_NOT(values.size() == 2);
          TEST_RETURN_IF_NOT(values[0] == -1);
          TEST_RETURN_IF_NOT(values[1] == 256);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }
      }
    }
    return Status::OK();
  };

  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input1_arg = builder.MakeInput<int64_t>({{4, 32, 256}});
    auto* input2_arg = builder.MakeScalarInitializer<int64_t>(2);
    auto* add_out = builder.MakeIntermediate();
    builder.AddNode("Add", {input1_arg, input2_arg}, {add_out});

    auto* shape_initializer = builder.MakeInitializer<int64_t>({2}, {-1, 256});
    auto* reshape_out = builder.MakeIntermediate();
    builder.AddNode("Reshape", {add_out, shape_initializer}, {reshape_out});

    auto* identity_out = builder.MakeOutput();
    builder.AddNode("Identity", {reshape_out}, {identity_out});
  };

  const std::vector<int> opsets{12, 13, 14};
  for (auto& opset_version : opsets) {
    std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamReshapeGraphTransformer>();
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer),
                                          TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [4, 32, 256] (float)            graph input [256, 256] (float)
                            |                                |
                             \_____________   ______________/
                                           \ /
                                          MatMul
                                            |
                                         Reshape
                                            |
                                         Identity
                                            |
                                    graph out [128, 256] (float)

Add an Identity node because currently we don't allow Reshape generate graph output.
*/
TEST(ComputeOptimizerTests, ReshapeMatMul_PropagationOnLeftBranch) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["MatMul"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["MatMul"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "MatMul") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          TEST_RETURN_IF_NOT(producer_node != nullptr);
          TEST_RETURN_IF_NOT(producer_node->OpType() == "Reshape");

          InlinedVector<int64_t> values;
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));
          TEST_RETURN_IF_NOT(values.size() == 2);
          TEST_RETURN_IF_NOT(values[0] == -1);
          TEST_RETURN_IF_NOT(values[1] == 256);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }
      }
    }
    return Status::OK();
  };

  std::vector<int> fist_dim_values = {-1, 128};
  for (auto first_dim_value : fist_dim_values) {
    auto build_test_case = [&first_dim_value](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{4, 32, 256}});
      auto* input2_arg = builder.MakeInput<float>({{256, 256}});
      auto* matmul_out = builder.MakeIntermediate();
      builder.AddNode("MatMul", {input1_arg, input2_arg}, {matmul_out});

      auto* shape_initializer = builder.MakeInitializer<int64_t>({2}, {first_dim_value, 256});
      auto* reshape_out = builder.MakeIntermediate();
      builder.AddNode("Reshape", {matmul_out, shape_initializer}, {reshape_out});

      auto* identity_out = builder.MakeOutput();
      builder.AddNode("Identity", {reshape_out}, {identity_out});
    };

    const std::vector<int> opsets{12, 13, 14};
    for (auto& opset_version : opsets) {
      std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamReshapeGraphTransformer>();
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer),
                                            TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [4, 32, 1024] (float)       graph input [1024] (float)     graph input [1024] (float)
                            |                         |                             /
                             \_____________   _______/  __________________________/
                                           \ /         /
                                    LayerNormalization
                                            |
                                         Reshape
                                            |
                                         Identity
                                            |
                                    graph out [128, 1024] (float)

Add an Identity node because currently we don't allow Reshape generate graph output.
*/
TEST(ComputeOptimizerTests, ReshapeLayerNormalization_PropagationOnOneBranch) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["LayerNormalization"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["LayerNormalization"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "LayerNormalization") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          TEST_RETURN_IF_NOT(producer_node != nullptr);
          TEST_RETURN_IF_NOT(producer_node->OpType() == "Reshape");

          InlinedVector<int64_t> values;
          constexpr bool require_constant = true;
          NodeArg* initializer_node_arg = graph.GetNodeArg(producer_node->InputDefs()[1]->Name());
          TEST_RETURN_IF_NOT(optimizer_utils::AppendTensorFromInitializer(graph, *initializer_node_arg, values, require_constant));
          TEST_RETURN_IF_NOT(values.size() == 2);
          TEST_RETURN_IF_NOT(values[0] == -1);
          TEST_RETURN_IF_NOT(values[1] == 1024);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[2]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }
      }
    }
    return Status::OK();
  };

  std::vector<int> fist_dim_values = {-1, 128};
  for (auto first_dim_value : fist_dim_values) {
    auto build_test_case = [&first_dim_value](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{4, 32, 1024}});
      auto* input2_arg = builder.MakeInput<float>({{1024}});
      auto* input3_arg = builder.MakeInput<float>({{1024}});
      auto* ln_out = builder.MakeIntermediate();
      builder.AddNode("LayerNormalization", {input1_arg, input2_arg, input3_arg}, {ln_out})
          .AddAttribute("axis", static_cast<int64_t>(-1));

      auto* shape_initializer = builder.MakeInitializer<int64_t>({2}, {first_dim_value, 1024});
      auto* reshape_out = builder.MakeIntermediate();
      builder.AddNode("Reshape", {ln_out, shape_initializer}, {reshape_out});

      auto* identity_out = builder.MakeOutput();
      builder.AddNode("Identity", {reshape_out}, {identity_out});
    };

    const std::vector<int> opsets{12, 13, 14};
    for (auto& opset_version : opsets) {
      std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamReshapeGraphTransformer>();
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer),
                                            TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

/*
Test graph include multiple equivalent subgraphs as below.
           graph input [4, 32, 1024] (float)       graph input [1024] (float)     graph input [1024] (float)
                            |                         |                             /
                             \_____________   _______/  __________________________/
                                           \ /         /
                                    LayerNormalization
                                            |
                                         Reshape
                                            |
                                         Identity
                                            |
                                    graph out [128, 1024] (float)

Add an Identity node because currently we don't allow Reshape generate graph output.
*/
TEST(ComputeOptimizerTests, ReshapeLayerNormalization_NoPropagation) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_pre["LayerNormalization"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count_post = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_post.size() == 3U);
    TEST_RETURN_IF_NOT(op_count_post["LayerNormalization"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Reshape"] == 1);
    TEST_RETURN_IF_NOT(op_count_post["Identity"] == 1);

    for (Node& node : graph.Nodes()) {
      if (node.OpType() == "LayerNormalization") {
        const auto& input_defs = node.InputDefs();

        {
          auto producer_node = graph.GetProducerNode(input_defs[0]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[1]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }

        {
          auto producer_node = graph.GetProducerNode(input_defs[2]->Name());
          TEST_RETURN_IF_NOT(producer_node == nullptr);
        }
      }
    }
    return Status::OK();
  };

  std::vector<int> fist_dim_values = {-1, 128};
  for (auto first_dim_value : fist_dim_values) {
    auto build_test_case = [&first_dim_value](ModelTestBuilder& builder) {
      auto* input1_arg = builder.MakeInput<float>({{4, 32, 1024}});
      auto* input2_arg = builder.MakeInput<float>({{1024}});
      auto* input3_arg = builder.MakeInput<float>({{1024}});
      auto* ln_out = builder.MakeIntermediate();
      builder.AddNode("LayerNormalization", {input1_arg, input2_arg, input3_arg}, {ln_out})
          .AddAttribute("axis", static_cast<int64_t>(1));

      auto* shape_initializer = builder.MakeInitializer<int64_t>({2}, {first_dim_value, 1024});
      auto* reshape_out = builder.MakeIntermediate();
      builder.AddNode("Reshape", {ln_out, shape_initializer}, {reshape_out});

      auto* identity_out = builder.MakeOutput();
      builder.AddNode("Identity", {reshape_out}, {identity_out});
    };

    const std::vector<int> opsets{12, 13, 14};
    for (auto& opset_version : opsets) {
      std::unique_ptr<GraphTransformer> transformer = std::make_unique<UpStreamReshapeGraphTransformer>();
      ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer),
                                            TransformerLevel::Level1,
                                            1, pre_graph_checker, post_graph_checker));
    }
  }
}

TEST(ComputeOptimizerTests, ReshapeMlmBertE2E) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  // Be noted all dropout have a ratio be 0.0, to make it easier to compare when running with the session.
  // This did not affect the transformer tests, because we did not remove the Dropout of ratio 0. in the middle.
  auto model_uri = MODEL_FOLDER "computation_reduction/reshape/mlm_bert_e2e.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{3};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<UpStreamReshapeGraphTransformer>(),
                                                     TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));

  /*
   Reshape node can be moved from the original place up to LayerNorm node generating "layernorm1_out".

                        LayerNorm
                     (layernorm1_out)
                        /       \
                    Reshape    Reshape
   */
  GraphViewer graph_viewer(graph);
  // Check the first Gather.
  {
    const std::vector<const Node*>& layer_norm1_out_consumers = graph.GetConsumerNodes("layernorm1_out");
    EXPECT_EQ(layer_norm1_out_consumers.size(), 2U);
    for (auto reshape_node : layer_norm1_out_consumers) {
      ASSERT_FALSE(reshape_node == nullptr);
      if (reshape_node->OpType().compare("Reshape") == 0) {
        const Node* parent_node = graph.GetProducerNode(reshape_node->InputDefs()[0]->Name());
        EXPECT_EQ(parent_node->OpType(), "LayerNormalization");
        EXPECT_EQ(parent_node->Name(), "layernorm1");

        InlinedVector<int64_t> new_shape_const_values;
        ASSERT_TRUE(optimizer_utils::AppendTensorFromInitializer(graph, *reshape_node->InputDefs()[1],
                                                                 new_shape_const_values, true));
        ASSERT_EQ(new_shape_const_values.size(), 2U);
        ASSERT_EQ(new_shape_const_values[0], -1);
        ASSERT_EQ(new_shape_const_values[1], 1024);
      }
    }
  }

  // Check the original place of Reshape.
  {
    const std::vector<const Node*>& consumers = graph.GetConsumerNodes("a10_out");
    ASSERT_TRUE(consumers.size() == 1);
    ASSERT_FALSE(consumers[0] == nullptr);
    EXPECT_EQ(consumers[0]->OpType(), "Cast");
    EXPECT_EQ(consumers[0]->Name(), "c10");
  }

  // Check result diff after the re-order
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("compute_optimizer_test_tmp_dir")};
  PathString new_model_uri{ConcatPathComponent(tmp_dir.Path(),
                                               ORT_TSTR("reshape_bert_e2e_optimized.onnx"))};
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

#endif

}  // namespace test
}  // namespace onnxruntime

#endif
