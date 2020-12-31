// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>

#include "gtest/gtest.h"
#include "test/framework/test_utils.h"
#include "orttraining/core/framework/module_gradient_graph_builder.h"

namespace onnxruntime {
namespace test {

void RunModuleGradientGraphBuilderTest(const std::string& file_path,
                                       const std::vector<std::string>& initializer_names_to_train,
                                       const std::map<std::string, int>& expected_forward_ops_count,
                                       const std::map<std::string, int>& expected_backward_ops_count) {
  onnxruntime::training::ModuleGradientGraphBuilderConfiguration config;
  config.initializer_names_to_train.assign(initializer_names_to_train.begin(), initializer_names_to_train.end());
  std::ifstream model_istream(file_path, std::ifstream::in | std::ifstream::binary);
  onnxruntime::training::ModuleGradientGraphBuilder module_gradient_graph_builder;
  ASSERT_TRUE(module_gradient_graph_builder.Initialize(model_istream, config).IsOK());
  ASSERT_TRUE(module_gradient_graph_builder.BuildAndSplit().IsOK());

  // Forward graph.
  std::istringstream forward_is(module_gradient_graph_builder.GetForwardModel());
  ONNX_NAMESPACE::ModelProto forward_model_proto;
  ASSERT_TRUE(Model::Load(forward_is, &forward_model_proto).IsOK());
  Model forward_model(forward_model_proto, nullptr, logging::LoggingManager::DefaultLogger());
  std::map<std::string, int> actual_forward_ops_count = CountOpsInGraph(forward_model.MainGraph());
  ASSERT_EQ(actual_forward_ops_count.size(), expected_forward_ops_count.size());
  for (const auto& op_count : actual_forward_ops_count) {
    ASSERT_TRUE(expected_forward_ops_count.find(op_count.first) != expected_forward_ops_count.end());
    ASSERT_EQ(op_count.second, expected_forward_ops_count.at(op_count.first));
  }

  std::cout << std::endl;

  // Backward graph.
  std::istringstream backward_is(module_gradient_graph_builder.GetBackwardModel());
  ONNX_NAMESPACE::ModelProto backward_model_proto;
  ASSERT_TRUE(Model::Load(backward_is, &backward_model_proto).IsOK());
  Model backward_model(backward_model_proto, nullptr, logging::LoggingManager::DefaultLogger());
  std::map<std::string, int> actual_backward_ops_count = CountOpsInGraph(backward_model.MainGraph());
  ASSERT_EQ(actual_backward_ops_count.size(), expected_backward_ops_count.size());
  for (const auto& op_count : actual_backward_ops_count) {
    ASSERT_TRUE(expected_backward_ops_count.find(op_count.first) != expected_backward_ops_count.end());
    ASSERT_EQ(op_count.second, expected_backward_ops_count.at(op_count.first));
  }

  // SplitGraphsInfo.
  const onnxruntime::training::SplitGraphsInfo& split_graphs_info = module_gradient_graph_builder.GetSplitGraphsInfo();
  ASSERT_TRUE(split_graphs_info.initializer_names_to_train == initializer_names_to_train);

  std::vector<std::string> expected_forward_input_names;
  std::vector<std::string> expected_forward_output_names;
  std::vector<std::string> actual_forward_input_names;
  std::vector<std::string> actual_forward_output_names;
  std::vector<std::string> expected_backward_input_names;
  std::vector<std::string> expected_backward_output_names;
  std::vector<std::string> actual_backward_input_names;
  std::vector<std::string> actual_backward_output_names;

  expected_forward_input_names.insert(expected_forward_input_names.end(), split_graphs_info.user_input_names.begin(),
                                      split_graphs_info.user_input_names.end());
  expected_forward_input_names.insert(expected_forward_input_names.end(),
                                      split_graphs_info.initializer_names_to_train.begin(),
                                      split_graphs_info.initializer_names_to_train.end());
  expected_forward_output_names.insert(expected_forward_output_names.end(), split_graphs_info.user_output_names.begin(),
                                       split_graphs_info.user_output_names.end());
  expected_forward_output_names.insert(expected_forward_output_names.end(),
                                       split_graphs_info.intermediate_tensor_names.begin(),
                                       split_graphs_info.intermediate_tensor_names.end());

  expected_backward_input_names.insert(expected_backward_input_names.end(),
                                       split_graphs_info.backward_user_input_names.begin(),
                                       split_graphs_info.backward_user_input_names.end());
  expected_backward_input_names.insert(expected_backward_input_names.end(),
                                       split_graphs_info.backward_intializer_names_as_input.begin(),
                                       split_graphs_info.backward_intializer_names_as_input.end());
  expected_backward_input_names.insert(expected_backward_input_names.end(),
                                       split_graphs_info.intermediate_tensor_names.begin(),
                                       split_graphs_info.intermediate_tensor_names.end());
  expected_backward_input_names.insert(expected_backward_input_names.end(),
                                       split_graphs_info.backward_output_grad_names.begin(),
                                       split_graphs_info.backward_output_grad_names.end());
  expected_backward_output_names.insert(expected_backward_output_names.end(),
                                        split_graphs_info.initializer_grad_names_to_train.begin(),
                                        split_graphs_info.initializer_grad_names_to_train.end());

  const std::vector<const NodeArg*>& forward_graph_inputs = forward_model.MainGraph().GetInputsIncludingInitializers();
  for (auto& node_arg : forward_graph_inputs) {
    actual_forward_input_names.emplace_back(node_arg->Name());
  }

  const std::vector<const NodeArg*>& forward_graph_outputs = forward_model.MainGraph().GetOutputs();
  for (auto& node_arg : forward_graph_outputs) {
    actual_forward_output_names.emplace_back(node_arg->Name());
  }

  const std::vector<const NodeArg*>& backward_graph_inputs =
      backward_model.MainGraph().GetInputsIncludingInitializers();
  for (auto& node_arg : backward_graph_inputs) {
    actual_backward_input_names.emplace_back(node_arg->Name());
  }

  const std::vector<const NodeArg*>& backward_graph_outputs = backward_model.MainGraph().GetOutputs();
  for (auto& node_arg : backward_graph_outputs) {
    actual_backward_output_names.emplace_back(node_arg->Name());
  }

  ASSERT_TRUE(expected_forward_input_names == actual_forward_input_names);
  ASSERT_TRUE(expected_forward_output_names == actual_forward_output_names);
  ASSERT_TRUE(expected_backward_input_names == actual_backward_input_names);
  ASSERT_TRUE(expected_backward_output_names == actual_backward_output_names);
}

TEST(ModuleGradientGraphBuilderTest, GraphSplit_Mnist) {
  std::string file_path = "testdata/test_training_model.onnx";
  std::vector<std::string> initializer_names_to_train{"W1", "B1", "W2", "B2", "W3", "B3"};
  std::map<std::string, int> expected_forward_ops_count = {{"Add", 3}, {"MatMul", 3}, {"Relu", 2}};
  std::map<std::string, int> expected_backward_ops_count = {
      {"Gemm", 5}, {"Identity", 3}, {"ReduceSum", 3}, {"com.microsoft.ReluGrad", 2}};

  RunModuleGradientGraphBuilderTest(file_path, initializer_names_to_train, expected_forward_ops_count,
                                    expected_backward_ops_count);
}

TEST(ModuleGradientGraphBuilderTest, GraphSplit_BertToy) {
  std::string file_path = "testdata/bert_toy_optimized.onnx";
  const auto model_path = ORT_TSTR(file_path);
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_path, p_model, nullptr, logging::LoggingManager::DefaultLogger()).IsOK());
  std::vector<std::string> initializer_names_to_train;
  const Graph& graph = p_model->MainGraph();
  const auto& initializers = graph.GetAllInitializedTensors();
  for (const auto& initializer : initializers) {
    if (initializer.first.rfind("bert.") == 0 || initializer.first.rfind("cls.") == 0) {
      initializer_names_to_train.emplace_back(initializer.first);
    }
  }

  std::map<std::string, int> expected_forward_ops_count = {{"Add", 43},
                                                           {"Cast", 3},
                                                           {"Div", 5},
                                                           {"Expand", 1},
                                                           {"Gather", 4},
                                                           {"Gemm", 2},
                                                           {"LayerNormalization", 12},
                                                           {"MatMul", 42},
                                                           {"Min", 1},
                                                           {"Mul", 1},
                                                           {"Reshape", 20},
                                                           {"Shape", 1},
                                                           {"Slice", 1},
                                                           {"Softmax", 5},
                                                           {"Sub", 1},
                                                           {"Tanh", 1},
                                                           {"Transpose", 21},
                                                           {"Unsqueeze", 2},
                                                           {"com.microsoft.BiasGelu", 6}};

  std::map<std::string, int> expected_backward_ops_count = {{"Div", 5},
                                                            {"Gemm", 36},
                                                            {"Identity", 50},
                                                            {"MatMul", 52},
                                                            {"Mul", 2},
                                                            {"ReduceSum", 34},
                                                            {"Reshape", 84},
                                                            {"Shape", 1},
                                                            {"Sub", 1},
                                                            {"Sum", 12},
                                                            {"Transpose", 73},
                                                            {"com.microsoft.BiasGeluGrad_dX", 6},
                                                            {"com.microsoft.GatherGrad", 4},
                                                            {"com.microsoft.LayerNormalizationGrad", 12},
                                                            {"com.microsoft.SoftmaxGrad", 5}};

  RunModuleGradientGraphBuilderTest(file_path, initializer_names_to_train, expected_forward_ops_count,
                                    expected_backward_ops_count);
}

}  // namespace test
}  // namespace onnxruntime
