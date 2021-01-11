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
                                       const std::map<std::string, int>& expected_backward_ops_count,
                                       const std::vector<std::string>& expected_intermediate_tensor_names,
                                       const std::vector<std::string>& expected_backward_user_input_names,
                                       const std::vector<std::string>& expected_backward_trainable_initializer_names,
                                       const std::vector<std::string>& expected_backward_output_grad_names) {
  onnxruntime::training::ModuleGradientGraphBuilderConfiguration config;
  config.initializer_names_to_train.assign(initializer_names_to_train.begin(), initializer_names_to_train.end());
  std::ifstream model_istream(file_path, std::ifstream::in | std::ifstream::binary);
  onnxruntime::training::ModuleGradientGraphBuilder module_gradient_graph_builder;
  ASSERT_TRUE(module_gradient_graph_builder.Initialize(model_istream, config).IsOK());
  ASSERT_TRUE(module_gradient_graph_builder.Build().IsOK());

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
  ASSERT_TRUE(split_graphs_info.intermediate_tensor_names == expected_intermediate_tensor_names);
  ASSERT_TRUE(split_graphs_info.backward_user_input_names == expected_backward_user_input_names);
  ASSERT_TRUE(split_graphs_info.backward_intializer_names_as_input == expected_backward_trainable_initializer_names);
  ASSERT_TRUE(split_graphs_info.backward_output_grad_names == expected_backward_output_grad_names);

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
  const std::vector<std::string>& expected_intermediate_tensor_names{"T3", "T6"};
  const std::vector<std::string>& expected_backward_user_input_names{"X"};
  const std::vector<std::string>& expected_backward_trainable_initializer_names{"W2", "W3"};
  const std::vector<std::string>& expected_backward_output_grad_names{"predictions_grad"};

  RunModuleGradientGraphBuilderTest(file_path, initializer_names_to_train, expected_forward_ops_count,
                                    expected_backward_ops_count, expected_intermediate_tensor_names,
                                    expected_backward_user_input_names, expected_backward_trainable_initializer_names,
                                    expected_backward_output_grad_names);
}

#ifdef USE_CUDA
// The transformers will generate different graph when USE_CUDA is defined or not for BertToy model,
// so test it for CUDA enabled only.
TEST(ModuleGradientGraphBuilderTest, GraphSplit_BertToy) {
  std::string file_path = "testdata/bert_toy_optimized.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(ToPathString(file_path), p_model, nullptr, logging::LoggingManager::DefaultLogger()).IsOK());
  std::vector<std::string> initializer_names_to_train;
  const Graph& graph = p_model->MainGraph();
  const auto& initializers = graph.GetAllInitializedTensors();
  for (const auto& initializer : initializers) {
    if (initializer.first.rfind("bert.") == 0 || initializer.first.rfind("cls.") == 0) {
      initializer_names_to_train.emplace_back(initializer.first);
    }
  }

  std::sort(initializer_names_to_train.begin(), initializer_names_to_train.end());

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

  const std::vector<std::string>& expected_intermediate_tensor_names{"106",
                                                                     "123",
                                                                     "177",
                                                                     "147",
                                                                     "182",
                                                                     "176",
                                                                     "196",
                                                                     "212",
                                                                     "223",
                                                                     "239",
                                                                     "293",
                                                                     "263",
                                                                     "298",
                                                                     "292",
                                                                     "312",
                                                                     "328",
                                                                     "339",
                                                                     "355",
                                                                     "409",
                                                                     "379",
                                                                     "414",
                                                                     "408",
                                                                     "428",
                                                                     "444",
                                                                     "455",
                                                                     "471",
                                                                     "525",
                                                                     "495",
                                                                     "530",
                                                                     "524",
                                                                     "544",
                                                                     "560",
                                                                     "571",
                                                                     "587",
                                                                     "641",
                                                                     "611",
                                                                     "646",
                                                                     "640",
                                                                     "660",
                                                                     "676",
                                                                     "687",
                                                                     "703",
                                                                     "705",
                                                                     "707",
                                                                     "731",
                                                                     "730",
                                                                     "111",
                                                                     "saved_mean",
                                                                     "saved_inv_std_var",
                                                                     "200",
                                                                     "saved_mean_token_6",
                                                                     "saved_inv_std_var_token_7",
                                                                     "227",
                                                                     "saved_mean_token_9",
                                                                     "saved_inv_std_var_token_10",
                                                                     "316",
                                                                     "saved_mean_token_12",
                                                                     "saved_inv_std_var_token_13",
                                                                     "343",
                                                                     "saved_mean_token_15",
                                                                     "saved_inv_std_var_token_16",
                                                                     "432",
                                                                     "saved_mean_token_18",
                                                                     "saved_inv_std_var_token_19",
                                                                     "459",
                                                                     "saved_mean_token_21",
                                                                     "saved_inv_std_var_token_22",
                                                                     "548",
                                                                     "saved_mean_token_24",
                                                                     "saved_inv_std_var_token_25",
                                                                     "575",
                                                                     "saved_mean_token_27",
                                                                     "saved_inv_std_var_token_28",
                                                                     "664",
                                                                     "saved_mean_token_30",
                                                                     "saved_inv_std_var_token_31",
                                                                     "691",
                                                                     "saved_mean_token_33",
                                                                     "saved_inv_std_var_token_34",
                                                                     "718",
                                                                     "saved_mean_token_36",
                                                                     "saved_inv_std_var_token_37",
                                                                     "214",
                                                                     "330",
                                                                     "446",
                                                                     "562",
                                                                     "678",
                                                                     "709"};

  const std::vector<std::string>& expected_backward_user_input_names{"input_ids", "token_type_ids"};

  const std::vector<std::string>& expected_backward_trainable_initializer_names{
      "bert.embeddings.LayerNorm.weight",
      "bert.encoder.layer.0.attention.output.LayerNorm.weight",
      "bert.encoder.layer.0.attention.output.dense.weight_transposed",
      "bert.encoder.layer.0.attention.self.key.weight_transposed",
      "bert.encoder.layer.0.attention.self.query.weight_transposed",
      "bert.encoder.layer.0.attention.self.value.weight_transposed",
      "bert.encoder.layer.0.intermediate.dense.bias",
      "bert.encoder.layer.0.intermediate.dense.weight_transposed",
      "bert.encoder.layer.0.output.LayerNorm.weight",
      "bert.encoder.layer.0.output.dense.weight_transposed",
      "bert.encoder.layer.1.attention.output.LayerNorm.weight",
      "bert.encoder.layer.1.attention.output.dense.weight_transposed",
      "bert.encoder.layer.1.attention.self.key.weight_transposed",
      "bert.encoder.layer.1.attention.self.query.weight_transposed",
      "bert.encoder.layer.1.attention.self.value.weight_transposed",
      "bert.encoder.layer.1.intermediate.dense.bias",
      "bert.encoder.layer.1.intermediate.dense.weight_transposed",
      "bert.encoder.layer.1.output.LayerNorm.weight",
      "bert.encoder.layer.1.output.dense.weight_transposed",
      "bert.encoder.layer.2.attention.output.LayerNorm.weight",
      "bert.encoder.layer.2.attention.output.dense.weight_transposed",
      "bert.encoder.layer.2.attention.self.key.weight_transposed",
      "bert.encoder.layer.2.attention.self.query.weight_transposed",
      "bert.encoder.layer.2.attention.self.value.weight_transposed",
      "bert.encoder.layer.2.intermediate.dense.bias",
      "bert.encoder.layer.2.intermediate.dense.weight_transposed",
      "bert.encoder.layer.2.output.LayerNorm.weight",
      "bert.encoder.layer.2.output.dense.weight_transposed",
      "bert.encoder.layer.3.attention.output.LayerNorm.weight",
      "bert.encoder.layer.3.attention.output.dense.weight_transposed",
      "bert.encoder.layer.3.attention.self.key.weight_transposed",
      "bert.encoder.layer.3.attention.self.query.weight_transposed",
      "bert.encoder.layer.3.attention.self.value.weight_transposed",
      "bert.encoder.layer.3.intermediate.dense.bias",
      "bert.encoder.layer.3.intermediate.dense.weight_transposed",
      "bert.encoder.layer.3.output.LayerNorm.weight",
      "bert.encoder.layer.3.output.dense.weight_transposed",
      "bert.encoder.layer.4.attention.output.LayerNorm.weight",
      "bert.encoder.layer.4.attention.output.dense.weight_transposed",
      "bert.encoder.layer.4.attention.self.key.weight_transposed",
      "bert.encoder.layer.4.attention.self.query.weight_transposed",
      "bert.encoder.layer.4.attention.self.value.weight_transposed",
      "bert.encoder.layer.4.intermediate.dense.bias",
      "bert.encoder.layer.4.intermediate.dense.weight_transposed",
      "bert.encoder.layer.4.output.LayerNorm.weight",
      "bert.encoder.layer.4.output.dense.weight_transposed",
      "bert.pooler.dense.weight",
      "cls.predictions.transform.LayerNorm.weight",
      "cls.predictions.transform.dense.bias",
      "cls.predictions.transform.dense.weight_transposed",
      "cls.seq_relationship.weight"};

  const std::vector<std::string>& expected_backward_output_grad_names{"prediction_scores_grad",
                                                                      "seq_relationship_score_grad"};

  RunModuleGradientGraphBuilderTest(file_path, initializer_names_to_train, expected_forward_ops_count,
                                    expected_backward_ops_count, expected_intermediate_tensor_names,
                                    expected_backward_user_input_names, expected_backward_trainable_initializer_names,
                                    expected_backward_output_grad_names);
}
#endif

}  // namespace test
}  // namespace onnxruntime
