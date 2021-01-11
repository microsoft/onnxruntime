// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "orttraining/core/framework/module_gradient_graph_builder.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"

namespace onnxruntime {
namespace training {

using namespace onnxruntime::common;

Status ModuleGradientGraphBuilder::Initialize(std::istream& model_istream,
                                              const ModuleGradientGraphBuilderConfiguration& config) {
  // Save the model and config.
  ONNX_NAMESPACE::ModelProto model_proto;
  ORT_RETURN_IF_ERROR(Model::Load(model_istream, &model_proto));
  ORT_RETURN_IF_ERROR(Model::Load(model_proto, model_, nullptr, *logger_));
  config_ = config;

  // Handle original model inputs, outputs and trainable initializers.
  // We need to move the trainable initializers to graph inputs and keep the order in config,
  // it's possible that the graph already has some trainable initializers in graph inputs,
  // so we need to NOT assign these trainable initializers to the user inputs list.
  Graph& graph = model_->MainGraph();
  std::unordered_set<std::string> initializer_names_to_train_set(config.initializer_names_to_train.begin(),
                                                                 config.initializer_names_to_train.end());
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  for (auto& node_arg : graph_inputs) {
    if (initializer_names_to_train_set.find(node_arg->Name()) == initializer_names_to_train_set.end()) {
      split_graphs_info_.user_input_names.emplace_back(node_arg->Name());
    }
  }

  const std::vector<const NodeArg*>& graph_outputs = graph.GetOutputs();
  for (auto& node_arg : graph_outputs) {
    split_graphs_info_.user_output_names.emplace_back(node_arg->Name());
  }

  split_graphs_info_.initializer_names_to_train.assign(config.initializer_names_to_train.begin(),
                                                       config.initializer_names_to_train.end());

  std::vector<const NodeArg*> input_args;
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    input_args.emplace_back(graph.GetNodeArg(input_name));
  }

  // Remove the training initializers from the graph and move them to graph inputs.
  for (const auto& initializer_name : split_graphs_info_.initializer_names_to_train) {
    input_args.emplace_back(graph.GetNodeArg(initializer_name));
    graph.RemoveInitializedTensor(initializer_name);
  }

  graph.SetInputs(input_args);
  return Status::OK();
}

// Build the gradient graphs from foward graph and save it to backward graph.
// Since the input shapes may differ, and the graph optimizers (mainly constant folding) may fold this
// shape info to constants, so the optimized graph (before gradient graph building) can not be shared.
// So each time we need to start from the beginning, i.e., 1) replace input shapes, 2) apply graph optimizers,
// 3) build gradient graph to backward graph, and finally 4) adjust the graph inputs and outputs.
Status ModuleGradientGraphBuilder::Build(const std::vector<std::vector<int64_t>>* input_shapes_ptr) {
  // Make a copy of the original model as forward graph.
  auto model_proto = model_->ToProto();
  ORT_RETURN_IF_ERROR(Model::Load(model_proto, forward_model_, nullptr, *logger_));

  // Replace the user input shapes if input_shapes_ptr is not null_ptr.
  if (input_shapes_ptr) {
    SetConcreteInputShapes(*input_shapes_ptr);
  }

  // Build the gradient graph to backward graph.
  ORT_RETURN_IF_ERROR(BuildGradientGraph());

  // Adjust the graph inputs and outputs.
  SetForwardOutputsAndBackwardInputs();
  SetBackwardOutputs();

  return Status::OK();
}

std::string SerializeModel(const std::shared_ptr<onnxruntime::Model>& model, const std::string& tag) {
  std::string model_str;
  if (!model->ToProto().SerializeToString(&model_str)) {
    ORT_THROW("Fail to serialize", tag, "model to string.");
  }

  return model_str;
}

std::string ModuleGradientGraphBuilder::GetForwardModel() const { return SerializeModel(forward_model_, "forward"); }

std::string ModuleGradientGraphBuilder::GetBackwardModel() const { return SerializeModel(backward_model_, "backward"); }

void ModuleGradientGraphBuilder::SetConcreteInputShapes(const std::vector<std::vector<int64_t>> input_shapes) {
  ORT_ENFORCE(input_shapes.size() == split_graphs_info_.user_input_names.size(),
              "The size of concrete input shapes and the size of user inputs does not match.");
  Graph& forward_graph = forward_model_->MainGraph();
  std::vector<const NodeArg*> input_args;
  size_t input_index = 0;
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    NodeArg* input_node_arg = forward_graph.GetNodeArg(input_name);
    ONNX_NAMESPACE::TensorShapeProto new_shape;
    for (size_t i = 0; i < input_shapes[input_index].size(); i++) {
      new_shape.add_dim()->set_dim_value(input_shapes[input_index][i]);
    }

    input_node_arg->SetShape(new_shape);
    input_args.emplace_back(input_node_arg);
    input_index++;
  }

  // Move over all training initializer inputs. They already have the concrete shapes.
  const std::vector<const NodeArg*>& graph_inputs = forward_graph.GetInputsIncludingInitializers();
  for (; input_index < graph_inputs.size(); input_index++) {
    input_args.emplace_back(graph_inputs[input_index]);
  }

  forward_graph.SetInputs(input_args);
}

Status ModuleGradientGraphBuilder::BuildGradientGraph() {
  // Resolve forward graph, register and apply transformers for pre-training.
  Graph& forward_graph = forward_model_->MainGraph();
  ORT_RETURN_IF_ERROR(forward_graph.Resolve());

  const TrainingSession::TrainingConfiguration::GraphTransformerConfiguration graph_transformer_config{};
  GraphTransformerManager graph_transformation_mgr{2};
  std::unique_ptr<CPUExecutionProvider> cpu_execution_provider =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  std::unordered_set<std::string> x_node_arg_names;
  std::set_union(config_.initializer_names_to_train.begin(), config_.initializer_names_to_train.end(),
                 config_.input_names_require_grad.begin(), config_.input_names_require_grad.end(),
                 std::inserter(x_node_arg_names, x_node_arg_names.begin()));
  auto add_transformers = [&](TransformerLevel level) {
    auto transformers_to_register = transformer_utils::GeneratePreTrainingTransformers(
        level, x_node_arg_names, graph_transformer_config, *cpu_execution_provider);
    for (auto& entry : transformers_to_register) {
      graph_transformation_mgr.Register(std::move(entry), level);
    }
  };

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    if (TransformerLevel::MaxLevel >= level) {
      add_transformers(level);
    }
  }

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR(
        graph_transformation_mgr.ApplyTransformers(forward_graph, static_cast<TransformerLevel>(i), *logger_));
  }

  // Build gradient graph to backward graph.
  GradientGraphConfiguration gradient_graph_config{};
  gradient_graph_config.use_invertible_layernorm_grad = config_.use_invertible_layernorm_grad;
  gradient_graph_config.set_gradients_as_graph_outputs = true;
  std::unordered_set<std::string> y_node_arg_names(split_graphs_info_.user_output_names.begin(),
                                                   split_graphs_info_.user_output_names.end());
  GradientGraphBuilder grad_graph_builder(&forward_graph, y_node_arg_names, x_node_arg_names, "", gradient_graph_config,
                                          *logger_);

  // Create backward model, start from an empty one.
  backward_model_ = std::make_shared<Model>("backward_model", false, ModelMetaData(), PathString(),
                                            IOnnxRuntimeOpSchemaRegistryList(), forward_graph.DomainToVersionMap(),
                                            std::vector<ONNX_NAMESPACE::FunctionProto>(), *logger_);
  Graph& backward_graph = backward_model_->MainGraph();
  ORT_RETURN_IF_ERROR(grad_graph_builder.Build(nullptr, &backward_graph));

  // Apply transformers to backward graph.
  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR(
        graph_transformation_mgr.ApplyTransformers(backward_graph, static_cast<TransformerLevel>(i), *logger_));
  }

  return Status::OK();
}

void ModuleGradientGraphBuilder::SetForwardOutputsAndBackwardInputs() {
  Graph& forward_graph = forward_model_->MainGraph();
  Graph& backward_graph = backward_model_->MainGraph();

  split_graphs_info_.user_output_grad_names.clear();
  for (const auto& output_name : split_graphs_info_.user_output_names) {
    split_graphs_info_.user_output_grad_names.emplace_back(output_name + "_grad");
  }

  // Try to get all intermediate tensor names.
  std::unordered_set<std::string> non_intermediate_input_candidiates(split_graphs_info_.user_input_names.begin(),
                                                                     split_graphs_info_.user_input_names.end());
  non_intermediate_input_candidiates.insert(split_graphs_info_.initializer_names_to_train.begin(),
                                            split_graphs_info_.initializer_names_to_train.end());
  non_intermediate_input_candidiates.insert(split_graphs_info_.user_output_grad_names.begin(),
                                            split_graphs_info_.user_output_grad_names.end());

  const auto& forward_initializers = forward_graph.GetAllInitializedTensors();

  const std::vector<const NodeArg*>& backward_graph_inputs = backward_graph.GetInputsIncludingInitializers();
  std::unordered_map<std::string, const NodeArg*> non_intermediate_tensor_arg_map;
  split_graphs_info_.intermediate_tensor_names.clear();
  for (auto& node_arg : backward_graph_inputs) {
    const std::string& node_arg_name = node_arg->Name();
    if (non_intermediate_input_candidiates.find(node_arg_name) != non_intermediate_input_candidiates.end()) {
      non_intermediate_tensor_arg_map[node_arg_name] = node_arg;
    } else if (forward_initializers.find(node_arg_name) != forward_initializers.end()) {
      backward_graph.AddInitializedTensor(*forward_initializers.at(node_arg_name));
    } else {
      split_graphs_info_.intermediate_tensor_names.emplace_back(node_arg_name);
    }
  }

  // Forward outputs contains user outputs only. Need to add all intermediate tensors to forward outputs.
  const std::vector<const NodeArg*>& forward_graph_outputs = forward_graph.GetOutputs();
  std::vector<const NodeArg*> new_forward_output_args;
  for (auto& node_arg : forward_graph_outputs) {
    new_forward_output_args.emplace_back(node_arg);
  }

  for (const auto& intermediate_tensor_name : split_graphs_info_.intermediate_tensor_names) {
    new_forward_output_args.emplace_back(forward_graph.GetNodeArg(intermediate_tensor_name));
  }

  forward_graph.SetOutputs(new_forward_output_args);

  // Adjust the backward graph inputs by following order:
  // 1. user inputs if needed, with same order of user inputs,
  // 2. trainable initializers if needed, with same order of trainable initializers,
  // 3. intermediate tensors,
  // 4. user output gradients if needed, with same order of user outputs.
  std::vector<const NodeArg*> new_backward_input_args;
  split_graphs_info_.backward_user_input_names.clear();
  split_graphs_info_.backward_intializer_names_as_input.clear();
  split_graphs_info_.backward_output_grad_names.clear();
  for (const auto& user_input_name : split_graphs_info_.user_input_names) {
    if (non_intermediate_tensor_arg_map.find(user_input_name) != non_intermediate_tensor_arg_map.end()) {
      split_graphs_info_.backward_user_input_names.emplace_back(user_input_name);
      new_backward_input_args.emplace_back(non_intermediate_tensor_arg_map[user_input_name]);
    }
  }

  for (const auto& initializer_name_to_train : split_graphs_info_.initializer_names_to_train) {
    if (non_intermediate_tensor_arg_map.find(initializer_name_to_train) != non_intermediate_tensor_arg_map.end()) {
      split_graphs_info_.backward_intializer_names_as_input.emplace_back(initializer_name_to_train);
      new_backward_input_args.emplace_back(non_intermediate_tensor_arg_map[initializer_name_to_train]);
    }
  }

  for (const auto& intermediate_tensor_name : split_graphs_info_.intermediate_tensor_names) {
    new_backward_input_args.emplace_back(backward_graph.GetNodeArg(intermediate_tensor_name));
  }

  for (const auto& user_output_grad_name : split_graphs_info_.user_output_grad_names) {
    if (non_intermediate_tensor_arg_map.find(user_output_grad_name) != non_intermediate_tensor_arg_map.end()) {
      split_graphs_info_.backward_output_grad_names.emplace_back(user_output_grad_name);
      new_backward_input_args.emplace_back(non_intermediate_tensor_arg_map[user_output_grad_name]);
    }
  }

  backward_graph.SetInputs(new_backward_input_args);
}

void ModuleGradientGraphBuilder::SetBackwardOutputs() {
  // Adjust backward graph outputs by the following order:
  // 1. user input grads if required, with same order of user inputs,
  // 2. trainable initailizer grads, with same order of trainable initializers.
  Graph& backward_graph = backward_model_->MainGraph();
  const std::vector<const NodeArg*>& backward_graph_outputs = backward_graph.GetOutputs();
  std::unordered_map<std::string, const NodeArg*> backward_output_arg_map;
  for (auto& node_arg : backward_graph_outputs) {
    backward_output_arg_map[node_arg->Name()] = node_arg;
  }

  std::unordered_set<std::string> user_input_require_grad_set(config_.input_names_require_grad.begin(),
                                                              config_.input_names_require_grad.end());

  std::vector<const NodeArg*> new_backward_output_args;
  split_graphs_info_.user_input_grad_names.clear();
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    if (user_input_require_grad_set.find(input_name) != user_input_require_grad_set.end()) {
      std::string input_gradient_name = input_name + "_grad";
      ORT_ENFORCE(backward_output_arg_map.find(input_gradient_name) != backward_output_arg_map.end(),
                  "Required user input grad is not found on gradient graph.");
      split_graphs_info_.user_input_grad_names[input_name] = input_gradient_name;
      new_backward_output_args.emplace_back(backward_output_arg_map[input_gradient_name]);
    }
  }

  // Add initializer gradients to graph outputs.
  split_graphs_info_.initializer_grad_names_to_train.clear();
  for (const auto& initializer_name : split_graphs_info_.initializer_names_to_train) {
    std::string initializer_gradient_name = initializer_name + "_grad";
    ORT_ENFORCE(backward_output_arg_map.find(initializer_gradient_name) != backward_output_arg_map.end(),
                "Trainable initializer grad is not found on gradient graph.");
    split_graphs_info_.initializer_grad_names_to_train.emplace_back(initializer_gradient_name);
    new_backward_output_args.emplace_back(backward_output_arg_map[initializer_gradient_name]);
  }

  backward_graph.SetOutputs(new_backward_output_args);
}

}  // namespace training
}  // namespace onnxruntime
