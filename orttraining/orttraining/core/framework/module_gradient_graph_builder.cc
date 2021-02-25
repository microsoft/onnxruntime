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
      training_graph_info_.user_input_names.emplace_back(node_arg->Name());
    }
  }

  const std::vector<const NodeArg*>& graph_outputs = graph.GetOutputs();
  for (auto& node_arg : graph_outputs) {
    training_graph_info_.user_output_names.emplace_back(node_arg->Name());
  }

  training_graph_info_.initializer_names_to_train.assign(config.initializer_names_to_train.begin(),
                                                       config.initializer_names_to_train.end());

  std::vector<const NodeArg*> input_args;
  for (const auto& input_name : training_graph_info_.user_input_names) {
    input_args.emplace_back(graph.GetNodeArg(input_name));
  }

  // Remove the training initializers from the graph and move them to graph inputs.
  for (const auto& initializer_name : training_graph_info_.initializer_names_to_train) {
    const NodeArg* node_arg = graph.GetNodeArg(initializer_name);
    ORT_ENFORCE(node_arg != nullptr);
    input_args.emplace_back(node_arg);
    graph.RemoveInitializedTensor(initializer_name);
  }

  graph.SetInputs(input_args);
  return Status::OK();
}

// Build the gradient graphs from original graph.
// Since the input shapes may differ, and the graph optimizers (mainly constant folding) may fold this
// shape info to constants, the optimized graph (before gradient graph building) can not be shared.
// So each time we need to start from the beginning, i.e., 1) replace input shapes, 2) apply graph optimizers,
// 3) build gradient graph, and finally 4) adjust the graph inputs and outputs.
Status ModuleGradientGraphBuilder::Build(const std::vector<std::vector<int64_t>>* input_shapes_ptr) {
  // Make a copy of the original model.
  auto model_proto = model_->ToProto();
  ORT_RETURN_IF_ERROR(Model::Load(model_proto, gradient_model_, nullptr, *logger_));

  // Replace the user input shapes if input_shapes_ptr is not null_ptr.
  if (input_shapes_ptr) {
    SetConcreteInputShapes(*input_shapes_ptr);
  }

  // Build the gradient graph.
  ORT_RETURN_IF_ERROR(BuildGradientGraph());

  // Add Yield Op.
  AddYieldOp();

  // Reorder outputs.
  ReorderOutputs();

  return Status::OK();
}

std::string ModuleGradientGraphBuilder::GetGradientModel() const {
  std::string model_str;
  if (!gradient_model_->ToProto().SerializeToString(&model_str)) {
    ORT_THROW("Fail to serialize gradient model to string.");
  }

  return model_str;
}

void ModuleGradientGraphBuilder::SetConcreteInputShapes(const std::vector<std::vector<int64_t>>& input_shapes) {
  ORT_ENFORCE(input_shapes.size() == training_graph_info_.user_input_names.size(),
              "The size of concrete input shapes and the size of user inputs does not match.");
  Graph& gradient_graph = gradient_model_->MainGraph();
  std::vector<const NodeArg*> input_args;
  size_t input_index = 0;
  for (const auto& input_name : training_graph_info_.user_input_names) {
    NodeArg* input_node_arg = gradient_graph.GetNodeArg(input_name);
    ONNX_NAMESPACE::TensorShapeProto new_shape;
    for (size_t i = 0; i < input_shapes[input_index].size(); i++) {
      new_shape.add_dim()->set_dim_value(input_shapes[input_index][i]);
    }

    input_node_arg->SetShape(new_shape);
    input_args.emplace_back(input_node_arg);
    input_index++;
  }

  // Move over all training initializer inputs. They already have the concrete shapes.
  const std::vector<const NodeArg*>& graph_inputs = gradient_graph.GetInputsIncludingInitializers();
  for (; input_index < graph_inputs.size(); input_index++) {
    input_args.emplace_back(graph_inputs[input_index]);
  }

  gradient_graph.SetInputs(input_args);
}

Status ModuleGradientGraphBuilder::BuildGradientGraph() {
  // Resolve original graph, register and apply transformers for pre-training.
  Graph& gradient_graph = gradient_model_->MainGraph();
  ORT_RETURN_IF_ERROR(gradient_graph.Resolve());

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
        graph_transformation_mgr.ApplyTransformers(gradient_graph, static_cast<TransformerLevel>(i), *logger_));
  }

  // Build gradient graph.
  GradientGraphConfiguration gradient_graph_config{};
  gradient_graph_config.use_invertible_layernorm_grad = config_.use_invertible_layernorm_grad;
  gradient_graph_config.set_gradients_as_graph_outputs = true;
  std::unordered_set<std::string> y_node_arg_names(training_graph_info_.user_output_names.begin(),
                                                   training_graph_info_.user_output_names.end());
  GradientGraphBuilder grad_graph_builder(&gradient_graph, y_node_arg_names, x_node_arg_names, "",
                                          gradient_graph_config, *logger_);

  ORT_RETURN_IF_ERROR(grad_graph_builder.Build());
  return Status::OK();
}

void ModuleGradientGraphBuilder::AddYieldOp() {
  Graph& gradient_graph = gradient_model_->MainGraph();
  GraphViewer gradient_graph_viewer(gradient_graph);
  const auto& gradient_node_topology_list = gradient_graph_viewer.GetNodesInTopologicalOrder();
  std::unordered_set<std::string> user_output_grad_names_set;
  for (const auto& name : training_graph_info_.user_output_names) {
    user_output_grad_names_set.insert(name + "_grad");
  }

  // If an NodeArg is output of one of nodes, it's not the user output gradient needed by backward graph.
  std::unordered_set<std::string> non_backward_user_output_grad_names;
  for (auto node_index : gradient_node_topology_list) {
    auto& node = *gradient_graph.GetNode(node_index);
    for (const auto& node_arg : node.OutputDefs()) {
      if (user_output_grad_names_set.find(node_arg->Name()) != user_output_grad_names_set.end()) {
        non_backward_user_output_grad_names.insert(node_arg->Name());
      }
    }
  }

  // YieldOps required_grad attribute specifies the indices of the required gradients.
  ONNX_NAMESPACE::AttributeProto required_grad;
  const std::string attribute_name = "required_grad";
  required_grad.set_name(attribute_name);
  required_grad.set_type(ONNX_NAMESPACE::AttributeProto::INTS);

  training_graph_info_.backward_output_grad_names_map.clear();
  for (std::size_t i = 0; i < training_graph_info_.user_output_names.size(); ++i) {
    const auto& name = training_graph_info_.user_output_names[i];
    std::string grad_name = name + "_grad";
    if (non_backward_user_output_grad_names.find(grad_name) == non_backward_user_output_grad_names.end()) {
      training_graph_info_.backward_output_grad_names_map.insert(std::make_pair(grad_name, i));
      required_grad.add_ints(static_cast<int64_t>(i));
    }
  }

  std::vector<NodeArg*> yield_input_node_args;
  std::vector<NodeArg*> yield_output_node_args;
  for (const auto& name : training_graph_info_.user_output_names) {
    yield_input_node_args.emplace_back(gradient_graph.GetNodeArg(name));
  }

  for (const auto& element : training_graph_info_.backward_output_grad_names_map) {
    yield_output_node_args.emplace_back(gradient_graph.GetNodeArg(element.first));
  }

  NodeAttributes attributes({{attribute_name, required_grad}});

  gradient_graph.AddNode("YieldOp", "YieldOp", "Yield Op", yield_input_node_args, yield_output_node_args, &attributes, kMSDomain);
}

void ModuleGradientGraphBuilder::ReorderOutputs() {
  // Adjust gradient graph outputs by the following order:
  // 1. user input grads if required, with same order of user inputs,
  // 2. trainable initailizer grads, with same order of trainable initializers.
  Graph& gradient_graph = gradient_model_->MainGraph();
  const std::vector<const NodeArg*>& gradient_graph_outputs = gradient_graph.GetOutputs();
  std::unordered_map<std::string, const NodeArg*> gradient_output_arg_map;
  for (auto& node_arg : gradient_graph_outputs) {
    gradient_output_arg_map[node_arg->Name()] = node_arg;
  }

  std::unordered_set<std::string> user_input_require_grad_set(config_.input_names_require_grad.begin(),
                                                              config_.input_names_require_grad.end());

  std::vector<const NodeArg*> new_output_args;
  training_graph_info_.user_input_grad_names.clear();
  for (const auto& input_name : training_graph_info_.user_input_names) {
    if (user_input_require_grad_set.find(input_name) != user_input_require_grad_set.end()) {
      std::string input_gradient_name = input_name + "_grad";
      ORT_ENFORCE(gradient_output_arg_map.find(input_gradient_name) != gradient_output_arg_map.end(),
                  "Required user input grad is not found on gradient graph.");
      training_graph_info_.user_input_grad_names[input_name] = input_gradient_name;
      new_output_args.emplace_back(gradient_output_arg_map[input_gradient_name]);
    }
  }

  // Add initializer gradients to graph outputs.
  training_graph_info_.initializer_grad_names_to_train.clear();
  for (const auto& initializer_name : training_graph_info_.initializer_names_to_train) {
    std::string initializer_gradient_name = initializer_name + "_grad";
    ORT_ENFORCE(gradient_output_arg_map.find(initializer_gradient_name) != gradient_output_arg_map.end(),
                "Trainable initializer grad is not found on gradient graph.");
    training_graph_info_.initializer_grad_names_to_train.emplace_back(initializer_gradient_name);
    new_output_args.emplace_back(gradient_output_arg_map[initializer_gradient_name]);
  }

  gradient_graph.SetOutputs(new_output_args);
}

}  // namespace training
}  // namespace onnxruntime
