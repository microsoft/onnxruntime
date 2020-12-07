// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/graph/graph_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "orttraining/core/framework/module_gradient_graph_builder.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"

namespace onnxruntime {
namespace training {

using namespace onnxruntime::common;

void GetInputAndOutputNames(const Node& node, std::unordered_set<std::string>& input_names,
                            std::unordered_set<std::string>& output_names) {
  std::for_each(node.InputDefs().begin(), node.InputDefs().end(),
                [&input_names](const NodeArg* node_arg) { input_names.insert(node_arg->Name()); });
  std::for_each(node.OutputDefs().begin(), node.OutputDefs().end(),
                [&output_names](const NodeArg* node_arg) { output_names.insert(node_arg->Name()); });
}

void RemoveNodes(Graph& graph, const std::vector<Node*>& nodes_to_remove) {
  for (Node* node_to_remove : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, *node_to_remove);
    graph.RemoveNode(node_to_remove->Index());
  }
}

void FilterInitializers(Graph& graph, const std::unordered_set<std::string>& input_names) {
  const auto& initializers = graph.GetAllInitializedTensors();
  std::unordered_set<std::string> initializer_names_to_remove;
  for (const auto& initializer : initializers) {
    if (input_names.find(initializer.first) == input_names.end()) {
      initializer_names_to_remove.insert(initializer.first);
    }
  }

  for (const auto& initializer_name : initializer_names_to_remove) {
    graph.RemoveInitializedTensor(initializer_name);
  }
}

Status ModuleGradientGraphBuilder::Initialize(std::istream& model_istream,
                                              const ModuleGradientGraphBuilderConfiguration& config) {
  // We need to apply the pre-training transformers before the gradient graph builder so we can build
  // an optimized gradient graph. The constant folding transformer depends on concrete shapes, without
  // constant folding with concrete shapes, shapes of some intermediate tensors will fail to infer.
  // This means we need to "apply transformers -> build gradient graph -> split" each time we have different
  // concrete input shapes. So this init func is just to save the original graph and config.
  ONNX_NAMESPACE::ModelProto model_proto;
  ORT_RETURN_IF_ERROR(Model::Load(model_istream, &model_proto));
  ORT_RETURN_IF_ERROR(Model::Load(model_proto, model_, nullptr, *logger_));

  // Handle original model inputs, outputs and trainable initializers.
  const std::vector<const NodeArg*>& graph_inputs = model_->MainGraph().GetInputsIncludingInitializers();
  for (auto& node_arg : graph_inputs) {
    split_graphs_info_.user_input_names.emplace_back(node_arg->Name());
  }

  const std::vector<const NodeArg*>& graph_outputs = model_->MainGraph().GetOutputs();
  for (auto& node_arg : graph_outputs) {
    split_graphs_info_.user_output_names.emplace_back(node_arg->Name());
  }

  split_graphs_info_.initializer_names_to_train.assign(config.initializer_names_to_train.begin(),
                                                       config.initializer_names_to_train.end());

  config_ = config;
  return Status::OK();
}

Status ModuleGradientGraphBuilder::BuildAndSplit(const std::vector<std::vector<int64_t>>& input_shapes) {
  // Make a copy of the original model.
  auto model_proto = model_->ToProto();
  std::shared_ptr<onnxruntime::Model> model_copied;
  ORT_RETURN_IF_ERROR(Model::Load(model_proto, model_copied, nullptr, *logger_));
  Graph& graph = model_copied->MainGraph();

  // Replace the input shapes.
  std::vector<const NodeArg*> input_args;
  size_t input_index = 0;
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    NodeArg* input_node_arg = graph.GetNodeArg(input_name);
    ONNX_NAMESPACE::TensorShapeProto new_shape;
    for (size_t i = 0; i < input_shapes[input_index].size(); i++) {
      new_shape.add_dim()->set_dim_value(input_shapes[input_index][i]);
    }

    input_node_arg->SetShape(new_shape);
    input_args.emplace_back(input_node_arg);
    input_index++;
  }

  graph.SetInputs(input_args);
  ORT_RETURN_IF_ERROR(graph.Resolve());

  // Register and apply transformers for pre-training.
  const TrainingSession::TrainingConfiguration::GraphTransformerConfiguration graph_transformer_config{};
  GraphTransformerManager graph_transformation_mgr{2};
  std::unique_ptr<CPUExecutionProvider> cpu_execution_provider =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  std::unordered_set<std::string> x_node_arg_names;
  std::set_union(config_.initializer_names_to_train.begin(), config_.initializer_names_to_train.end(),
                 config_.input_names_require_grad.begin(), config_.input_names_require_grad.end(),
                 std::inserter(x_node_arg_names, x_node_arg_names.begin()));
  auto add_transformers = [&](TransformerLevel level) {
    std::unordered_map<std::string, std::string> updated_weight_names{};
    auto transformers_to_register = transformer_utils::GeneratePreTrainingTransformers(
        level, x_node_arg_names, graph_transformer_config, *cpu_execution_provider, updated_weight_names, {});
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
    ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, static_cast<TransformerLevel>(i), *logger_));
  }

  // Build gradient graph.
  GradientGraphConfiguration gradient_graph_config{};
  gradient_graph_config.use_invertible_layernorm_grad = config_.use_invertible_layernorm_grad;
  gradient_graph_config.set_gradients_as_graph_outputs = config_.set_gradients_as_graph_outputs;
  std::unordered_set<std::string> y_node_arg_names(split_graphs_info_.user_output_names.begin(),
                                                   split_graphs_info_.user_output_names.end());
  GradientGraphBuilder grad_graph_builder(&graph, y_node_arg_names, x_node_arg_names,
                                          "",
                                          gradient_graph_config, *logger_);
  ORT_RETURN_IF_ERROR(grad_graph_builder.Build());

  // Fix inputs/outputs related to gradients.
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::unordered_set<std::string> input_names;
  std::unordered_set<std::string> output_names;
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    GetInputAndOutputNames(node, input_names, output_names);
  }

  input_args.clear();
  for (auto& input_name : split_graphs_info_.user_input_names) {
    input_args.emplace_back(graph.GetNodeArg(input_name));
  }

  // Add the entry points of gradients (normally loss_gard) to the graph inputs. Using the order of graph outputs.
  split_graphs_info_.user_output_grad_names.clear();
  split_graphs_info_.backward_output_grad_names.clear();
  for (const auto& output_name : split_graphs_info_.user_output_names) {
    std::string output_gradient_name = output_name + "_grad";
    if (input_names.find(output_gradient_name) != input_names.end()) {
      split_graphs_info_.user_output_grad_names.emplace_back(output_gradient_name);
      // Only add to graph input when it's not an output of a node.
      if (output_names.find(output_gradient_name) == output_names.end()) {
        split_graphs_info_.backward_output_grad_names.emplace_back(output_gradient_name);
        NodeArg* output_gradient_node_arg = graph.GetNodeArg(output_gradient_name);
        output_gradient_node_arg->UpdateTypeAndShape(*graph.GetNodeArg(output_name), true, true, *logger_);
        input_args.emplace_back(output_gradient_node_arg);
      }
    }
  }

  graph.SetInputs(input_args);

  std::vector<const NodeArg*> output_args;
  for (auto& output_name : split_graphs_info_.user_output_names) {
    output_args.emplace_back(graph.GetNodeArg(output_name));
  }

  // Add initializer gradients to graph outputs.
  split_graphs_info_.initializer_grad_names_to_train.clear();
  for (const auto& initializer_name : split_graphs_info_.initializer_names_to_train) {
    std::string initializer_gradient_name = initializer_name + "_grad";
    if (output_names.find(initializer_gradient_name) != output_names.end()) {
      split_graphs_info_.initializer_grad_names_to_train.emplace_back(initializer_gradient_name);
      output_args.emplace_back(graph.GetNodeArg(initializer_gradient_name));
    }
  }

  // Add input gradients to graph outputs if it's required.
  for (const auto& input_name : config_.input_names_require_grad) {
    std::string input_gradient_name = input_name + "_grad";
    if (output_names.find(input_gradient_name) != output_names.end()) {
      output_args.emplace_back(graph.GetNodeArg(input_gradient_name));
    }
  }

  graph.SetOutputs(output_args);
  graph.Resolve();

  // Run the transformers again mainly for backward part, e.g., constant fold from those Shape nodes in backward graph.
  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, static_cast<TransformerLevel>(i), *logger_));
  }

  // Create two copies of gradient model for forward and backward models respectively.
  auto gradient_model_proto = model_copied->ToProto();
  ORT_RETURN_IF_ERROR(Model::Load(gradient_model_proto, forward_model_, nullptr, *logger_));
  ORT_RETURN_IF_ERROR(Model::Load(gradient_model_proto, backward_model_, nullptr, *logger_));

  // Split the graph in the copies of gradient model.
  ORT_RETURN_IF_ERROR(Split());

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

Status ModuleGradientGraphBuilder::Split() {
  // Get forward model, also collect some information for backward model generation.
  Graph& forward_graph = forward_model_->MainGraph();
  GraphViewer forward_graph_viewer(forward_graph);
  const auto& forward_node_topology_list = forward_graph_viewer.GetNodesInTopologicalOrder();
  std::vector<Node*> forward_nodes_to_remove;
  std::unordered_set<std::string> forward_input_names;
  std::unordered_set<std::string> forward_output_names;
  std::unordered_set<std::string> backward_input_names;
  std::unordered_set<std::string> backward_output_names;
  for (auto node_index : forward_node_topology_list) {
    auto& node = *forward_graph.GetNode(node_index);
    // Currently we are using node description to distinguish the forward and backward nodes.
    if (node.Description() == "Backward pass") {
      forward_nodes_to_remove.emplace_back(&node);
      GetInputAndOutputNames(node, backward_input_names, backward_output_names);
    } else {
      GetInputAndOutputNames(node, forward_input_names, forward_output_names);
    }
  }

  std::unordered_set<std::string> intermediate_arg_names;
  for (const auto& forward_output_name : forward_output_names) {
    if (backward_input_names.find(forward_output_name) != backward_input_names.end()) {
      intermediate_arg_names.insert(forward_output_name);
    }
  }

  RemoveNodes(forward_graph, forward_nodes_to_remove);
  FilterInitializers(forward_graph, forward_input_names);

  // All user inputs should be also part of the forward graph inputs.
  std::vector<const NodeArg*> forward_input_args;
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    forward_input_args.emplace_back(forward_graph.GetNodeArg(input_name));
  }

  // Add initializers to forward graph inputs.
  for (const auto& initializer_name : split_graphs_info_.initializer_names_to_train) {
    forward_input_args.emplace_back(forward_graph.GetNodeArg(initializer_name));
    forward_graph.RemoveInitializedTensor(initializer_name);
  }

  forward_graph.SetInputs(forward_input_args);

  // All user outputs should be also part of the forward graph outputs.
  std::vector<const NodeArg*> forward_output_args;
  for (const auto& output_name : split_graphs_info_.user_output_names) {
    forward_output_args.emplace_back(forward_graph.GetNodeArg(output_name));
  }

  // Add intermediate args to forward graph outputs.
  split_graphs_info_.intermediate_tensor_names.clear();
  for (const auto& intermediate_arg_name : intermediate_arg_names) {
    // Ignore the user outputs.
    if (std::find(split_graphs_info_.user_output_names.begin(), split_graphs_info_.user_output_names.end(),
                  intermediate_arg_name) == split_graphs_info_.user_output_names.end()) {
      split_graphs_info_.intermediate_tensor_names.emplace_back(intermediate_arg_name);
      forward_output_args.emplace_back(forward_graph.GetNodeArg(intermediate_arg_name));
    }
  }

  forward_graph.SetOutputs(forward_output_args);
  forward_graph.Resolve();

  // Get backward graph.
  Graph& backward_graph = backward_model_->MainGraph();
  GraphViewer backward_graph_viewer(backward_graph);
  const auto& backward_node_topology_list = backward_graph_viewer.GetNodesInTopologicalOrder();
  std::vector<Node*> backward_nodes_to_remove;
  for (auto node_index : backward_node_topology_list) {
    auto& node = *backward_graph.GetNode(node_index);
    if (node.Description() != "Backward pass") {
      backward_nodes_to_remove.emplace_back(&node);
    }
  }

  RemoveNodes(backward_graph, backward_nodes_to_remove);

  // User inputs to backward graph inputs.
  split_graphs_info_.backward_user_input_names.clear();
  std::vector<const NodeArg*> backward_input_args;
  for (const auto& input_name : split_graphs_info_.user_input_names) {
    // Only takes those in the backward inputs.
    if (backward_input_names.find(input_name) != backward_input_names.end()) {
      split_graphs_info_.backward_user_input_names.emplace_back(input_name);
      backward_input_args.emplace_back(backward_graph.GetNodeArg(input_name));
    }
  }

  // Add initializer args to backward graph inputs if any node uses them.
  split_graphs_info_.backward_intializer_names_as_input.clear();
  for (const auto& initializer_name : split_graphs_info_.initializer_names_to_train) {
    // Some initializers will be inputs for backward graph.
    if (backward_input_names.find(initializer_name) != backward_input_names.end()) {
      split_graphs_info_.backward_intializer_names_as_input.emplace_back(initializer_name);
      backward_input_args.emplace_back(backward_graph.GetNodeArg(initializer_name));
      backward_graph.RemoveInitializedTensor(initializer_name);
    }
  }

  // Add intermediate args to backward graph inputs.
  for (const auto& intermediate_arg_name : split_graphs_info_.intermediate_tensor_names) {
    NodeArg* intermediate_node_arg = backward_graph.GetNodeArg(intermediate_arg_name);
    intermediate_node_arg->UpdateTypeAndShape(*forward_graph.GetNodeArg(intermediate_arg_name), true, true, *logger_);
    backward_input_args.emplace_back(intermediate_node_arg);
  }

  // Grad of user outputs to backward graph inputs.
  for (const auto& output_grad_name : split_graphs_info_.backward_output_grad_names) {
    backward_input_args.emplace_back(backward_graph.GetNodeArg(output_grad_name));
  }

  backward_graph.SetInputs(backward_input_args);

  // Exclude user outputs from the backward graph.
  const std::vector<const NodeArg*>& backward_graph_outputs = backward_graph.GetOutputs();
  std::vector<const NodeArg*> backward_output_args;
  for (auto& node_arg : backward_graph_outputs) {
    if (backward_output_names.find(node_arg->Name()) != backward_output_names.end()) {
      backward_output_args.emplace_back(node_arg);
    }
  }

  backward_graph.SetOutputs(backward_output_args);
  FilterInitializers(backward_graph, backward_input_names);
  backward_graph.Resolve();
  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
