// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "orttraining/core/framework/ortmodule_graph_builder.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"

namespace onnxruntime {
namespace training {

using namespace onnxruntime::common;

Status OrtModuleGraphBuilder::Initialize(std::istream& model_istream,
                                         const OrtModuleGraphBuilderConfiguration& config) {
  // Save the model and config.
  ONNX_NAMESPACE::ModelProto model_proto;
  ORT_RETURN_IF_ERROR(Model::Load(model_istream, &model_proto));
  ORT_RETURN_IF_ERROR(Model::Load(model_proto, original_model_, nullptr, *logger_));
  config_ = config;

  // Handle original model inputs, outputs and trainable initializers.
  // We need to move all the initializers to graph inputs and keep the order in config,
  // it's possible that the graph already has some initializers in graph inputs,
  // so we need to NOT assign these initializers to the user inputs list.
  Graph& graph = original_model_->MainGraph();
  std::unordered_set<std::string> initializer_names(config.initializer_names.begin(), config.initializer_names.end());
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  for (auto& node_arg : graph_inputs) {
    if (initializer_names.find(node_arg->Name()) == initializer_names.end()) {
      graph_info_.user_input_names.emplace_back(node_arg->Name());
    }
  }

  const std::vector<const NodeArg*>& graph_outputs = graph.GetOutputs();
  for (auto& node_arg : graph_outputs) {
    graph_info_.user_output_names.emplace_back(node_arg->Name());
  }

  graph_info_.initializer_names_to_train.assign(config.initializer_names_to_train.begin(),
                                                config.initializer_names_to_train.end());
  graph_info_.initializer_names.assign(config.initializer_names.begin(), config.initializer_names.end());

  std::vector<const NodeArg*> input_args;
  for (const auto& input_name : graph_info_.user_input_names) {
    input_args.emplace_back(graph.GetNodeArg(input_name));
  }

  // Remove all the initializers from the graph and move them to graph inputs.
  for (const auto& initializer_name : config_.initializer_names) {
    const NodeArg* node_arg = graph.GetNodeArg(initializer_name);
    ORT_ENFORCE(node_arg != nullptr, "node arg is nullptr for initializer name: ", initializer_name);

    input_args.emplace_back(node_arg);
    graph.RemoveInitializedTensor(initializer_name);
  }

  graph.SetInputs(input_args);
  logging::LoggingManager::SetDefaultLoggerSeverity(config_.loglevel);
  return Status::OK();
}

// Build the inference/gradient graphs from original graph.
// Since the input shapes may differ, and the graph optimizers (mainly constant folding) may fold this
// shape info to constants, the optimized graph (before gradient graph building) can not be shared.
// So each time we need to start from the beginning, i.e., 1) replace input shapes, 2) apply graph optimizers,
// 3) build gradient graph, and finally 4) adjust the graph inputs and outputs.
Status OrtModuleGraphBuilder::Build(const TrainingGraphTransformerConfiguration& pre_grad_graph_transformer_config,
                                    const std::vector<std::vector<int64_t>>* input_shapes_ptr) {
  // Make a copy of the original model.
  auto original_model_proto = original_model_->ToProto();
  ORT_RETURN_IF_ERROR(Model::Load(original_model_proto, forward_model_, nullptr, *logger_));

  // Replace the user input shapes if input_shapes_ptr is not null_ptr.
  if (input_shapes_ptr) {
    ORT_RETURN_IF_ERROR(SetConcreteInputShapes(*input_shapes_ptr));
  }

  // If this graph will be used only for inferencing, stop right here.
  // No need to apply the optimizations for training or build a gradient graph.
  if (!config_.build_gradient_graph) {
    return Status::OK();
  }

  // Optimize the forward graph and then build the gradient graph.
  std::unordered_set<std::string> x_node_arg_names;
  ORT_RETURN_IF_ERROR(OptimizeForwardGraph(pre_grad_graph_transformer_config, x_node_arg_names));
  ORT_RETURN_IF_ERROR(BuildGradientGraph(x_node_arg_names));

  if (config_.enable_caching) {
    GetFrontierTensors();
  }

  // Handle user outputs and output grads.
  HandleOutputsAndGrads();

  // Reorder outputs.
  ReorderOutputs();

  // Find module outputs needed for backward computation
  FindModuleOutputNeededForBackward();

  return Status::OK();
}

std::string OrtModuleGraphBuilder::GetGradientModel() const {
  std::string model_str;
  if (!gradient_model_->ToProto().SerializeToString(&model_str)) {
    ORT_THROW("Fail to serialize gradient model to string.");
  }
  return model_str;
}

std::string OrtModuleGraphBuilder::GetForwardModel() const {
  std::string model_str;
  if (!forward_model_->ToProto().SerializeToString(&model_str)) {
    ORT_THROW("Fail to serialize forward model to string.");
  }
  return model_str;
}

Status OrtModuleGraphBuilder::SetConcreteInputShapes(const std::vector<std::vector<int64_t>>& input_shapes) {
  ORT_ENFORCE(input_shapes.size() == graph_info_.user_input_names.size(),
              "The size of concrete input shapes and the size of user inputs does not match.");
  Graph& forward_graph = forward_model_->MainGraph();
  std::vector<const NodeArg*> input_args;
  size_t input_index = 0;
  for (const auto& input_name : graph_info_.user_input_names) {
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
  return forward_graph.Resolve();
}

Status OrtModuleGraphBuilder::OptimizeForwardGraph(const TrainingGraphTransformerConfiguration& config,
                                                   std::unordered_set<std::string>& x_node_arg_names) {
  // Resolve original graph, register and apply transformers for pre-training.
  Graph& forward_graph = forward_model_->MainGraph();
  ORT_RETURN_IF_ERROR(forward_graph.Resolve());

  GraphTransformerManager graph_transformation_mgr{3};
  std::unique_ptr<CPUExecutionProvider> cpu_execution_provider =
      std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  std::set_union(config_.initializer_names_to_train.begin(), config_.initializer_names_to_train.end(),
                 config_.input_names_require_grad.begin(), config_.input_names_require_grad.end(),
                 std::inserter(x_node_arg_names, x_node_arg_names.begin()));
  auto add_transformers = [&](TransformerLevel level) {
    auto transformers_to_register = transformer_utils::GeneratePreTrainingTransformers(
        level, x_node_arg_names, config, *cpu_execution_provider);
    for (auto& entry : transformers_to_register) {
      ORT_RETURN_IF_ERROR(graph_transformation_mgr.Register(std::move(entry), level));
    }
    return Status::OK();
  };

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    if (TransformerLevel::MaxLevel >= level) {
      ORT_RETURN_IF_ERROR(add_transformers(level));
    }
  }

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR(
        graph_transformation_mgr.ApplyTransformers(forward_graph, static_cast<TransformerLevel>(i), *logger_));
  }

  if (!config.optimized_pre_grad_filepath.empty()) {
    ORT_RETURN_IF_ERROR(Model::Save(*forward_model_, config.optimized_pre_grad_filepath));
  }

  return Status::OK();
}

Status OrtModuleGraphBuilder::BuildGradientGraph(const std::unordered_set<std::string>& x_node_arg_names) {
  // Copy the forward graph to create the gradient graph.
  ORT_RETURN_IF_ERROR(Model::Load(forward_model_->ToProto(), gradient_model_, nullptr, *logger_));
  Graph& gradient_graph = gradient_model_->MainGraph();

  // Build gradient graph.
  GradientGraphConfiguration gradient_graph_config{};
  gradient_graph_config.use_memory_efficient_gradient = config_.use_memory_efficient_gradient;
  gradient_graph_config.set_gradients_as_graph_outputs = true;
  std::unordered_set<std::string> y_node_arg_names(graph_info_.user_output_names.begin(),
                                                   graph_info_.user_output_names.end());
  GradientGraphBuilder grad_graph_builder(&gradient_graph, y_node_arg_names, x_node_arg_names, "",
                                          gradient_graph_config, *logger_);

  const std::unordered_set<std::string>& non_differentiable_output_names =
      grad_graph_builder.GetNonDifferentiableYNodeArgNames();
  for (size_t i = 0; i < graph_info_.user_output_names.size(); ++i) {
    if (non_differentiable_output_names.count(graph_info_.user_output_names[i]) > 0) {
      graph_info_.output_grad_indices_non_differentiable.emplace_back(i);
    }
  }

  ORT_RETURN_IF_ERROR(grad_graph_builder.Build());

  UpdatePythonOpInputsRequireGradInfo(grad_graph_builder.GetPythonOpInputRequireGradInfo());

  return Status::OK();
}

void OrtModuleGraphBuilder::GetFrontierTensors() {
  const Graph& graph = gradient_model_->MainGraph();
  for (const auto& param : graph_info_.initializer_names_to_train) {
    std::vector<const Node*> consumer_nodes = graph.GetConsumerNodes(param);
    // Initial support is limited to caching Cast output. This can
    // be extended to accomodate more ops whose result depends only
    // on the weight tensor which is a WIP.
    for (const Node* node : consumer_nodes) {
      if (node != nullptr && node->OpType() == "Cast") {
        graph_info_.frontier_node_arg_map[param] = node->OutputDefs()[0]->Name();
      }
    }
  }
}

void OrtModuleGraphBuilder::HandleOutputsAndGrads() {
  Graph& gradient_graph = gradient_model_->MainGraph();
  GraphViewer gradient_graph_viewer(gradient_graph);
  const auto& gradient_node_topology_list = gradient_graph_viewer.GetNodesInTopologicalOrder();
  std::unordered_set<std::string> user_output_grad_names_set;
  for (const auto& name : graph_info_.user_output_names) {
    user_output_grad_names_set.insert(GradientBuilderBase::GradientName(name));
  }

  // If an output gradient is output of one of nodes, need to add this output to PT's output gradient.
  std::unordered_set<std::string> internal_output_grad_names;
  for (auto node_index : gradient_node_topology_list) {
    auto& node = *gradient_graph.GetNode(node_index);
    for (const auto& node_arg : node.OutputDefs()) {
      if (user_output_grad_names_set.find(node_arg->Name()) != user_output_grad_names_set.end()) {
        internal_output_grad_names.insert(node_arg->Name());
      }
    }
  }

  for (const auto& output_grad_name : internal_output_grad_names) {
    Node* producer_node = gradient_graph.GetMutableProducerNode(output_grad_name);
    int producer_node_arg_index = graph_utils::GetNodeOutputIndexFromOutputName(*producer_node, output_grad_name);
    const TypeProto* type_info = producer_node->MutableOutputDefs()[producer_node_arg_index]->TypeAsProto();
    auto& external_node_arg = gradient_graph.GetOrCreateNodeArg(
        gradient_graph.GenerateNodeArgName(GradientBuilderBase::ExternalOutputName(output_grad_name)), type_info);
    auto& output_node_arg = gradient_graph.GetOrCreateNodeArg(
        gradient_graph.GenerateNodeArgName(output_grad_name + "_add_output"), type_info);
    Node& add_node = gradient_graph.AddNode(
        output_grad_name + "_add", "Add", "",
        std::array{&external_node_arg, producer_node->MutableOutputDefs()[producer_node_arg_index]},
        std::array{&output_node_arg});
    graph_utils::ReplaceDownstreamNodeInput(gradient_graph, *producer_node, producer_node_arg_index, add_node, 0);
  }

  NodeAttributes attributes{};

  // YieldOps non_differentiable_outputs attribute specifies the indices of outputs that are not differentiable
  const auto& non_differentiable_indices = graph_info_.output_grad_indices_non_differentiable;
  const std::string non_differentiable_outputs_name = "non_differentiable_outputs";
  ONNX_NAMESPACE::AttributeProto non_differentiable_outputs;
  non_differentiable_outputs.set_name(non_differentiable_outputs_name);
  non_differentiable_outputs.set_type(ONNX_NAMESPACE::AttributeProto::INTS);

  if (non_differentiable_indices.size() > 0) {
    for (auto index : non_differentiable_indices) {
      non_differentiable_outputs.add_ints(index);
    }
  }

  // YieldOps full_shape_outputs attribute specifies the indices of outputs that must be full shape.
  // We need this info to set make TypeAndShapeInferenceFunction work properly.
  ONNX_NAMESPACE::AttributeProto full_shape_outputs;
  const std::string full_shape_outputs_name = "full_shape_outputs";
  full_shape_outputs.set_name(full_shape_outputs_name);
  full_shape_outputs.set_type(ONNX_NAMESPACE::AttributeProto::INTS);

  std::vector<NodeArg*> yield_input_node_args;
  std::vector<NodeArg*> yield_output_node_args;
  graph_info_.output_grad_indices_require_full_shape.clear();
  for (size_t i = 0; i < graph_info_.user_output_names.size(); i++) {
    std::string name = graph_info_.user_output_names[i];
    yield_input_node_args.emplace_back(gradient_graph.GetNodeArg(name));
    std::string grad_name = GradientBuilderBase::GradientName(name);
    if (internal_output_grad_names.find(grad_name) != internal_output_grad_names.end()) {
      grad_name = GradientBuilderBase::ExternalOutputName(grad_name);
    } else {
      // If output grad is the direct input of backward graph, we need to materialize it
      // to a all-0 tensor with same shape of output, otherwise, since it will be an input of
      // Add node, it's OK to use scalar-0 tensor to save memory.
      graph_info_.output_grad_indices_require_full_shape.emplace_back(i);
      full_shape_outputs.add_ints(static_cast<int64_t>(i));
    }

    if (std::find(non_differentiable_indices.begin(), non_differentiable_indices.end(), i) ==
        non_differentiable_indices.end()) {
      NodeArg* grad_node_arg = gradient_graph.GetNodeArg(grad_name);
      ORT_ENFORCE(grad_node_arg != nullptr, "Differentiable param grad node arg should exist.");
      yield_output_node_args.emplace_back(grad_node_arg);
      graph_info_.module_output_gradient_name.emplace_back(grad_name);
    }
  }

  size_t input_count = yield_input_node_args.size();
  for (auto& iter : graph_info_.frontier_node_arg_map) {
    std::string name = iter.second;
    yield_input_node_args.emplace_back(gradient_graph.GetNodeArg(name));
    graph_info_.cached_node_arg_names.emplace_back(name);
  }

  const auto& frontier_tensors = graph_info_.frontier_node_arg_map;
  if (frontier_tensors.size() > 0) {
    for (size_t index = input_count; index < input_count + frontier_tensors.size(); index++) {
      non_differentiable_outputs.add_ints(index);
    }
  }

  // YieldOps non_differentiable_outputs /attribute specifies the indices of outputs that are not differentiable
  if (non_differentiable_indices.size() > 0 || frontier_tensors.size() > 0) {
    attributes.insert({non_differentiable_outputs_name, non_differentiable_outputs});
  }

  attributes.insert({full_shape_outputs_name, full_shape_outputs});

  // Handle potential duplicated output_gradient names
  std::unordered_map<std::string, std::vector<size_t>> name_to_idx;
  for (size_t i = 0; i < yield_output_node_args.size(); ++i) {
    ORT_ENFORCE(yield_output_node_args[i] != nullptr);
    const std::string& name = yield_output_node_args[i]->Name();
    auto it = name_to_idx.find(name);
    if (it == name_to_idx.end()) {
      name_to_idx.insert(std::make_pair(name, std::vector<size_t>{i}));
    } else {
      it->second.push_back(i);
    }
  }

  for (auto& name_idx_pair : name_to_idx) {
    // Only process if there is a duplicated name
    if (name_idx_pair.second.size() > 1) {
      const std::string& arg_name = name_idx_pair.first;
      const std::vector<size_t>& indices = name_idx_pair.second;

      // Replace duplicated names with indexed names
      std::vector<NodeArg*> sum_input_node_args;
      std::vector<NodeArg*> sum_output_node_arg;
      sum_output_node_arg.push_back(gradient_graph.GetNodeArg(arg_name));

      int duplicate_counter = 0;
      for (size_t idx : indices) {
        std::string indexed_arg_name = arg_name + "_" + std::to_string(duplicate_counter++);
        auto& indexed_node_arg =
            gradient_graph.GetOrCreateNodeArg(indexed_arg_name, yield_output_node_args[idx]->TypeAsProto());
        sum_input_node_args.push_back(&indexed_node_arg);
        yield_output_node_args[idx] = &indexed_node_arg;
      }

      // Insert the Sum node to sum-up the duplicated gradients
      gradient_graph.AddNode("Sum_for_" + arg_name, "Sum", "Sum up duplicated gradient", sum_input_node_args,
                             sum_output_node_arg, {}, kOnnxDomain);
    }
  }

  gradient_graph.AddNode("YieldOp", "YieldOp", "Yield Op", yield_input_node_args, yield_output_node_args, &attributes,
                         kMSDomain);
}

void OrtModuleGraphBuilder::ReorderOutputs() {
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
  graph_info_.user_input_grad_names.clear();
  for (const auto& input_name : graph_info_.user_input_names) {
    if (user_input_require_grad_set.find(input_name) != user_input_require_grad_set.end()) {
      std::string input_gradient_name = GradientBuilderBase::GradientName(input_name);
      ORT_ENFORCE(gradient_output_arg_map.find(input_gradient_name) != gradient_output_arg_map.end(),
                  "Required user input grad is not found on gradient graph.");
      graph_info_.user_input_grad_names[input_name] = input_gradient_name;
      new_output_args.emplace_back(gradient_output_arg_map[input_gradient_name]);
    }
  }

  // Add initializer gradients to graph outputs.
  graph_info_.initializer_grad_names_to_train.clear();
  for (const auto& initializer_name : config_.initializer_names_to_train) {
    std::string initializer_gradient_name = GradientBuilderBase::GradientName(initializer_name);
    ORT_ENFORCE(gradient_output_arg_map.find(initializer_gradient_name) != gradient_output_arg_map.end(),
                "Trainable initializer grad is not found on gradient graph.");
    graph_info_.initializer_grad_names_to_train.emplace_back(initializer_gradient_name);
    new_output_args.emplace_back(gradient_output_arg_map[initializer_gradient_name]);
  }

  gradient_graph.SetOutputs(new_output_args);
}

void OrtModuleGraphBuilder::FindModuleOutputNeededForBackward() {
  Graph& gradient_graph = gradient_model_->MainGraph();
  ORT_THROW_IF_ERROR(gradient_graph.Resolve());
  GraphViewer gradient_graph_viewer(gradient_graph);
  const auto& exec_order = gradient_graph_viewer.GetNodesInTopologicalOrder();

  size_t yield_node_order = 0;
  bool yield_node_found = false;
  std::unordered_map<NodeIndex, size_t> id_to_exec_order;
  for (size_t i = 0; i < exec_order.size(); ++i) {
    if (gradient_graph_viewer.GetNode(exec_order[i])->OpType() == "YieldOp") {
      yield_node_order = i;
      yield_node_found = true;
    }
    id_to_exec_order.insert({exec_order[i], i});
  }
  ORT_ENFORCE(yield_node_found, "YieldOp is not found in the training graph");

  const Node* yield_node = gradient_graph_viewer.GetNode(exec_order[yield_node_order]);
  auto yield_input_node_args = yield_node->InputDefs();

  for (size_t i = 0; i < yield_input_node_args.size(); ++i) {
    const NodeArg* yield_input = yield_input_node_args[i];

    const Node* producer_node = gradient_graph.GetProducerNode(yield_input->Name());
    if (producer_node->OpType() == "Identity") {
      yield_input = producer_node->InputDefs()[0];
    }

    std::vector<const Node*> consumer_nodes = gradient_graph.GetConsumerNodes(yield_input->Name());
    for (const Node* n : consumer_nodes) {
      // If a module output has a consumer that is executed after the YieldOp, marked it needed for backward
      if (id_to_exec_order[n->Index()] > yield_node_order) {
        graph_info_.module_output_indices_requires_save_for_backward.emplace_back(i);
        break;
      }
    }
  }

  // Graph resolve will have the YieldOp outputs' shapes inferred. To avoid lossing these information when
  // transferring model from backend to frontend (in case any graph optimization requires these shape information),
  // add them to graph's ValueInfo.
  for (const auto& node_def : yield_node->OutputDefs()) {
    if (node_def->TypeAsProto()) {
      gradient_graph.AddValueInfo(node_def);
    }
  }
}

void OrtModuleGraphBuilder::UpdatePythonOpInputsRequireGradInfo(
    const std::unordered_map<std::string, std::vector<int64_t>>& python_op_input_require_grad_info) {
  Graph& gradient_graph = gradient_model_->MainGraph();
  // Input require grad info are not alwarys correct after torch export.
  // So we update the info here according to ORT gradient graph.
  // Be noted: we only update PythonOp that is differentiable.
  GraphViewer gradient_graph_viewer(gradient_graph);
  const auto& gradient_node_topology_list = gradient_graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : gradient_node_topology_list) {
    auto& node = *gradient_graph.GetNode(node_index);
    if (node.OpType() == "PythonOp") {
      if (python_op_input_require_grad_info.find(node.Name()) != python_op_input_require_grad_info.end()) {
        auto input_requires_grads_attr = graph_utils::GetNodeAttribute(node, "input_requires_grads");
        if (input_requires_grads_attr != nullptr) {
          node.ClearAttribute("input_requires_grads");
        }
        node.AddAttribute("input_requires_grads", python_op_input_require_grad_info.at(node.Name()));
      }
    }
  }
}

}  // namespace training
}  // namespace onnxruntime
