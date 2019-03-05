// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#include "core/training/gradients.h"
#include "core/graph/op.h"
#include "core/common/logging/logging.h"
#include "core/graph/schema_registry.h"
#include "core/training/gradient_registry.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

GradientGraphBuilder::GradientGraphBuilder(const Graph* graph,
                                           const std::vector<std::string>& y_node_arg_names,
                                           const std::vector<std::string>& x_node_arg_names,
                                           std::string loss_node_arg_name) : graph_(graph),
                                                                             loss_node_arg_name_(loss_node_arg_name) {
  for (const auto& name : y_node_arg_names) {
    const NodeArg* node_arg = graph->GetNodeArg(name);
    if (node_arg != nullptr) {
      y_node_args_.push_back(node_arg);
    } else {
      ORT_THROW("Node arg ", name, " is not found in the graph.");
    }
  }

  for (const auto& name : x_node_arg_names) {
    const NodeArg* node_arg = graph->GetNodeArg(name);
    if (node_arg != nullptr) {
      x_node_args_.push_back(node_arg);
    } else {
      ORT_THROW("Node arg ", name, " is not found in the graph.");
    }
  }
}

std::unordered_set<const Node*> GradientGraphBuilder::BFS(const std::vector<const NodeArg*>& starting_node_args) {
  std::unordered_set<const Node*> visited;
  std::deque<const Node*> queue;

  for (auto node_arg : starting_node_args) {
    // gets nodes that consumes node_arg as inputs
    auto nodes = graph_->GetConsumerNodes(node_arg->Name());

    if (nodes.empty()) {
      continue;
    }
    visited.insert(nodes.begin(), nodes.end());
    queue.insert(queue.end(), nodes.begin(), nodes.end());
  }

  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();

    for (auto edge_it = n->OutputEdgesBegin(); edge_it != n->OutputEdgesEnd(); edge_it++) {
      const Node& node = edge_it->GetNode();
      if (visited.find(&node) == visited.end()) {
        visited.insert(&node);
        queue.push_back(&node);
      }
    }
  }
  return visited;
}

GraphAugmenter::GraphDefs GradientGraphBuilder::Build() {
  GraphAugmenter::GraphDefs gradient_graph_defs;

  // add "gradient of the loss" node, always 1.
  {
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.add_dims(1);
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_proto.add_float_data(1.f);
    tensor_proto.set_name(GradientBuilderBase::GradientName(loss_node_arg_name_));

    gradient_graph_defs.AddInitializers({tensor_proto});
  }

  std::unordered_set<const Node*> visited = BFS(x_node_args_);

  // backward pass
  std::unordered_set<const Node*> backward_visited;
  std::deque<const Node*> backward_queue;

  std::unordered_set<const NodeArg*> visited_node_args;

  for (auto node_arg : y_node_args_) {
    const Node* node = graph_->GetProducerNode(node_arg->Name());
    if (visited.find(node) != visited.end()) {
      backward_visited.insert(node);
      backward_queue.push_back(node);
    }
    visited_node_args.insert(node_arg);
  }

  while (!backward_queue.empty()) {
    const Node* n = backward_queue.front();
    backward_queue.pop_front();

    for (auto edge_it = n->InputEdgesBegin(); edge_it != n->InputEdgesEnd(); edge_it++) {
      const Node& prev_node = edge_it->GetNode();

      if (visited.find(&prev_node) != visited.end()) {
        const NodeArg* node_arg = prev_node.OutputDefs()[edge_it->GetSrcArgIndex()];

        std::string gradient_node_arg_name = GradientBuilderBase::GradientName(node_arg->Name());

        if (backward_visited.find(&prev_node) == backward_visited.end()) {
          backward_visited.insert(&prev_node);
          backward_queue.push_back(&prev_node);

          pending_.insert({gradient_node_arg_name, 0});
          gradients_to_accumulate_.insert({gradient_node_arg_name, std::vector<std::string>()});
        }
        pending_[gradient_node_arg_name]++;

        visited_node_args.insert(node_arg);
      }
    }
  }

  visited_node_args.insert(x_node_args_.begin(), x_node_args_.end());

  // so far, backward_visited are the minimum node in between
  // visited_node_args are the node_args involved

  for (auto node : backward_visited) {
    //TODO: might not need two sets, the union of them might be enough
    std::unordered_set<std::string> input_args_need_grad, output_args_need_grad;
    for (auto arg : node->InputDefs()) {
      if (visited_node_args.find(arg) != visited_node_args.end()) {
        input_args_need_grad.insert(arg->Name());
      }
    }
    for (auto arg : node->OutputDefs()) {
      if (visited_node_args.find(arg) != visited_node_args.end()) {
        output_args_need_grad.insert(arg->Name());
      }
    }

    GradientDef node_defs = GetGradientForOp(node, output_args_need_grad, input_args_need_grad);

    // updates arg name if gradient accumulation is needed
    for (auto& op_def : node_defs) {
      for (auto& arg : op_def.output_args) {
        std::string& arg_name = arg.name;
        auto found = pending_.find(arg_name);
        if (found != pending_.end() && found->second > 1) {
          auto iter = gradients_to_accumulate_.find(arg_name);
          if (iter != gradients_to_accumulate_.end()) {
            arg_name += "_" + std::to_string(iter->second.size());
            iter->second.push_back(arg_name);
          }
        }
      }
    }

    gradient_graph_defs.AddNodeDefs(node_defs);
  }

  // Accumulate Gradients
  for (auto pair : pending_) {
    if (pair.second > 1) {
      std::string arg_name = pair.first;
      std::vector<ArgDef> input_args, output_args;
      output_args.push_back({arg_name, graph_->GetNodeArg(arg_name)->TypeAsProto()});

      for (auto node_arg_name : gradients_to_accumulate_[arg_name]) {
        input_args.push_back({node_arg_name, graph_->GetNodeArg(node_arg_name)->TypeAsProto()});
      }

      gradient_graph_defs.AddNodeDefs({NodeDef("AddN", input_args, output_args)});
    }
  }

  // Set the gradients as graph outputs.
  for (auto x_node_arg : x_node_args_) {
    gradient_graph_defs.AddGraphOutputs({GradientBuilderBase::GradientName(x_node_arg->Name())});
  }

  return gradient_graph_defs;
}

}  // namespace training
}  // namespace onnxruntime
