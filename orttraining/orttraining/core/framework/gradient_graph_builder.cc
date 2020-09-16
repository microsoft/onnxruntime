// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/graph/op.h"
#include "core/graph/schema_registry.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/gradient_config.h"
#include "orttraining/core/optimizer/insert_output_rewriter.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"

using namespace ONNX_NAMESPACE;
using namespace std;

namespace onnxruntime {
namespace training {

using namespace common;

GradientGraphBuilder::GradientGraphBuilder(Graph* graph,
                                           const unordered_set<string>& y_node_arg_names,
                                           const unordered_set<string>& x_node_arg_names,
                                           const std::string& loss_node_arg_name,
                                           const GradientGraphConfiguration& gradient_graph_config,
                                           const logging::Logger& logger)
    : graph_(graph),
      loss_node_arg_name_(loss_node_arg_name),
      gradient_graph_config_(gradient_graph_config),
      logger_(logger) {
  auto rule_based_graph_transformer =
      onnxruntime::make_unique<RuleBasedGraphTransformer>("pre_training_rule_based_graph_transformer");
  rule_based_graph_transformer->Register(make_unique<InsertMaxPoolOutput>());
  rule_based_graph_transformer->Register(make_unique<AdjustBatchNormOutputs>());

  graph_transformation_mgr_.Register(std::move(rule_based_graph_transformer),
                                     TransformerLevel::Level2);

  for (const auto& name : y_node_arg_names) {
    const NodeArg* node_arg = graph->GetNodeArg(name);
    if (!node_arg) {
      std::stringstream ss;
      bool first = true;
      for (auto& p_output_node : graph->GetOutputs()) {
        if (first) {
          first = false;
        } else {
          ss << ", ";
        }
        ss << "'" << p_output_node->Name() << "'";
      }
      ORT_THROW("Node arg '", name, "' is not found in the graph. Available output names = ", ss.str());
    }
    y_node_args_.insert(node_arg);

    const Node* node = graph_->GetProducerNode(name);
    if (!node) {
      ORT_THROW(name, " couldn't find the producer node.");
    }
    y_nodes_.insert(node);
  }

  for (const auto& name : x_node_arg_names) {
    const NodeArg* node_arg = graph->GetNodeArg(name);
    if (!node_arg) {
      ORT_THROW("Node arg ", name, " is not found in the graph.");
    }
    x_node_args_.insert(node_arg);

    vector<const Node*> nodes = graph_->GetConsumerNodes(name);
    if (nodes.empty()) {
      ORT_THROW(name, " couldn't find the consumer node.");
    }

    string grad_arg_name = GradientBuilderBase::GradientName(name);
    pending_[grad_arg_name] = static_cast<int>(nodes.size());

    x_nodes_.insert(nodes.begin(), nodes.end());
  }
}

NodeSet GradientGraphBuilder::ReverseBFS(const NodeSet& nodes) {
  NodeSet visited(nodes);
  deque<const Node*> queue(nodes.begin(), nodes.end());

  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();

    for (auto edge_it = n->InputEdgesBegin(); edge_it != n->InputEdgesEnd(); ++edge_it) {
      auto it = STOP_GRADIENT_EDGES.find(n->OpType());
      if (it != STOP_GRADIENT_EDGES.end() && it->second.count(edge_it->GetDstArgIndex())) {
        LOGS(logger_, WARNING) << "Skip building gradient for input_" << edge_it->GetDstArgIndex()
                               << " of node: " << n->Name();
        continue;
      }

      const Node& node = edge_it->GetNode();
      if (visited.find(&node) == visited.end()) {
        queue.push_back(&node);
        visited.insert(&node);
      }
    }
  }
  return visited;
}

Status GradientGraphBuilder::CheckNodeArgsReachable(const NodeSet& reachable_nodes) {
  for (const NodeArg* node_arg : x_node_args_) {
    auto nodes = graph_->GetConsumerNodes(node_arg->Name());

    bool reachable = false;
    for (const Node* node : nodes) {
      if (reachable_nodes.find(node) != reachable_nodes.end()) {
        reachable = true;
        break;
      }
    }
    if (!reachable) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Cannot compute the partial derivative for '", node_arg->Name(),
                             "' as it's unreachable from the output node(s).");
    }
  }
  return Status::OK();
}

Status GradientGraphBuilder::Build(const std::unordered_set<std::string>* p_initializer_names_to_preserve) {
  auto opt_ret = graph_transformation_mgr_.ApplyTransformers(*graph_, TransformerLevel::Level2, logger_);
  ORT_RETURN_IF_ERROR(opt_ret);

  GraphAugmenter::GraphDefs gradient_graph_defs;
  // add "gradient of the loss" node, always 1.
  if (loss_node_arg_name_ != "") {
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_proto.add_float_data(1.f);
    tensor_proto.set_name(GradientBuilderBase::GradientName(loss_node_arg_name_));

    gradient_graph_defs.AddInitializers({tensor_proto});
  }

  NodeSet reachable_nodes = ReverseBFS(y_nodes_);

  ORT_RETURN_IF_ERROR(CheckNodeArgsReachable(reachable_nodes));

  // Going forward to figure out which node_args need backprop-ed.
  deque<const Node*> queue(x_nodes_.begin(), x_nodes_.end());
  NodeSet visited(x_nodes_);
  unordered_set<const NodeArg*> visited_node_args = x_node_args_;
  visited_node_args.insert(y_node_args_.begin(), y_node_args_.end());

  while (!queue.empty()) {
    const Node* node = queue.front();
    queue.pop_front();

    for (auto edge_it = node->OutputEdgesBegin(); edge_it != node->OutputEdgesEnd(); ++edge_it) {
      const Node& next_node = edge_it->GetNode();

      if (reachable_nodes.find(&next_node) == reachable_nodes.end()) continue;

      auto it = STOP_GRADIENT_EDGES.find(next_node.OpType());
      if (it != STOP_GRADIENT_EDGES.end() && it->second.count(edge_it->GetDstArgIndex())) {
        LOGS(logger_, WARNING) << "Skip building gradient for input_" << edge_it->GetDstArgIndex()
                               << " of node: " << next_node.Name();
        continue;
      }

      const NodeArg* node_arg = node->OutputDefs()[edge_it->GetSrcArgIndex()];
      string grad_node_arg_name = GradientBuilderBase::GradientName(node_arg->Name());

      pending_[grad_node_arg_name] += 1;

      visited_node_args.insert(node_arg);

      if (visited.find(&next_node) == visited.end()) {
        queue.push_back(&next_node);
        visited.insert(&next_node);
      }
    }
  }

  // so far, visited are the minimum node in between
  // visited_node_args are the node_args involved
  for (auto node : visited) {
    //TODO: might not need two sets, the union of them might be enough
    unordered_set<string> input_args_need_grad, output_args_need_grad;
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

    GradientDef node_defs = GetGradientForOp(gradient_graph_config_, graph_, node, output_args_need_grad, input_args_need_grad, logger_);

    // updates arg name if gradient accumulation is needed
    for (auto& op_def : node_defs) {
      for (auto& arg : op_def.output_args) {
        auto found = pending_.find(arg.name);
        if (found != pending_.end() && found->second > 1) {
          auto idx = gradients_to_accumulate_[arg].size();
          string indexed_arg_name = arg.name + "_" + to_string(idx);
          gradients_to_accumulate_[arg].push_back(ArgDef(indexed_arg_name, arg.type_proto));

          arg.name = indexed_arg_name;
        }
      }
    }
    gradient_graph_defs.AddNodeDefs(node_defs);
  }

  // Accumulate Gradients
  for (auto gradient_pair : gradients_to_accumulate_) {
    gradient_graph_defs.AddNodeDefs(
        {NodeDef("Sum",
                 gradient_pair.second,
                 {gradient_pair.first},
                 NodeAttributes(),
                 "AccumulateGrad_" + gradient_pair.first.name)});
  }

  if (gradient_graph_config_.set_gradients_as_graph_outputs) {
    for (auto x_node_arg : x_node_args_) {
      gradient_graph_defs.AddGraphOutputs({GradientBuilderBase::GradientName(x_node_arg->Name())});
    }
  }

  return GraphAugmenter::AugmentGraph(*graph_, gradient_graph_defs, p_initializer_names_to_preserve);
}

}  // namespace training
}  // namespace onnxruntime
