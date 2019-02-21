// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#include "core/graph/gradients.h"

#include <fstream>
#include <iostream>
#include <numeric>
#include <stack>

#include "gsl/pointers"
#include "core/optimizer/initializer.h"
#include "core/graph/function.h"
#include "core/graph/function_impl.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/op.h"
#include "core/common/logging/logging.h"
#include "onnx/checker.h"
#include "core/graph/schema_registry.h"
using namespace ONNX_NAMESPACE;
using namespace ONNX_NAMESPACE::Utils;
using namespace ONNX_NAMESPACE::checker;
using namespace ::onnxruntime::common;
using namespace ::onnxruntime::GradientOps;

namespace onnxruntime {
/**
This builder class constructs the backward gradient graph

@param fw_graph The forward computation graph
@param y_node_args List of NodeArgs whoes initial gradients will be provided
@param x_node_args List of NodeArgs that need the gradients

@remarks Given initial graidents at 'y_node_args' w.r.t some loss function L,
  the backward graph computes the partial derivative of 'L' w.r.t the 'x_node_args'
*/
GradientGraphBuilder::GradientGraphBuilder(Graph* fw_graph, Graph* bw_graph,
                                           std::vector<NodeArg*> y_node_args,
                                           std::vector<NodeArg*> x_node_args,
                                           std::string loss_node_arg_name) : y_node_args_(y_node_args),
                                                                             x_node_args_(x_node_args),
                                                                             fw_graph_(fw_graph),
                                                                             bw_graph_(bw_graph),
                                                                             loss_node_arg_name_(loss_node_arg_name) {
  Build();
}

GradientGraphBuilder::GradientGraphBuilder(Graph* fw_graph,
                                           Graph* bw_graph,
                                           const std::vector<std::string>& y_node_arg_names,
                                           const std::vector<std::string>& x_node_arg_names,
                                           std::string loss_node_arg_name) : fw_graph_(fw_graph),
                                                                             bw_graph_(bw_graph),
                                                                             loss_node_arg_name_(loss_node_arg_name) {
  for (auto y_node_arg_name : y_node_arg_names) {
    y_node_args_.push_back(fw_graph->GetNodeArg(y_node_arg_name));
  }

  for (auto x_node_arg_name : x_node_arg_names) {
    auto temp = fw_graph->GetNodeArg(x_node_arg_name);
    if (temp == nullptr) {
      temp = &fw_graph->GetOrCreateNodeArg(x_node_arg_name, nullptr);
    }
    x_node_args_.push_back(temp);
  }
  Build();
}

void GradientGraphBuilder::AddLossGradient() {
  // add loss gradient
  onnx::TensorProto tensor_proto;
  tensor_proto.add_dims(1);
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_proto.add_float_data(1.f);
  tensor_proto.set_name(loss_node_arg_name_ + "_grad");

  bw_graph_->AddInitializedTensor(tensor_proto);
}

Status GradientGraphBuilder::Build() {
  AddLossGradient();

  std::vector<std::string> weights_to_train = {};
  for (auto x_node_arg : x_node_args_) {
    weights_to_train.push_back(x_node_arg->Name());
  }
  bw_graph_->SetWeightsToTrain(weights_to_train);

  std::unordered_set<const Node*> visited;
  std::deque<const Node*> queue;

  // forward pass
  for (auto node_arg : x_node_args_) {
    // nodes that used x_node_args_ as the inputs
    auto nodes = fw_graph_->GetConsumerNodes(node_arg->Name());

    if (nodes.empty()) {
      continue;
    }
    visited.insert(nodes.begin(), nodes.end());
    queue.insert(queue.end(), nodes.begin(), nodes.end());
  }

  while (!queue.empty()) {
    const Node* n = queue.front();
    queue.pop_front();

    // !!! Can change to OutputNodesBegin if needed
    for (auto edge_it = n->OutputEdgesBegin(); edge_it != n->OutputEdgesEnd(); edge_it++) {
      const Node& next_node = edge_it->GetNode();
      if (visited.find(&next_node) == visited.end()) {
        visited.insert(&next_node);
        queue.push_back(&next_node);
      }
    }
  }

  // backward pass
  std::unordered_set<const Node*> backward_visited;
  std::deque<const Node*> backward_queue;

  std::unordered_set<const NodeArg*> visited_noded_args;

  for (auto node_arg : y_node_args_) {
    const Node* node = fw_graph_->GetProducerNode(node_arg->Name());
    if (visited.find(node) != visited.end()) {
      backward_visited.insert(node);
      backward_queue.push_back(node);
    }
    visited_noded_args.insert(node_arg);
  }

  while (!backward_queue.empty()) {
    const Node* n = backward_queue.front();
    backward_queue.pop_front();

    for (auto edge_it = n->InputEdgesBegin(); edge_it != n->InputEdgesEnd(); edge_it++) {
      const Node& prev_node = edge_it->GetNode();

      if (visited.find(&prev_node) != visited.end()) {
        const NodeArg* node_arg = prev_node.OutputDefs()[edge_it->GetSrcArgIndex()];

        //auto key = std::make_pair<NodeIndex, int>(prev_node.Index(), edge_it->GetSrcArgIndex());
        if (backward_visited.find(&prev_node) == backward_visited.end()) {
          backward_visited.insert(&prev_node);
          backward_queue.push_back(&prev_node);

          pending_.insert({node_arg->Name(), 0});
          gradients_to_accumulate_.insert({node_arg->Name(), std::vector<std::string>()});
        }
        pending_[node_arg->Name()]++;

        visited_noded_args.insert(node_arg);
      }
    }
  }

  visited_noded_args.insert(x_node_args_.begin(), x_node_args_.end());

  // so far, backward_visited are the minimum node in between
  // visited_noded_args are the node_args involved

  for (auto node : backward_visited) {
    std::string gradient_op_type = GradientOpType(node->OpType());

    GradientOpSchema gradient_schema = GradOpSchemaRegistryHelper::GradientOpRegistry[gradient_op_type];
    std::vector<NodeArg*> input_args, output_args;

    // TODO: Not all inputs are required
    for (auto input_mapping : gradient_schema.InputMappings()) {
      NodeArg& input_arg = GetOrCreateNodeArg(node, input_mapping);
      input_args.push_back(&input_arg);
    }

    // TODO: Not all outputs are needed
    for (auto output_mapping : gradient_schema.OutputMappings()) {
      NodeArg& output_arg = GetOrCreateNodeArg(node, output_mapping, visited_noded_args);
      output_args.push_back(&output_arg);
    }

    bw_graph_->AddNode(
        "",  // TODO: currently, using op type as node name
        gradient_op_type,
        "some description",
        input_args,
        output_args,
        &node->GetAttributes(),
        node->Domain());
  }

  // Accumulate Gradients
  for (auto pair : pending_) {
    if (pair.second > 1) {
      std::string arg_name = pair.first;

      std::vector<NodeArg*> input_args, output_args;

      output_args.push_back(bw_graph_->GetNodeArg(GradientName(arg_name)));

      for (auto node_arg_name : gradients_to_accumulate_[arg_name]) {
        input_args.push_back(bw_graph_->GetNodeArg(node_arg_name));
      }

      bw_graph_->AddNode(
          "",  // TODO: currently, using op type as node name
          "AddN",
          "some description",
          input_args,
          output_args,
          nullptr,
          "");
    }
  }

  return Status::OK();
}

void GradientGraphBuilder::CopyInitializedTensor(const std::string& tensor_name) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  fw_graph_->GetInitializedTensor(tensor_name, tensor_proto);

  auto initializer = std::make_unique<Initializer>(tensor_proto);

  ONNX_NAMESPACE::TensorProto new_tensor_proto;
  initializer->ToProto(&new_tensor_proto);

  bw_graph_->AddInitializedTensor(new_tensor_proto);
}

NodeArg& GradientGraphBuilder::GetOrCreateNodeArg(const Node* node, DefsMapping mapping,
                                                  const std::unordered_set<const NodeArg*>& visited_node_arg) {
  size_t index = mapping.second;

  if (mapping.first == "I") {
    const NodeArg* input_arg = node->InputDefs()[index];
    std::string arg_name = input_arg->Name();

    return bw_graph_->GetOrCreateNodeArg(arg_name, input_arg->TypeAsProto());
  } else if (mapping.first == "GI") {
    if (index >= node->InputDefs().size()) {
      return bw_graph_->GetOrCreateNodeArg("", nullptr);
    }

    const NodeArg* input_arg = node->InputDefs()[index];
    std::string arg_name = "";

    if (visited_node_arg.find(input_arg) != visited_node_arg.end()) {
      std::string input_arg_name = input_arg->Name();
      arg_name = GradientName(input_arg_name);

      // use pending to check if needed
      if (pending_[input_arg_name] > 1) {
        auto iter = gradients_to_accumulate_.find(input_arg_name);
        if (iter != gradients_to_accumulate_.end()) {
          arg_name += "_" + std::to_string(iter->second.size());
          iter->second.push_back(arg_name);
        }
      }
    }

    return bw_graph_->GetOrCreateNodeArg(arg_name, input_arg->TypeAsProto());
  } else if (mapping.first == "GO") {
    const NodeArg* output_arg = node->OutputDefs()[index];
    std::string arg_name = GradientName(output_arg->Name());

    return bw_graph_->GetOrCreateNodeArg(arg_name, output_arg->TypeAsProto());
  } else {
    // TODO: raise exception
  }
  return bw_graph_->GetOrCreateNodeArg("", nullptr);
}

}  // namespace onnxruntime
