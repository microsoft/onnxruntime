// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

bool IsGraphInput(const GraphViewer& graph, const std::string& name) {
  const auto& graph_inputs = graph.GetInputs();
  std::vector<std::string> input_names(graph_inputs.size());
  std::transform(graph_inputs.begin(), graph_inputs.end(), input_names.begin(), [](auto in) {
    return in->Name();
  });
  return (std::find(input_names.begin(), input_names.end(), name) != input_names.end());
}

bool IsGraphInitializer(const GraphViewer& graph, const std::string& name, bool check_outer_scope = true) {
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  return graph.GetInitializedTensor(name, initializer);
}

const Node* GetInputNode(const Node& node, int arg_index) {
  int index = 0;
  for (auto nit = node.InputNodesBegin(); nit != node.InputNodesEnd(); ++nit, ++index) {
    if (index == arg_index) {
      return &(*nit);
    }
  }

  return nullptr;
}

std::size_t getNodeInputNum(const Node& node) {
  std::size_t node_num = 0;
  for (auto it = node.InputNodesBegin(); it != node.InputNodesEnd(); ++it) {
    node_num++;
  }

  return node_num;
}

bool isInputNode(const Node* node, const std::string& name) {
  auto outputs = node->OutputDefs();
  return std::any_of(outputs.begin(), outputs.end(), [&](auto out) {
    return (out->Name() == name);
  });
}

bool canEvalShapeGeneral(const GraphViewer& graph, const Node* node, std::vector<NodeIndex>& input_nodes) {
  if (node == nullptr) {
    return false;
  }

  std::vector<const Node*> in_nodes;
  for (auto nit = node->InputNodesBegin(); nit != node->InputNodesEnd(); ++nit) {
    in_nodes.push_back(&(*nit));
  }

  if (node->OpType() == "Shape") {
    input_nodes.push_back(node->Index());
    return true;
  }

  auto inputs = node->InputDefs();
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const std::string& input_name = inputs.at(i)->Name();
    // If it is an initializer, it can be constant folded
    if (IsGraphInitializer(graph, input_name)) {
      continue;
    }

    // Input for sure cannot be constant folded
    if (IsGraphInput(graph, input_name)) {
      return false;
    }

    // find the node corresponding to the name
    auto nit = std::find_if(in_nodes.begin(), in_nodes.end(), [&](auto n) {
      return isInputNode(n, input_name);
    });
    if (nit == in_nodes.end()) {
      return false;
    }

    auto input_node = (*nit);
    // shape node, it is OK
    if (input_node->OpType() == "Shape") {
      continue;
    }

    if (canEvalShapeGeneral(graph, input_node, input_nodes)) {
      continue;
    }

    return false;
  }

  input_nodes.push_back(node->Index());
  return true;
}

bool canEvalNodeArgument(const GraphViewer& graph, const Node* node, std::vector<std::size_t> indices, std::vector<NodeIndex>& input_nodes) {
  input_nodes.clear();
  std::vector<const Node*> in_nodes;
  for (auto nit = node->InputNodesBegin(); nit != node->InputNodesEnd(); ++nit) {
    in_nodes.push_back(&(*nit));
  }

  auto inputs = node->InputDefs();
  for (auto index : indices) {
    // an initializer itself is a constant
    auto input_name = inputs.at(index)->Name();
    if (IsGraphInitializer(graph, input_name)) {
      continue;
    }

    // Input cannot be constant folded
    if (IsGraphInput(graph, input_name)) {
      return false;
    }

    // find the node corresponding to the name
    auto nit = std::find_if(in_nodes.begin(), in_nodes.end(), [&](auto n) {
      return isInputNode(n, input_name);
    });
    if (nit == in_nodes.end()) {
      return false;
    }

    if (!canEvalShapeGeneral(graph, *nit, input_nodes)) {
      return false;
    }
  }

  return true;
}

}  // namespace onnxruntime
