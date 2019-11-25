// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/gelu_approximation.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "float.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// FastGelu supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)"};

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}

static bool CheckInputShape(const NodeArg& input, const NodeArg& bias) {
  const TensorShapeProto* input_shape = input.Shape();
  const TensorShapeProto* bias_shape = bias.Shape();

  if (nullptr == input_shape || nullptr == bias_shape) {
    return false;
  }

  if (input_shape->dim_size() < 1 ||
      bias_shape->dim_size() != 1) {
    return false;
  }

  int last_dim = input_shape->dim_size() - 1;
  if (!utils::HasDimValue(input_shape->dim(last_dim)) ||
      !utils::HasDimValue(bias_shape->dim(0)) ||
      input_shape->dim(last_dim).dim_value() != bias_shape->dim(0).dim_value()) {
    return false;
  }

  return true;
}

static bool CheckInputShape(const NodeArg& input) {
  const TensorShapeProto* input_shape = input.Shape();
  if (nullptr == input_shape) {
    return false;
  }

  if (input_shape->dim_size() < 1) {
    return false;
  }

  return true;
}

static bool IsCandidateNode(Node& node, const std::unordered_set<std::string>& compatible_providers) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "AddGeluFusion", {1}, kMSDomain)) {
    return graph_utils::IsSupportedProvider(node, compatible_providers) &&
           IsSupportedDataType(node) &&
           CheckInputShape(*(node.InputDefs()[0]), *(node.InputDefs()[1]));
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gelu", {1}, kMSDomain)) {
    return graph_utils::IsSupportedProvider(node, compatible_providers) &&
           IsSupportedDataType(node) &&
           CheckInputShape(*(node.InputDefs()[0]));
  }
  return false;
}

Status GeluApproximation::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (IsCandidateNode(node, GetCompatibleExecutionProviders())) {
      Node& fastgelu = graph.AddNode(
          graph.GenerateNodeName("FastGelu"),
          "FastGelu",
          "Gelu approximation",
          node.MutableInputDefs(),
          node.MutableOutputDefs(), nullptr, kMSDomain);

      fastgelu.SetExecutionProviderType(node.GetExecutionProviderType());

      graph_utils::RemoveNodeOutputEdges(graph, node);
      graph.RemoveNode(node.Index());

      modified = true;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
