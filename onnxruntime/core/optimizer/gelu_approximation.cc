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

namespace geluapproximation_internal {
// FastGelu supports limited data types.
static constexpr const char* const supported_data_types[] = {"tensor(float16)", "tensor(float)", "tensor(bfloat16)"};

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(std::begin(supported_data_types), std::end(supported_data_types), *(input_arg->Type())) ==
        std::end(supported_data_types)) {
      return false;
    }
  }
  return true;
}

static bool CheckInputShape(const Node& node, const NodeArg& input, const NodeArg& bias) {
  const TensorShapeProto* bias_shape = bias.Shape();
  if (nullptr == bias_shape || bias_shape->dim_size() != 1 || !utils::HasDimValue(bias_shape->dim(0))) {
    return false;
  }
  auto bias_length = bias_shape->dim(0).dim_value();

  const TensorShapeProto* input_shape = input.Shape();
  if (nullptr != input_shape) {
    if (input_shape->dim_size() >= 1) {
      int last_dim = input_shape->dim_size() - 1;
      if (utils::HasDimValue(input_shape->dim(last_dim)) && input_shape->dim(last_dim).dim_value() == bias_length) {
        return true;
      }
    }
    return false;
  }

  // Input does not have shape. We will check its parent node.
  // When the parent is MatMul and its 2nd input has shape like {*, bias_length},
  // it means that the shape of MatMul output is good for FastGelu.
  const Node* parent_node = graph_utils::GetInputNode(node, 0);
  if (nullptr != parent_node &&
      graph_utils::IsSupportedOptypeVersionAndDomain(*parent_node, "MatMul", {1, 9, 13}, kOnnxDomain)) {
    const NodeArg& input_b = *(parent_node->InputDefs()[1]);
    if (optimizer_utils::ValidateShape(input_b, {-1, bias_length})) {
      return true;
    }
  }

  return false;
}

static bool CheckGeluInputShape(const NodeArg& input) {
  const TensorShapeProto* input_shape = input.Shape();
  return nullptr != input_shape && input_shape->dim_size() >= 1;
}

static bool IsCandidateNode(const Node& node, const InlinedHashSet<std::string_view>& compatible_providers) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "BiasGelu", {1}, kMSDomain)) {
    return graph_utils::IsSupportedProvider(node, compatible_providers) && IsSupportedDataType(node) &&
           CheckInputShape(node, *(node.InputDefs()[0]), *(node.InputDefs()[1]));
  } else if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gelu", {1}, kMSDomain)) {
    return graph_utils::IsSupportedProvider(node, compatible_providers) && IsSupportedDataType(node) &&
           CheckGeluInputShape(*(node.InputDefs()[0]));
  }
  return false;
}

}  // namespace geluapproximation_internal 

Status GeluApproximation::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                    const logging::Logger& logger) const {
  using namespace geluapproximation_internal;
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  int count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr) continue;  // we removed the node as part of an earlier fusion

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (IsCandidateNode(node, GetCompatibleExecutionProviders())) {
      Node& fastgelu = graph.AddNode(graph.GenerateNodeName("FastGelu"), "FastGelu", "Gelu approximation",
                                     node.MutableInputDefs(), node.MutableOutputDefs(), nullptr, kMSDomain);

      fastgelu.SetExecutionProviderType(node.GetExecutionProviderType());

      graph_utils::RemoveNodeOutputEdges(graph, node);
      graph.RemoveNode(node.Index());

      count++;
    }
  }

  if (count > 0) {
    modified = true;
    LOGS(logger, INFO) << "Total Gelu Approximation (FastGelu) node count: " << count;
  }

  return Status::OK();
}
}  // namespace onnxruntime
