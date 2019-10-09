// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <deque>
#include "core/graph/graph_utils.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/initializer.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

namespace {
void HandleActivationNodeEdges(Graph& g, const Node& act, Node& fused_conv) {
  Node::EdgeSet output_edges;
  for (auto it = act.OutputEdgesBegin(); it != act.OutputEdgesEnd(); ++it) {
    output_edges.insert(*it);
  }

  //remove output edge of activation
  //connect fused_conv node and nodes after activation nodes
  for (auto& output_edge : output_edges) {
    NodeIndex dst_node_index = output_edge.GetNode().Index();
    int src_arg_index = output_edge.GetSrcArgIndex();
    int dst_arg_index = output_edge.GetDstArgIndex();
    g.RemoveEdge(act.Index(), dst_node_index, src_arg_index, dst_arg_index);
    g.AddEdge(fused_conv.Index(), dst_node_index, 0, dst_arg_index);
  }
}

// get min/max values from Clip if they are constant. Returns false if mutable and cannot be used
static bool GetClipConstantMinMax(const Graph& graph, const Node& node, float& min, float& max) {
  min = std::numeric_limits<float>::lowest();
  max = std::numeric_limits<float>::max();

  // Clip opset 6 has min and max as attributes. they're inputs from opset 11 on.
  bool min_max_are_attributes = graph_utils::IsSupportedOptypeVersionAndDomain(node, "Clip", {6});
  bool can_use = true;

  if (min_max_are_attributes) {
    min = graph_utils::GetNodeAttribute(node, "min")->f();
    max = graph_utils::GetNodeAttribute(node, "max")->f();
  } else {
    // update min/max if provided via a constant initializer
    // return true if value is default or coming from a constant initializer
    // return false if value is mutable
    auto get_constant_value = [&graph](const Node& node, size_t input_idx, float& value) {
      const auto& input_defs = node.InputDefs();
      const NodeArg* input = (input_defs.size() > input_idx) ? input_defs[input_idx] : nullptr;

      if (input == nullptr || !input->Exists()) {
        // optional input not specified so using default value
        return true;
      }

      bool is_constant = true;
      const ONNX_NAMESPACE::TensorProto* initializer = graph_utils::GetConstantInitializer(graph, input->Name());
      if (initializer) {
        Initializer i(*initializer);
        switch (initializer->data_type()) {
          case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
            value = *i.data<float>();
            break;
          // double isn't currently supported
          //case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
          //  value = static_cast<float>(*i.data<double>());
          //  break;
          case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
            value = math::halfToFloat(i.data<BFloat16>()->val);
            break;
          default:
            ORT_THROW("Unexpected data type for Clip input of ", initializer->data_type());
        }
      } else {
        is_constant = false;
      }

      return is_constant;
    };

    // 'min' is input 1, 'max' is input 2. both are optional
    can_use = get_constant_value(node, 1, min) &&
              get_constant_value(node, 2, max);
  }

  return can_use;
}

}  // namespace

Status ConvActivationFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  std::deque<onnxruntime::NodeIndex> removed_nodes;
  for (auto index : order) {
    auto* node = graph.GetNode(index);
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Conv", {1, 11}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        node->GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto& next_node = *(node->OutputNodesBegin());
    if (next_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
      continue;
    }

    // Test if this is an activation that can be fused and also extract the
    // activation's parameters.
    std::vector<float> activation_params;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Relu", {6}) &&
        !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Sigmoid", {6}) &&
        !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Tanh", {6})) {
      if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "LeakyRelu", {6})) {
        activation_params.push_back(graph_utils::GetNodeAttribute(next_node, "alpha")->f());
      } else if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Clip", {6, 11})) {
        float min, max;
        if (GetClipConstantMinMax(graph, next_node, min, max)) {
          activation_params.push_back(min);
          activation_params.push_back(max);
        } else {
          continue;
        }
      } else {
        continue;
      }
    }

    Node& fused_conv = graph.AddNode(graph.GenerateNodeName("fused " + node->Name()), "FusedConv",
                                     "fused Conv " + node->Name() + "with activation " + next_node.OpType(),
                                     node->MutableInputDefs(),
                                     graph.IsNodeOutputsInGraphOutputs(next_node)
                                         ? const_cast<Node&>(next_node).MutableOutputDefs()
                                         : node->MutableOutputDefs(),
                                     &node->GetAttributes(),
                                     "com.microsoft");

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_conv.SetExecutionProviderType(node->GetExecutionProviderType());

    // Add attributes to specify the activation type and parameters.
    fused_conv.AddAttribute("activation", next_node.OpType());
    if (activation_params.size() > 0) {
      fused_conv.AddAttribute("activation_params", activation_params);
    }

    if (!graph.IsNodeOutputsInGraphOutputs(next_node)) {
      HandleActivationNodeEdges(graph, next_node, fused_conv);

      // Replace the input of the node following activation node
      const NodeArg* act_output_def = next_node.OutputDefs()[0];
      NodeArg* fused_conv_output_def = fused_conv.MutableOutputDefs()[0];
      for (auto it = next_node.OutputNodesBegin(); it != next_node.OutputNodesEnd(); ++it) {
        auto output_node = graph.GetNode((*it).Index());
        if (!output_node) {
          return Status(ONNXRUNTIME, INVALID_ARGUMENT);
        }

        auto& input_defs = output_node->MutableInputDefs();
        for (auto& def : input_defs) {
          if (def == act_output_def) {
            def = fused_conv_output_def;
          }
        }
      }
    }

    removed_nodes.push_front(node->Index());
    removed_nodes.push_front(next_node.Index());
  }

  for (auto node : removed_nodes) {
    graph.RemoveNode(node);
  }

  if (!removed_nodes.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
