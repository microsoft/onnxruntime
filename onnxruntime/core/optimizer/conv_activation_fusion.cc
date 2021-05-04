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
// get min/max values from Clip if they are constant. Returns false if mutable and cannot be used
static bool GetClipConstantMinMax(const Graph& graph, const Node& node, float& min, float& max) {
  min = std::numeric_limits<float>::lowest();
  max = std::numeric_limits<float>::max();

  // Clip opset 6 has min and max as attributes. they're inputs from opset 11 on.
  bool min_max_are_attributes = graph_utils::IsSupportedOptypeVersionAndDomain(node, "Clip", {6});
  bool min_max_are_constant_values = true;

  if (min_max_are_attributes) {
    min = graph_utils::GetNodeAttribute(node, "min")->f();
    max = graph_utils::GetNodeAttribute(node, "max")->f();
  } else {
    // update min/max if provided via a constant initializer
    // return true if value is default or coming from a constant initializer and update 'value'
    // return false if value is mutable
    auto update_if_constant_value = [&graph](const Node& node, size_t input_idx, float& value) {
      const auto& input_defs = node.InputDefs();
      const NodeArg* input = (input_defs.size() > input_idx) ? input_defs[input_idx] : nullptr;

      if (input == nullptr || !input->Exists()) {
        // optional input not specified so using default value
        return true;
      }

      bool is_constant = true;
      const ONNX_NAMESPACE::TensorProto* initializer = graph_utils::GetConstantInitializer(graph, input->Name());
      if (initializer) {
        Initializer i(*initializer, graph.ModelPath());
        switch (initializer->data_type()) {
          case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
            value = *i.data<float>();
            break;
          // double isn't currently supported
          //case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
          //  value = static_cast<float>(*i.data<double>());
          //  break;
          case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
            value = math::halfToFloat(i.data<MLFloat16>()->val);
            break;
          default:
            ORT_THROW("Unexpected data type for Clip input of ", initializer->data_type());
        }
      } else {
        is_constant = false;
      }

      return is_constant;
    };

    // 'min' is input 1, 'max' is input 2. both are optional.
    // if the input is constant, 'min' or 'max' is updated by the call to get_if_constant_value
    min_max_are_constant_values = update_if_constant_value(node, 1, min) &&
                                  update_if_constant_value(node, 2, max);
  }

  return min_max_are_constant_values;
}

}  // namespace

Status ConvActivationFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node = graph.GetNode(index);
    // check that node hasn't already been removed
    if (!node)
      continue;

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Conv", {1, 11}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        node->GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto& next_node = *(node->OutputNodesBegin());

    if (next_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
      continue;
    }

    if (!graph.GetNodeOutputsInGraphOutputs(*node).empty()) {
      continue;
    }

    if (node->GetExecutionProviderType() == onnxruntime::kCudaExecutionProvider) {
      if (node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() != 
          ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        continue;
      }
      if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Relu", {6, 13})) {
        Node& conv_node = *node;
        Node& act_node = *graph.GetNode(next_node.Index());
        auto node_name = graph.GenerateNodeName(conv_node.Name() + "_" + act_node.Name());
        Node& fused_conv = graph.AddNode(node_name,
                                         "FusedConv",
                                         node_name,
                                         conv_node.MutableInputDefs(),
                                         {},
                                         &conv_node.GetAttributes(),
                                         onnxruntime::kMSDomain);
        fused_conv.SetExecutionProviderType(conv_node.GetExecutionProviderType());
        fused_conv.AddAttribute("activation", "Relu");
        graph_utils::FinalizeNodeFusion(graph, {conv_node, act_node}, fused_conv);
        modified = true;
      } else if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Add", {6, 7, 13})) {
        const auto& last_node = *(next_node.OutputNodesBegin());
        if (last_node.GetExecutionProviderType() != node->GetExecutionProviderType()) {
          continue;
        }
        if (graph_utils::IsSupportedOptypeVersionAndDomain(last_node, "Relu", {6, 13}) && 
            next_node.GetOutputEdgesCount() == 1) {
          Node& conv_node = *node;
          Node& add_node = *graph.GetNode(next_node.Index());
          Node& act_node = *graph.GetNode(last_node.Index());
          auto conv_inputs = conv_node.MutableInputDefs();
          auto conv_outputs = conv_node.MutableOutputDefs();
          auto add_inputs = add_node.MutableInputDefs();
          for (auto add_input : add_inputs) {
            if (add_input->Name() != conv_outputs[0]->Name()) {
              conv_inputs.push_back(add_input);
              break;
            }
          }
          auto node_name = graph.GenerateNodeName(conv_node.Name() + "_" +
                                                  add_node.Name() + "_" +
                                                  act_node.Name());
          Node& fused_conv = graph.AddNode(node_name,
                                           "FusedConv",
                                           node_name,
                                           conv_inputs,
                                           {}, &conv_node.GetAttributes(),
                                           onnxruntime::kMSDomain);
          fused_conv.SetExecutionProviderType(conv_node.GetExecutionProviderType());
          fused_conv.AddAttribute("activation", "Relu");
          graph_utils::FinalizeNodeFusion(graph, {conv_node, add_node, act_node}, fused_conv);
          modified = true;
        }
      }
    } else {
      // Test if this is an activation that can be fused and also extract the
      // activation's parameters.
      std::vector<float> activation_params;
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Relu", {6, 13}) &&
          !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Sigmoid", {6, 13}) &&
          !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Tanh", {6, 13})) {
        if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "LeakyRelu", {6})) {
          activation_params.push_back(graph_utils::GetNodeAttribute(next_node, "alpha")->f());
        } else if (graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Clip", {6, 11, 12, 13})) {
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

      Node& conv_node = *node;
      Node& act_node = *graph.GetNode(next_node.Index());

      Node& fused_conv = graph.AddNode(graph.GenerateNodeName("fused " + conv_node.Name()), "FusedConv",
                                       "fused Conv " + conv_node.Name() + "with activation " + act_node.OpType(),
                                       conv_node.MutableInputDefs(),
                                       {},
                                       &conv_node.GetAttributes(),
                                       "com.microsoft");

      // Assign provider to this new node. Provider should be same as the provider for old node.
      fused_conv.SetExecutionProviderType(conv_node.GetExecutionProviderType());

      // Add attributes to specify the activation type and parameters.
      fused_conv.AddAttribute("activation", next_node.OpType());
      if (activation_params.size() > 0) {
        fused_conv.AddAttribute("activation_params", activation_params);
      }

      // move output definitions and edges from act_node to fused_conv. delete conv_node and act_node.
      graph_utils::FinalizeNodeFusion(graph, {conv_node, act_node}, fused_conv);

      modified = true;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime
