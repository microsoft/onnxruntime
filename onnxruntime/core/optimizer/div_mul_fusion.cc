// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/div_mul_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

/**
Transform that fuses two Div -> Mul nodes to a single Div node
when the first input to Div is 1.
1 / x1 *  x2 -> x2 / x1
 */
Status DivMulFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Div", {7, 13}) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    const auto& next_node = *node.OutputNodesBegin();
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Mul", {7, 13}) ||
        next_node.GetOutputEdgesCount() != 1 ||
        // Make sure the two nodes do not span execution providers.
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    // Check that the appropriate input to the Div node is a constant.
    if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[0])) {
      continue;
    }

    const auto* initializer = graph_utils::GetConstantInitializer(graph, node.InputDefs()[0]->Name());
    ORT_ENFORCE(initializer);
    if (!initializer) {
      continue;
    }

    int32_t data_type = initializer->data_type();
    Initializer div_A(*initializer, graph.ModelPath());
    if (div_A.size() > 1) {
      continue;
    }
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        if (*div_A.data<float>() != 1.f) {
          continue;
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        if (math::halfToFloat(div_A.data<MLFloat16>()->val) != 1.f) {
          continue;
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
        if (*div_A.data<double>() != static_cast<double>(1.f)) {
          continue;
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        if (*div_A.data<int32_t>() != static_cast<int32_t>(1)) {
          continue;
        }
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        if (*div_A.data<int64_t>() != static_cast<int64_t>(1)) {
          continue;
        }
        break;
      default:
        continue;
    }

    if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
      continue;
    }
    auto& div_node = node;
    auto& mul_node = *graph.GetNode(div_node.OutputNodesBegin()->Index());  // get mutable next node
    const auto& div_output = div_node.OutputDefs();
    auto& mul_inputs = mul_node.MutableInputDefs();

    //get other input of mul
    auto& other_input = mul_inputs[0] == div_output[0] ? mul_inputs[1] : mul_inputs[0];

    graph_utils::ReplaceNodeInput(div_node, 0, *other_input);
    // move the output definition and edges from the mul_node to the div_node and delete the mul_node
    graph_utils::FinalizeNodeFusion(graph, div_node, mul_node);
  }

  modified = true;

  return Status::OK();
}
}  // namespace onnxruntime
