// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/relu_quantizelinear.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status ReluQuantTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_relu = graph.GetNode(node_index);
    if (p_relu == nullptr)
      continue;  // node removed as part of an earlier fusion

    Node& relu = *p_relu;
    ORT_RETURN_IF_ERROR(Recurse(relu, modified, graph_level, logger));

    if (relu.OpType() != "Relu" ||
        !graph_utils::MatchesOpSetDomain(relu, kOnnxDomain) ||
        !graph_utils::IsSupportedProvider(relu, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, relu, 1)) {
      continue;
    }

    Node& q_node = *graph.GetNode(relu.OutputNodesBegin()->Index());
    if (q_node.OpType() != "QuantizeLinear" ||
        !graph_utils::MatchesOpSetDomain(q_node, kOnnxDomain) ||
        !graph_utils::IsSupportedProvider(q_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    std::vector<NodeArg*>& q_input_defs = q_node.MutableInputDefs();

    constexpr size_t q_input_cnt_required = 3;
    if (q_input_defs.size() != q_input_cnt_required) {
      continue;
    }

    constexpr size_t zp_idx = 2;
    const ONNX_NAMESPACE::TensorProto* zp_tensor_proto = nullptr;
    if (!graph_utils::NodeArgIsConstant(graph, *q_input_defs[zp_idx]) ||
        !graph.GetInitializedTensor(q_input_defs[zp_idx]->Name(), zp_tensor_proto)) {
      continue;
    }

    using ONNX_TENSOR_ELEM_TYPE = ONNX_NAMESPACE::TensorProto::DataType;
    Initializer zero_point(*zp_tensor_proto, graph.ModelPath());
    if (zero_point.size() != 1 ||
        zero_point.data_type() == ONNX_TENSOR_ELEM_TYPE::TensorProto_DataType_INT8 && zero_point.data<int8_t>()[0] != -128 ||
        zero_point.data_type() == ONNX_TENSOR_ELEM_TYPE::TensorProto_DataType_UINT8 && zero_point.data<uint8_t>()[0] != 0) {
      continue;
    }

    graph_utils::RemoveNodeOutputEdges(graph, relu);
    q_input_defs[0] = relu.MutableInputDefs()[0];
    graph.RemoveNode(relu.Index());

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
