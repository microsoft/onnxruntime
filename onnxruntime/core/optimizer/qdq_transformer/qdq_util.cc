// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/qdq_util.h"

#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace QDQ {

bool MatchQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, QOpName, {10, 13});
}

bool MatchDQNode(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, DQOpName, {10, 13});
}

bool IsQDQPairSupported(
    const Node& q_node, const Node& dq_node,
    const std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>& get_const_initializer,
    const Path& model_path) {
  ConstPointerContainer<std::vector<NodeArg*>> dq_input_defs = dq_node.InputDefs();
  ConstPointerContainer<std::vector<NodeArg*>> q_input_defs = q_node.InputDefs();

  // Q/DQ contains optional input is not supported
  // non-scalar Q/DQ scale and zero point needs are not supported
  if (dq_input_defs.size() != InputIndex::TOTAL_COUNT ||
      q_input_defs.size() != InputIndex::TOTAL_COUNT ||
      !optimizer_utils::IsScalar(*q_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*q_input_defs[InputIndex::ZERO_POINT_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::ZERO_POINT_ID])) {
    return false;
  }

  // if Q/DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_scale_tensor_proto =
      get_const_initializer(q_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* q_zp_tensor_proto =
      get_const_initializer(q_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  if (nullptr == q_zp_tensor_proto ||
      nullptr == dq_zp_tensor_proto ||
      nullptr == q_scale_tensor_proto ||
      nullptr == dq_scale_tensor_proto) {
    return false;
  }

  // check Q/DQ have same scale and zero point
  Initializer q_zp(*q_zp_tensor_proto, model_path);
  Initializer q_scale(*q_scale_tensor_proto, model_path);
  Initializer dq_zp(*dq_zp_tensor_proto, model_path);
  Initializer dq_scale(*dq_scale_tensor_proto, model_path);

  return q_zp.data_type() == dq_zp.data_type() &&
         *q_zp.data<int8_t>() == *dq_zp.data<int8_t>() &&
         *q_scale.data<float>() == *dq_scale.data<float>();
}

bool IsDQSupported(
    const Node& dq_node,
    const std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)>& get_const_initializer) {
  ConstPointerContainer<std::vector<NodeArg*>> dq_input_defs = dq_node.InputDefs();

  // DQ contains optional input is not supported
  // non-scalar DQ scale and zero point needs are not supported
  if (dq_input_defs.size() != InputIndex::TOTAL_COUNT ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::SCALE_ID]) ||
      !optimizer_utils::IsScalar(*dq_input_defs[InputIndex::ZERO_POINT_ID])) {
    return false;
  }

  // if DQ scale and zero point are not constant, return false
  const ONNX_NAMESPACE::TensorProto* dq_scale_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::SCALE_ID]->Name());
  const ONNX_NAMESPACE::TensorProto* dq_zp_tensor_proto =
      get_const_initializer(dq_input_defs[InputIndex::ZERO_POINT_ID]->Name());
  if (nullptr == dq_zp_tensor_proto ||
      nullptr == dq_scale_tensor_proto) {
    return false;
  }

  return true;
}

static Status RemoveDQQPair(Graph& graph, Node& dq_node, Node& q_node) {
  // remove edge between parent of DQ to DQ
  std::pair<NodeIndex, int> input_edge_info{0, -1};
  auto* dq_input_edge_0 = graph_utils::GetInputEdge(dq_node, 0);
  if (dq_input_edge_0) {
    input_edge_info.first = dq_input_edge_0->GetNode().Index();
    input_edge_info.second = dq_input_edge_0->GetSrcArgIndex();
    graph.RemoveEdge(dq_input_edge_0->GetNode().Index(), dq_node.Index(),
                     dq_input_edge_0->GetSrcArgIndex(), dq_input_edge_0->GetDstArgIndex());
  }

  graph_utils::RemoveNodeOutputEdges(graph, dq_node);  // Remove DQ node output edges

  auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(q_node, 0);
  graph_utils::RemoveNodeOutputEdges(graph, q_node);  // Remove Q node output edges
  for (auto& output_edge : output_edges) {
    // set input NodeArg of Q's children to the 1st input of DQ
    graph.GetNode(output_edge.dst_node)->MutableInputDefs()[output_edge.dst_arg_index] =
        dq_node.MutableInputDefs()[0];

    // add edge between parent of DQ to children of Q
    if (input_edge_info.second != -1) {
      graph.AddEdge(input_edge_info.first, output_edge.dst_node,
                    input_edge_info.second, output_edge.dst_arg_index);
    }
  }

  ORT_RETURN_IF_NOT(graph.RemoveNode(dq_node.Index()));
  ORT_RETURN_IF_NOT(graph.RemoveNode(q_node.Index()));

  return Status::OK();
}

Status CancelOutRedundantDQQPairs(Graph& graph, gsl::span<const NodeIndex> node_indices,
                                  const std::unordered_set<std::string>& compatible_providers,
                                  const logging::Logger& logger,
                                  bool& modified) {
  auto get_const_initializer = [&graph](const std::string& initializer_name) {
    return graph.GetConstantInitializer(initializer_name, true);
  };

  for (NodeIndex node_idx : node_indices) {
    Node* node = graph.GetNode(node_idx);
    if (!node) {
      continue;
    }

    Node& dq_node = *node;
    if (!QDQ::MatchDQNode(dq_node) ||
        !graph_utils::IsSupportedProvider(dq_node, compatible_providers) ||
        // check that DQ has one output edge, does not produce graph output
        !optimizer_utils::CheckOutputEdges(graph, dq_node, 1)) {
      continue;
    }

    Node& q_node = *graph.GetNode(dq_node.OutputNodesBegin()->Index());
    if (!QDQ::MatchQNode(q_node) ||
        !graph_utils::IsSupportedProvider(q_node, compatible_providers) ||
        // check that Q does not produce graph output
        graph.NodeProducesGraphOutput(q_node)) {
      continue;
    }

    // TODO requires scalar scale and zero point which may be stricter than needed
    if (!IsQDQPairSupported(q_node, dq_node, get_const_initializer, graph.ModelPath())) {
      continue;
    }

    LOGS(logger, VERBOSE) << "Removing redundant DQ/Q pair: "
                          << MakeString("(\"", dq_node.Name(), "\", index: ", dq_node.Index(), ")")
                          << " -> "
                          << MakeString("(\"", q_node.Name(), "\", index: ", q_node.Index(), ")");
    ORT_RETURN_IF_ERROR(RemoveDQQPair(graph, dq_node, q_node));
    modified = true;
  }

  return Status::OK();
}

}  // namespace QDQ
}  // namespace onnxruntime
