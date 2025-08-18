// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/where_dummy_dq.h"

#include "core/framework/tensorprotoutils.h"
#include "core/common/common.h"
#include "core/util/qmath.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {
bool WhereDummyDq::SatisfyCondition(const Graph& graph, const Node& node) const {
  if (!(node.OpType() == "Where")) {
    return false;
  }
  const auto& where_inputs = node.InputDefs();
  const Node* parent_node_1 = graph.GetProducerNode(where_inputs[1]->Name());
  const Node* parent_node_2 = graph.GetProducerNode(where_inputs[2]->Name());

  bool is_p1_dq = (parent_node_1 && parent_node_1->OpType() == QDQ::DQOpName);
  bool is_p2_dq = (parent_node_2 && parent_node_2->OpType() == QDQ::DQOpName);

  // WhereDummyDq focus on WhereOp with one DQ input and one scalar initializer input
  if (is_p1_dq && !parent_node_2) {
    return (where_inputs[2]->Shape()->dim_size() == 0);
  }
  if (!parent_node_1 && is_p2_dq) {
    return (where_inputs[1]->Shape()->dim_size() == 0);
  }
  return false;
}

Status WhereDummyDq::InsertDummyDQ(Node& node, Graph& graph, bool& modified, const logging::Logger& logger) const {
  const auto& where_inputs = node.InputDefs();
  const Node* parent_node_1 = graph.GetProducerNode(where_inputs[1]->Name());
  const Node* parent_node_2 = graph.GetProducerNode(where_inputs[2]->Name());

  // With SatisfyCondition, we must have one DQ and one initializer
  const Node* dq_node = parent_node_1 ? parent_node_1 : parent_node_2;
  int const_idx = parent_node_1 ? 2 : 1;

  const ONNX_NAMESPACE::TensorProto* dq_node_scale_proto = nullptr;
  graph.GetInitializedTensor(dq_node->InputDefs()[1]->Name(), dq_node_scale_proto);
  const ONNX_NAMESPACE::TensorProto* dq_node_zp_proto = nullptr;
  graph.GetInitializedTensor(dq_node->InputDefs()[2]->Name(), dq_node_zp_proto);

  // Dummy data initializer.
  ONNX_NAMESPACE::TensorProto dummy_data_proto;
  dummy_data_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_data"));
  // Set data type to dq node's zp dtype
  dummy_data_proto.set_data_type(dq_node_zp_proto->data_type());

  // Dummy zero point initializer.
  ONNX_NAMESPACE::TensorProto dummy_zp_proto;
  dummy_zp_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_zp"));
  dummy_zp_proto.set_data_type(dq_node_zp_proto->data_type());

  switch (dummy_zp_proto.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      int8_t zp = 0;
      int8_t dummy_data = 1;
      dummy_zp_proto.set_raw_data(&zp, 1);
      dummy_data_proto.set_raw_data(&dummy_data, 1);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      uint8_t zp = 0;
      uint8_t dummy_data = 1;
      dummy_zp_proto.set_raw_data(&zp, 1);
      dummy_data_proto.set_raw_data(&dummy_data, 1);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      int16_t zp = 0;
      int16_t dummy_data = 1;
      dummy_zp_proto.set_raw_data(&zp, 2);
      dummy_data_proto.set_raw_data(&dummy_data, 2);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      uint16_t zp = 0;
      uint16_t dummy_data = 1;
      dummy_zp_proto.set_raw_data(&zp, 2);
      dummy_data_proto.set_raw_data(&dummy_data, 2);
      break;
    }
    default:
      LOGS(logger, WARNING) << "Currently support existing DQ's zero point with INT8, UINT8, INT16, UINT16";
      return Status::OK();
  }

  // Set dummy scale to the original value
  const ONNX_NAMESPACE::TensorProto* const_node_data_proto = nullptr;
  graph.GetInitializedTensor(where_inputs[const_idx]->Name(), const_node_data_proto);
  Initializer initializer(graph, *const_node_data_proto, graph.ModelPath());
  if (dq_node_scale_proto->data_type() != const_node_data_proto->data_type()) {
    // WhereDummyDq fills the const value to the dummy DQ's scale
    LOGS(logger, WARNING) << "Currently only support existing DQ's scale with same datatype as scalar";
    return Status::OK();
  }

  // Dummy scale initializer.
  ONNX_NAMESPACE::TensorProto dummy_scale_proto;
  dummy_scale_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_scale"));
  dummy_scale_proto.set_data_type(dq_node_scale_proto->data_type());
  switch (initializer.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      float* where_const_scalar = initializer.data<float>();
      dummy_scale_proto.set_raw_data(where_const_scalar, sizeof(float));
      break;
    }
    default:
      LOGS(logger, WARNING) << "Currently support scalar with FLOAT";
      return Status::OK();
  }

  // Start editing the graph
  NodeArg& dummy_data_arg = graph_utils::AddInitializer(graph, dummy_data_proto);
  NodeArg& dummy_scale_arg = graph_utils::AddInitializer(graph, dummy_scale_proto);
  NodeArg& dummy_zp_arg = graph_utils::AddInitializer(graph, dummy_zp_proto);

  ONNX_NAMESPACE::TypeProto dummy_dq_type_proto = utils::TypeProtoFromTensorProto(*const_node_data_proto);
  dummy_dq_type_proto.mutable_tensor_type()->set_elem_type(const_node_data_proto->data_type());
  NodeArg& dummy_dq_arg =
      graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_dummy_dq"), &dummy_dq_type_proto);
  Node& dummy_dq_node =
      graph.AddNode(
          graph.GenerateNodeArgName(node.Name() + "_dummy_dq"),
          QDQ::DQOpName,
          "DeQuantizeLinear from WhereDummyDq GraphTransformer",
          {&dummy_data_arg, &dummy_scale_arg, &dummy_zp_arg},
          {&dummy_dq_arg},
          nullptr,
          dq_node->Domain());

  node.MutableInputDefs()[const_idx] = &dummy_dq_arg;
  if (graph.GetConsumerNodes(where_inputs[const_idx]->Name()).size() == 0) {
    graph.RemoveInitializedTensor(where_inputs[const_idx]->Name());
  }
  graph.AddEdge(dummy_dq_node.Index(), node.Index(), 0, const_idx);
  modified = true;

  return Status::OK();
}

Status WhereDummyDq::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  const GraphViewer graph_viewer{graph};
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (const auto node_idx : node_indices) {
    auto* node_ptr = graph.GetNode(node_idx);
    if (!node_ptr) {
      continue;
    }

    Node& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (this->SatisfyCondition(graph, node)) {
      ORT_RETURN_IF_ERROR(WhereDummyDq::InsertDummyDQ(node, graph, modified, logger));
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime