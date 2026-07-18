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
  // This transformer targets a very specific pattern around `Where` when used in a QDQ graph:
  //   cond, DQ(xq), const_scalar  -> Where -> Q(yq)
  // or
  //   cond, const_scalar, DQ(xq)  -> Where -> Q(yq)
  //
  // When one `Where` branch is a scalar initializer (no producer node), WhereNodeGroupSelector
  // requires both data branches to be produced by DQ nodes so the `Where` can be grouped into a
  // single node-unit. We insert a "dummy" DQ for the scalar branch to satisfy that requirement.
  if (!(node.OpType() == "Where")) {
    return false;
  }

  // ONNX Where inputs: [0]=condition, [1]=X, [2]=Y
  const auto& where_inputs = node.InputDefs();
  const auto& where_outputs = node.OutputDefs();

  const Node* parent_node_1 = graph.GetProducerNode(where_inputs[1]->Name());
  const Node* parent_node_2 = graph.GetProducerNode(where_inputs[2]->Name());

  // Only apply when the `Where` output is immediately consumed by a single QuantizeLinear.
  // If there are multiple consumers (or not a Q), inserting an extra DQ would not help form a
  // clean QDQ node-unit and may create additional overhead.
  std::vector<const Node*> child_nodes = graph.GetConsumerNodes(where_outputs[0]->Name());
  if (child_nodes.size() != 1 || child_nodes[0]->OpType() != QDQ::QOpName) {
    return false;
  }

  const bool is_p1_dq = (parent_node_1 && parent_node_1->OpType() == QDQ::DQOpName);
  const bool is_p2_dq = (parent_node_2 && parent_node_2->OpType() == QDQ::DQOpName);

  // We require exactly one branch to be fed by a DQ and the other branch to be a scalar initializer
  // (represented as a NodeArg with rank 0 shape and no producer node).
  if (is_p1_dq && graph_utils::IsConstantInitializer(graph, where_inputs[2]->Name(), true)) {
    return where_inputs[2]->HasTensorOrScalarShape() ? (where_inputs[2]->Shape()->dim_size() == 0) : false;
  }
  if (graph_utils::IsConstantInitializer(graph, where_inputs[1]->Name(), true) && is_p2_dq) {
    return where_inputs[1]->HasTensorOrScalarShape() ? (where_inputs[1]->Shape()->dim_size() == 0) : false;
  }

  return false;
}

Status WhereDummyDq::InsertDummyDQ(Node& node, Graph& graph, bool& modified, const logging::Logger& logger) const {
  // Inserts a DeQuantizeLinear node on the scalar initializer branch of `Where` so that both
  // data branches (X and Y) are produced by DQ nodes, enabling downstream QDQ grouping.
  const auto& where_inputs = node.InputDefs();
  const auto& where_outputs = node.OutputDefs();
  const Node* parent_node_1 = graph.GetProducerNode(where_inputs[1]->Name());
  const Node* parent_node_2 = graph.GetProducerNode(where_inputs[2]->Name());
  const Node* child_node = graph.GetConsumerNodes(where_outputs[0]->Name())[0];

  // From SatisfyCondition():
  // - exactly one of parent_node_1/parent_node_2 is a DQ node
  // - the other input is a scalar initializer (rank-0 tensor) with no producer node
  const Node* dq_node = parent_node_1 ? parent_node_1 : parent_node_2;
  const int const_idx = parent_node_1 ? 2 : 1;

  // Guardrail: only insert dummy DQ when the quantized dtype matches the output Q's dtype.
  // If they differ, we cannot safely synthesize quantization parameters.
  const int32_t dt_input = dq_node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  const int32_t dt_output = child_node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
  if (dt_input != dt_output) {
    LOGS(logger, WARNING) << "WhereDummyDq: skip inserting dummy DQ due to mismatched quantized dtype between input DQ "
                             "and output Q. DQ input dtype="
                          << dt_input << ", Q output dtype=" << dt_output;
    return Status::OK();
  }

  const ONNX_NAMESPACE::TensorProto* dq_node_scale_proto = nullptr;
  if (!graph.GetInitializedTensor(dq_node->InputDefs()[1]->Name(), dq_node_scale_proto) ||
      dq_node_scale_proto == nullptr) {
    LOGS(logger, WARNING) << "WhereDummyDq expects dq branch to have an initializer scale. "
                          << "DQ: " << dq_node->Name();
    return Status::OK();
  };

  const ONNX_NAMESPACE::TensorProto* dq_node_zp_proto = nullptr;
  if (!graph.GetInitializedTensor(dq_node->InputDefs()[2]->Name(), dq_node_zp_proto) ||
      dq_node_zp_proto == nullptr) {
    LOGS(logger, WARNING) << "WhereDummyDq expects dq branch to have an initializer zero point. "
                          << "DQ: " << dq_node->Name();
    return Status::OK();
  };

  // Create initializers for the dummy DQ input triplet: (xq, scale, zero_point).
  // We choose values so that DeQuantizeLinear(dummy_xq, dummy_scale, dummy_zp) reconstructs
  // the original scalar float value as closely as possible.
  //
  // Note: We only support float scalar constants currently.
  // Dummy data initializer.
  ONNX_NAMESPACE::TensorProto dummy_xq_proto;
  dummy_xq_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_xq"));
  // Set data type to dq node's zp dtype
  dummy_xq_proto.set_data_type(dq_node_zp_proto->data_type());

  // Dummy zero point initializer.
  ONNX_NAMESPACE::TensorProto dummy_zp_proto;
  dummy_zp_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_zp"));
  dummy_zp_proto.set_data_type(dq_node_zp_proto->data_type());

  // Dummy scale initializer.
  ONNX_NAMESPACE::TensorProto dummy_scale_proto;
  dummy_scale_proto.set_name(graph.GenerateNodeArgName(node.Name() + "_dummy_scale"));
  dummy_scale_proto.set_data_type(dq_node_scale_proto->data_type());

  // Get original float input
  const ONNX_NAMESPACE::TensorProto* const_node_data_proto = nullptr;
  graph.GetInitializedTensor(where_inputs[const_idx]->Name(), const_node_data_proto);
  Initializer initializer(graph, *const_node_data_proto, graph.ModelPath());
  if (dq_node_scale_proto->data_type() != const_node_data_proto->data_type()) {
    // WhereDummyDq fills the const value to the dummy DQ's scale
    LOGS(logger, WARNING) << "Currently only support existing DQ's scale with same datatype as scalar. "
                          << "DQ: " << dq_node->Name() << ", scalar(const): " << where_inputs[const_idx]->Name();
    return Status::OK();
  }
  float dummy_xf = 0;
  switch (initializer.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      dummy_xf = *initializer.data<float>();
      break;
    }
    default:
      LOGS(logger, WARNING) << "Unsupported dtype of constant input. "
                            << "DQ: " << dq_node->Name() << ", scalar(const): " << where_inputs[const_idx]->Name();
      return Status::OK();
  }

  // TensorProto stores INT8/UINT8/INT16/UINT16 values via `int32_data`.
  // Keep values in-range for unsigned cases (0..255 / 0..65535) before writing.
  int32_t dummy_zp_i32 = 0;
  int32_t dummy_xq_i32 = 0;
  float dummy_scale = 1.0f;

  switch (dummy_zp_proto.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      dummy_zp_i32 = 0;
      dummy_xq_i32 = (dummy_xf > 0) ? 127 : ((dummy_xf == 0) ? dummy_zp_i32 : -128);
      dummy_scale = (dummy_xf == 0) ? 1 : (float)dummy_xf / (dummy_xq_i32 - dummy_zp_i32);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
      dummy_zp_i32 = 127;
      dummy_xq_i32 = (dummy_xf > 0) ? 255 : ((dummy_xf == 0) ? dummy_zp_i32 : 0);
      dummy_scale = (dummy_xf == 0) ? 1 : (float)dummy_xf / (dummy_xq_i32 - dummy_zp_i32);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      dummy_zp_i32 = 0;
      dummy_xq_i32 = (dummy_xf > 0) ? 32767 : ((dummy_xf == 0) ? dummy_zp_i32 : -32768);
      dummy_scale = (dummy_xf == 0) ? 1 : (float)dummy_xf / (dummy_xq_i32 - dummy_zp_i32);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
      dummy_zp_i32 = 32767;
      dummy_xq_i32 = (dummy_xf > 0) ? 65535 : ((dummy_xf == 0) ? dummy_zp_i32 : 0);
      dummy_scale = (dummy_xf == 0) ? 1 : (float)dummy_xf / (dummy_xq_i32 - dummy_zp_i32);
      break;
    }
    default:
      LOGS(logger, WARNING) << "Currently support existing DQ's zero point with INT8, UINT8, INT16, UINT16. "
                            << "DQ: " << dq_node->Name() << ", scalar(const): " << where_inputs[const_idx]->Name();
      return Status::OK();
  }

  dummy_zp_proto.add_int32_data(dummy_zp_i32);
  dummy_xq_proto.add_int32_data(dummy_xq_i32);
  dummy_scale_proto.add_float_data(dummy_scale);

  // Start editing the graph:
  // - add the initializers
  // - add a DeQuantizeLinear node consuming them
  // - rewire the scalar branch of `Where` to use the DQ output
  // - drop the original scalar initializer if it becomes unused
  NodeArg& dummy_xq_arg = graph_utils::AddInitializerWithOrtValue(graph, dummy_xq_proto);
  NodeArg& dummy_scale_arg = graph_utils::AddInitializerWithOrtValue(graph, dummy_scale_proto);
  NodeArg& dummy_zp_arg = graph_utils::AddInitializerWithOrtValue(graph, dummy_zp_proto);

  ONNX_NAMESPACE::TypeProto dummy_dq_type_proto = utils::TypeProtoFromTensorProto(*const_node_data_proto);
  dummy_dq_type_proto.mutable_tensor_type()->set_elem_type(const_node_data_proto->data_type());
  NodeArg& dummy_dq_arg =
      graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node.Name() + "_dummy_dq"), &dummy_dq_type_proto);
  Node& dummy_dq_node =
      graph.AddNode(
          graph.GenerateNodeArgName(node.Name() + "_dummy_dq"),
          QDQ::DQOpName,
          "DeQuantizeLinear from WhereDummyDq GraphTransformer",
          {&dummy_xq_arg, &dummy_scale_arg, &dummy_zp_arg},
          {&dummy_dq_arg},
          node,
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
