// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// DO NOT include ORT header files as this is meant to be a header-only utility that can be copied
// to other projects.

#ifndef INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_QDQ_UTILS_H_
#define INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_QDQ_UTILS_H_

#include <functional>
#include <optional>
#include <set>
#include <vector>

#include "core/session/onnxruntime_cxx_api.h"

namespace OrtEpUtils {
namespace QDQ {

class EdgeEnd {
 public:
  EdgeEnd(const OrtNode& node, int src_arg_index, int dst_arg_index) noexcept;

  const OrtNode& GetNode() const noexcept { return *node_; }
  int GetSrcArgIndex() const { return src_arg_index_; }
  int GetDstArgIndex() const { return dst_arg_index_; }

 private:
  const OrtNode* node_;
  const int src_arg_index_;
  const int dst_arg_index_;
};

struct EdgeEndCompare {
  bool operator()(const EdgeEnd& lhs, const EdgeEnd& rhs) const;
};

using EdgeSet = std::set<EdgeEnd, EdgeEndCompare>;
using EdgeConstIterator = EdgeSet::const_iterator;

// Struct to represent a DequantizeLinear -> Op -> QuantizeLinear node group
struct NodeGroup {
  std::vector<const OrtNode*> dq_nodes;
  std::vector<const OrtNode*> q_nodes;
  const OrtNode* target_node;
  const OrtNode* redundant_clip_node;  // optional

  // Validator to check if the set of nodes can form a valid QDQ NodeGroup.
  // Checks target node is only consumer of each DQ, and that the outputs remain valid if the QDQ node group was to
  // be converted into a single node with a quantized operator.
  static Ort::Status CanCreateNodeGroup(const OrtGraph& graph,
                                        const OrtNode& target_node,
                                        const OrtNode* redundant_clip_node,
                                        const std::vector<const OrtNode*>& dq_nodes,
                                        const std::vector<const OrtNode*>& q_nodes);
};

// Definition of one input or output
// If the optional quant_param is present, then this is a quantized input,
// otherwise this is a regular input
struct NodeUnitIODef {
  // The quantization parameter. Scale is mandatory. Zero-point and axis are optional.
  struct QuantParam {
    const OrtValueInfo& scale;
    const OrtValueInfo* zero_point{nullptr};
    std::optional<int64_t> axis{std::nullopt};
  };

  const OrtValueInfo& value_info;
  const std::optional<QuantParam> quant_param;
};

class NodeUnit {
 private:
  struct PrivateTag {};

 public:
  // NodeUnit type
  enum class Type : uint8_t {
    SingleNode,  // The NodeUnit contains a single node
    QDQGroup,    // The NodeUnit contain a QDQ group of nodes, such as "DQ->Sigmoid->Q"
  };

  NodeUnit(const OrtNode& target_node, Type type, PrivateTag tag);

  static Ort::Status MakeSingleNode(const OrtNode& node, /*out*/ std::unique_ptr<NodeUnit>& node_unit);
  static Ort::Status MakeQDQGroup(const OrtGraph& graph, const NodeGroup& node_group,
                                  /*out*/ std::unique_ptr<NodeUnit>& node_unit);

 private:
  std::vector<const OrtNode*> dq_nodes_;  // dq nodes for this NodeUnit, not necessarily all inputs
  const OrtNode& target_node_;
  const OrtNode* redundant_clip_node_ = nullptr;  // Optional redundant clip node for the QDQ group, nullptr if not present.
  std::vector<const OrtNode*> q_nodes_;           // q-nodes for this NodeUnit. not necessarily all outputs
  Type type_;

  std::vector<NodeUnitIODef> inputs_;
  std::vector<NodeUnitIODef> outputs_;

  size_t input_edge_count_ = 0;  // total number of input edges

  // output edges, hiding any Q nodes involved. src_idx will be value from target node. only used for QDQ node group.
  EdgeSet output_edges_;
};

}  // namespace QDQ
}  // namespace OrtEpUtils

// End of header
#endif  // INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_QDQ_UTILS_H_

//
// IMPLEMENTATION BELOW
//
#ifdef ORT_EP_UTILS_QDQ_UTILS_IMPL

#include <algorithm>
#include <cassert>
#include <sstream>
#include <utility>

#ifndef ORT_EP_UTILS_C_RETURN_IF_ERROR
#define ORT_EP_UTILS_C_RETURN_IF_ERROR(fn) \
  do {                                     \
    OrtStatus* _status = (fn);             \
    if (_status != nullptr) {              \
      return Ort::Status{_status};         \
    }                                      \
  } while (0)
#endif

#ifndef ORT_EP_UTILS_CXX_RETURN_IF_ERROR
#define ORT_EP_UTILS_CXX_RETURN_IF_ERROR(fn) \
  do {                                       \
    Ort::Status _status = (fn);              \
    if (!_status.IsOK()) {                   \
      return _status;                        \
    }                                        \
  } while (0)
#endif

#ifndef ORT_EP_UTILS_C_RETURN_IF
#define ORT_EP_UTILS_C_RETURN_IF(cond, ort_api, msg)               \
  do {                                                             \
    if ((cond)) {                                                  \
      return Ort::Status{(ort_api).CreateStatus(ORT_FAIL, (msg))}; \
    }                                                              \
  } while (0)
#endif

namespace OrtEpUtils {
namespace QDQ {

namespace {
Ort::Status GetNodeInputs(const OrtNode& node, std::vector<const OrtValueInfo*>& node_inputs) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t num_inputs = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumInputs(&node, &num_inputs));

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetInputs(&node, inputs.data(), inputs.size()));

  node_inputs = std::move(inputs);
  return Ort::Status{nullptr};
}

Ort::Status GetNodeOutputs(const OrtNode& node, std::vector<const OrtValueInfo*>& node_outputs) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t num_outputs = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(&node, &num_outputs));

  std::vector<const OrtValueInfo*> outputs(num_outputs);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOutputs(&node, outputs.data(), outputs.size()));

  node_outputs = std::move(outputs);
  return Ort::Status{nullptr};
}

Ort::Status GetAxisAttribute(const OrtNode& node, std::optional<int64_t>& axis) {
  const OrtApi& ort_api = Ort::GetApi();

  const OrtOpAttr* axis_attr = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetAttributeByName(&node, "axis", &axis_attr));

  axis = std::nullopt;

  if (axis_attr != nullptr) {
    OrtOpAttrType attr_type = OrtOpAttrType::ORT_OP_ATTR_UNDEFINED;
    int64_t axis_val = 0;
    size_t attr_bytes = 0;

    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.OpAttr_GetType(axis_attr, &attr_type));
    ORT_EP_UTILS_C_RETURN_IF(attr_type != OrtOpAttrType::ORT_OP_ATTR_INT, ort_api, "Axis attr should be INT");
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ReadOpAttr(axis_attr, attr_type, &axis_val, sizeof(axis_val), &attr_bytes));

    axis = axis_val;
  }

  return Ort::Status{nullptr};
}

Ort::Status GetNodeInputEdgeCount(const std::vector<const OrtValueInfo*>& inputs, size_t& num_input_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  // Sum the number of inputs with a producer node.
  num_input_edges = 0;

  for (const OrtValueInfo* input : inputs) {
    if (input == nullptr) continue;  // Skip missing optional input

    const OrtNode* producer_node = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueProducer(input, &producer_node, /*output_index*/ nullptr));
    num_input_edges += static_cast<size_t>(producer_node != nullptr);
  }

  return Ort::Status{nullptr};
}

Ort::Status GetNodeInputEdgeCount(const OrtNode& node, size_t& num_input_edges) {
  std::vector<const OrtValueInfo*> inputs;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(node, inputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputEdgeCount(inputs, num_input_edges));
  return Ort::Status{nullptr};
}

Ort::Status GetNodeInputEdges(const std::vector<const OrtValueInfo*>& inputs, EdgeSet& input_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
    const OrtValueInfo* input = inputs[input_idx];
    if (input == nullptr) continue;  // Skip missing optional input

    const OrtNode* producer_node = nullptr;
    size_t producer_output_index = 0;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueProducer(input, &producer_node, &producer_output_index));

    if (producer_node == nullptr) {
      continue;  // No producer node (e.g., a graph input)
    }

    input_edges.insert(EdgeEnd(*producer_node, static_cast<int>(producer_output_index),
                               static_cast<int>(input_idx)));
  }

  return Ort::Status{nullptr};
}

Ort::Status GetNodeInputEdges(const OrtNode& node, EdgeSet& input_edges) {
  std::vector<const OrtValueInfo*> inputs;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(node, inputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputEdges(inputs, input_edges));
  return Ort::Status{nullptr};
}

Ort::Status GetNodeOutputEdges(const std::vector<const OrtValueInfo*>& outputs, EdgeSet& output_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  for (size_t src_idx = 0; src_idx < outputs.size(); ++src_idx) {
    const OrtValueInfo* value_info = outputs[src_idx];

    size_t num_consumers = 0;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers));

    std::vector<const OrtNode*> consumer_nodes(num_consumers);
    std::vector<int64_t> consumer_indices(num_consumers);

    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueConsumers(value_info, consumer_nodes.data(),
                                                                       consumer_indices.data(), num_consumers));

    // Build up an edge with consumer nodes as the destinations.
    for (size_t c_idx = 0; c_idx < num_consumers; c_idx++) {
      const OrtNode* dst_node = consumer_nodes[c_idx];
      int64_t dst_idx = consumer_indices[c_idx];

      if (dst_idx < 0) {
        continue;  // Skip implicit input to consumer node.
      }

      EdgeEnd edge_end(*dst_node, static_cast<int>(src_idx), static_cast<int>(dst_idx));
      output_edges.insert(edge_end);
    }
  }

  return Ort::Status{nullptr};
}

Ort::Status GetNodeOutputEdges(const OrtNode& node, EdgeSet& output_edges) {
  std::vector<const OrtValueInfo*> outputs;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(node, outputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(outputs, output_edges));
  return Ort::Status{nullptr};
}

enum class QLinearOpType : uint8_t {
  Unknown,  // Unknown or not a linear quantized op
  DequantizeLinear,
  QuantizeLinear,
  QLinearConv,
  QLinearMatMul,
  QLinearAdd,
  QLinearSigmoid,
  QLinearAveragePool,
  QLinearMul,
  QLinearReduceMean,
  QLinearConcat,
  QLinearGlobalAveragePool,
  QLinearLeakyRelu,
};

QLinearOpType GetQLinearOpType(const std::string& op_type) {
  if (op_type == "DequantizeLinear")
    return QLinearOpType::DequantizeLinear;
  else if (op_type == "QuantizeLinear")
    return QLinearOpType::QuantizeLinear;
  else if (op_type == "QLinearConv")
    return QLinearOpType::QLinearConv;
  else if (op_type == "QLinearMatMul")
    return QLinearOpType::QLinearMatMul;
  else if (op_type == "QLinearAdd")
    return QLinearOpType::QLinearAdd;
  else if (op_type == "QLinearSigmoid")
    return QLinearOpType::QLinearSigmoid;
  else if (op_type == "QLinearAveragePool")
    return QLinearOpType::QLinearAveragePool;
  else if (op_type == "QLinearMul")
    return QLinearOpType::QLinearMul;
  else if (op_type == "QLinearReduceMean")
    return QLinearOpType::QLinearReduceMean;
  else if (op_type == "QLinearConcat")
    return QLinearOpType::QLinearConcat;
  else if (op_type == "QLinearGlobalAveragePool")
    return QLinearOpType::QLinearGlobalAveragePool;
  else if (op_type == "QLinearLeakyRelu")
    return QLinearOpType::QLinearLeakyRelu;

  return QLinearOpType::Unknown;
}

// Ops have 1 input
bool IsUnaryQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearSigmoid ||
         type == QLinearOpType::QLinearAveragePool ||
         type == QLinearOpType::QLinearGlobalAveragePool ||
         type == QLinearOpType::QLinearLeakyRelu ||
         type == QLinearOpType::QLinearReduceMean;
}

// Ops have 2 inputs
bool IsBinaryQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearConv ||
         type == QLinearOpType::QLinearMatMul ||
         type == QLinearOpType::QLinearAdd ||
         type == QLinearOpType::QLinearMul;
}

// Ops have 1 or more inputs
bool IsVariadicQLinearOp(QLinearOpType type) {
  return type == QLinearOpType::QLinearConcat;
}

Ort::Status GetQDQIODefs(const OrtNode& target_node, const NodeGroup& node_group, bool is_input,
                         /*out*/ std::vector<NodeUnitIODef>& result) {
  const std::vector<const OrtNode*>& dq_or_q_nodes = is_input ? node_group.dq_nodes : node_group.q_nodes;
  std::vector<const OrtValueInfo*> target_node_io_defs;
  EdgeSet target_node_io_edges;

  if (is_input) {
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(target_node, target_node_io_defs));
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputEdges(target_node_io_defs, target_node_io_edges));
  } else {
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(target_node, target_node_io_defs));
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(target_node_io_defs, target_node_io_edges));
  }

  // Find all the quantized IO defs and indices (for the input/output of the target node)
  std::unordered_map<size_t, NodeUnitIODef> quantized_io_defs;
  quantized_io_defs.reserve(target_node_io_defs.size());

  for (const EdgeEnd& target_node_edge : target_node_io_edges) {
    const OrtNode& node = target_node_edge.GetNode();  // node on the other side of the edge.

    // If we can find the node in the dq or q nodes this is a quantized input/output
    if (std::find(dq_or_q_nodes.cbegin(), dq_or_q_nodes.cend(), &node) != dq_or_q_nodes.cend()) {
      std::vector<const OrtValueInfo*> node_inputs;
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(node, node_inputs));

      std::optional<int64_t> axis;
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetAxisAttribute(node, axis));

      // quantization scale and zp are always the input[1, 2]
      NodeUnitIODef::QuantParam quant_param{*node_inputs[1], node_inputs.size() == 3 ? node_inputs[2] : nullptr, axis};

      if (is_input) {
        // DQ is input to the target node, use the DstArgIndex
        auto idx = target_node_edge.GetDstArgIndex();
        // This is a DQ node, we are using x, x_scale, x_zp (input[0, 1, 2])
        quantized_io_defs.insert({idx, NodeUnitIODef{*node_inputs[0], quant_param}});
      } else {
        // Q is output of the target node, use the SrcArgIndex
        auto idx = target_node_edge.GetSrcArgIndex();
        // This is a Q node, we are using y (output[0]), y_scale, y_zp (input[1, 2])
        std::vector<const OrtValueInfo*> node_outputs;
        ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(node, node_outputs));

        quantized_io_defs.insert({idx, NodeUnitIODef{*node_outputs[0], quant_param}});
      }
    }
  }

  // Construct the IODefs for this QDQ NodeGroup
  result.reserve(target_node_io_defs.size());
  for (size_t i = 0; i < target_node_io_defs.size(); i++) {
    // If we can find the NodeUnitIODef for this index, this is a quantized input/output
    if (quantized_io_defs.find(i) != quantized_io_defs.cend()) {
      result.push_back(std::move(quantized_io_defs.at(i)));
    } else {
      // This is a regular input
      result.push_back({*target_node_io_defs[i], std::nullopt});
    }
  }

  return Ort::Status{nullptr};
}

}  // namespace

EdgeEnd::EdgeEnd(const OrtNode& node, int src_arg_index, int dst_arg_index) noexcept
    : node_(&node), src_arg_index_(src_arg_index), dst_arg_index_(dst_arg_index) {}

// explicit EdgeEnd(const OrtNode& node) noexcept;
bool EdgeEndCompare::operator()(const EdgeEnd& lhs, const EdgeEnd& rhs) const {
  const OrtApi& ort_api = Ort::GetApi();
  const OrtNode& l_node = lhs.GetNode();
  const OrtNode& r_node = rhs.GetNode();

  size_t l_node_id = 0;
  size_t r_node_id = 0;
  Ort::Status l_status{ort_api.Node_GetId(&l_node, &l_node_id)};
  Ort::Status r_status{ort_api.Node_GetId(&r_node, &r_node_id)};

  assert(l_status.IsOK() && r_status.IsOK());

  if (l_node_id == r_node_id) {
    if (lhs.GetSrcArgIndex() == rhs.GetSrcArgIndex()) {
      return lhs.GetDstArgIndex() < rhs.GetDstArgIndex();
    }

    return lhs.GetSrcArgIndex() < rhs.GetSrcArgIndex();
  }

  return l_node_id < r_node_id;
}

NodeUnit::NodeUnit(const OrtNode& target_node, Type type, PrivateTag)
    : target_node_(target_node),
      type_(type) {}

/*static*/
Ort::Status NodeUnit::MakeSingleNode(const OrtNode& node, /*out*/ std::unique_ptr<NodeUnit>& result) {
  const OrtApi& ort_api = Ort::GetApi();

  std::vector<const OrtValueInfo*> node_inputs;
  std::vector<const OrtValueInfo*> node_outputs;
  const char* op_type = nullptr;
  size_t num_input_edges = 0;
  EdgeSet output_edges;

  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(node, node_inputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(node, node_outputs));
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(&node, &op_type));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputEdgeCount(node_inputs, num_input_edges));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(node_outputs, output_edges));

  std::vector<NodeUnitIODef> node_unit_inputs;
  std::vector<NodeUnitIODef> node_unit_outputs;

  auto qlinear_type = GetQLinearOpType(op_type);
  if (qlinear_type == QLinearOpType::Unknown) {
    // Not a Qlinear op, add all inputs / outputs.
    auto add_all_io = [](std::vector<NodeUnitIODef>& node_unit_defs, const std::vector<const OrtValueInfo*>& node_defs) {
      node_unit_defs.reserve(node_defs.size());

      for (const auto& node_def : node_defs) {
        node_unit_defs.push_back(NodeUnitIODef{*node_def, std::nullopt});
      }
    };

    add_all_io(node_unit_inputs, node_inputs);
    add_all_io(node_unit_outputs, node_outputs);
  } else if (IsUnaryQLinearOp(qlinear_type)) {
    // Unary QLinear Op has 5 inputs
    // x, x_scale, x_zp, y_scale, y_zp (optional)
    node_unit_inputs.push_back(NodeUnitIODef{*node_inputs[0], NodeUnitIODef::QuantParam{*node_inputs[1], node_inputs[2]}});
    node_unit_outputs.push_back(NodeUnitIODef{*node_outputs[0],
                                              NodeUnitIODef::QuantParam{*node_inputs[3],
                                                                        node_inputs.size() > 4 ? node_inputs[4] : nullptr}});
  } else if (IsBinaryQLinearOp(qlinear_type)) {
    // Binary QLinear Op has 9 inputs
    // x1, x1_scale, x1_zp, x2/w, x2_scale, x2_zp, y_scale , y_zp, B
    node_unit_inputs.push_back(NodeUnitIODef{*node_inputs[0], NodeUnitIODef::QuantParam{*node_inputs[1], node_inputs[2]}});
    node_unit_inputs.push_back(NodeUnitIODef{*node_inputs[3], NodeUnitIODef::QuantParam{*node_inputs[4], node_inputs[5]}});

    if (node_inputs.size() == 9) {                                               // has Bias
      node_unit_inputs.push_back(NodeUnitIODef{*node_inputs[8], std::nullopt});  // for Bias the scale and zp are optional
    }

    node_unit_outputs.push_back(NodeUnitIODef{*node_outputs[0], NodeUnitIODef::QuantParam{*node_inputs[6], node_inputs[7]}});
  } else if (qlinear_type == QLinearOpType::DequantizeLinear) {
    // DequantizeLinear has 3 inputs
    // x, x_scale, x_zp
    // output is not quantized

    // Get the DQ axis attribute if available.
    std::optional<int64_t> axis;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetAxisAttribute(node, axis));

    node_unit_inputs.push_back(NodeUnitIODef{*node_inputs[0],
                                             NodeUnitIODef::QuantParam{*node_inputs[1],
                                                                       node_inputs.size() == 3 ? node_inputs[2] : nullptr,
                                                                       axis}});
    node_unit_outputs.push_back(NodeUnitIODef{*node_outputs[0], std::nullopt});
  } else if (qlinear_type == QLinearOpType::QuantizeLinear) {
    // QuantizeLinear the input is not quantized and has 3 inputs
    // x, y_scale, y_zp (optional)
    // The output is quantized

    // Get the Q axis attribute if available.
    std::optional<int64_t> axis;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetAxisAttribute(node, axis));

    node_unit_inputs.push_back(NodeUnitIODef{*node_inputs[0], std::nullopt});
    node_unit_outputs.push_back(NodeUnitIODef{*node_outputs[0],
                                              NodeUnitIODef::QuantParam{*node_inputs[1],
                                                                        node_inputs.size() == 3 ? node_inputs[2] : nullptr,
                                                                        axis}});
  } else if (IsVariadicQLinearOp(qlinear_type)) {
    size_t input_num = (node_inputs.size() - 2) / 3;
    for (size_t i = 0; i < input_num; i++) {
      node_unit_inputs.push_back(NodeUnitIODef{*node_inputs[3 * i + 2], NodeUnitIODef::QuantParam{*node_inputs[3 * i + 3],
                                                                                                  node_inputs[3 * i + 4]}});
    }
    node_unit_outputs.push_back(NodeUnitIODef{*node_outputs[0], NodeUnitIODef::QuantParam{*node_inputs[0], node_inputs[1]}});
  } else {
    std::ostringstream oss;
    oss << "QLinear op [" << static_cast<uint8_t>(qlinear_type) << "] is not supported";
    return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
  }

  auto node_unit = std::make_unique<NodeUnit>(node, Type::SingleNode, PrivateTag{});
  node_unit->redundant_clip_node_ = nullptr;
  node_unit->input_edge_count_ = num_input_edges;
  node_unit->inputs_ = std::move(node_unit_inputs);
  node_unit->outputs_ = std::move(node_unit_outputs);
  node_unit->output_edges_ = std::move(output_edges);

  result = std::move(node_unit);
  return Ort::Status{nullptr};
}

/*static*/
Ort::Status NodeUnit::MakeQDQGroup(const OrtGraph& graph, const NodeGroup& node_group,
                                   /*out*/ std::unique_ptr<NodeUnit>& result) {
#if 0
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(NodeGroup::CanCreateNodeGroup(graph, *node_group.target_node,
                                                                 node_group.redundant_clip_node,
                                                                 node_group.dq_nodes, node_group.q_nodes));
#else
  (void)graph;
#endif

  std::vector<NodeUnitIODef> node_unit_inputs;
  std::vector<NodeUnitIODef> node_unit_outputs;
  const OrtNode* output_producer = node_group.redundant_clip_node ? node_group.redundant_clip_node
                                                                  : node_group.target_node;

  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetQDQIODefs(*node_group.target_node, node_group, true, node_unit_inputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetQDQIODefs(*output_producer, node_group, false, node_unit_outputs));

  size_t input_edge_count = 0;
  for (const OrtNode* dq_node : node_group.dq_nodes) {
    size_t dq_num_input_edges = 0;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputEdgeCount(*dq_node, dq_num_input_edges));

    input_edge_count += dq_num_input_edges;
  }

  // add edges for inputs that are not from DQ nodes. there is one edge to each DQ node.
  // other inputs could come from initializers or graph inputs (no edges) or other nodes (edge).
  size_t target_node_num_input_edges = 0;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputEdgeCount(*node_group.target_node, target_node_num_input_edges));

  input_edge_count += target_node_num_input_edges - node_group.dq_nodes.size();

  // create output edges. each target node output either goes to Q node/s or non-Q node/s.
  // CanCreateNodeGroup ensures this.
  // If redundant clip node is present, the target node has only one output edge to the redundant clip node.
  EdgeSet node_unit_output_edges;
  EdgeSet output_edges;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(*output_producer, output_edges));

  for (const EdgeEnd& cur_edge : output_edges) {
    const OrtNode& node = cur_edge.GetNode();  // node on other side of edge.

    // if node is in q_nodes we look past the Q node and add those edges.
    if (std::find(node_group.q_nodes.cbegin(), node_group.q_nodes.cend(), &node) != node_group.q_nodes.cend()) {
      int src_idx = cur_edge.GetSrcArgIndex();
      EdgeSet q_out_edges;
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(node, q_out_edges));

      for (const EdgeEnd& q_cur_edge : q_out_edges) {
        node_unit_output_edges.insert(EdgeEnd{q_cur_edge.GetNode(), src_idx, q_cur_edge.GetDstArgIndex()});
      }
    } else {
      // non-Q node, or Q node that isn't in the QDQ node group (unexpected but may be possible). add as-is.
      node_unit_output_edges.insert(cur_edge);
    }
  }

  auto node_unit = std::make_unique<NodeUnit>(*node_group.target_node, Type::QDQGroup, PrivateTag{});
  node_unit->dq_nodes_ = node_group.dq_nodes;
  node_unit->redundant_clip_node_ = node_group.redundant_clip_node;
  node_unit->q_nodes_ = node_group.q_nodes;
  node_unit->inputs_ = std::move(node_unit_inputs);
  node_unit->outputs_ = std::move(node_unit_outputs);
  node_unit->output_edges_ = std::move(node_unit_output_edges);

  result = std::move(node_unit);
  return Ort::Status{nullptr};
}

Ort::Status NodeGroup::CanCreateNodeGroup(const OrtGraph& graph,
                                          const OrtNode& target_node,
                                          const OrtNode* redundant_clip_node,
                                          const std::vector<const OrtNode*>& dq_nodes,
                                          const std::vector<const OrtNode*>& q_nodes) {
  const OrtApi& ort_api = Ort::GetApi();

  const char* target_node_name = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetName(&target_node, &target_node_name));

  // Within a QDQ node group, a target node input is the only consumer of each DQ.
  // This should have been ensured by the EnsureUniqueDQForNodeUnit graph transformer, but other graph modifications
  // may have happened since. Verify that this is still true.
  for (const OrtNode* dq_node : dq_nodes) {
    const char* dq_node_name = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetName(dq_node, &dq_node_name));

    std::vector<const OrtValueInfo*> dq_outputs;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(*dq_node, dq_outputs));

    bool dq_produces_graph_output = false;
    for (const OrtValueInfo* dq_output : dq_outputs) {
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsGraphOutput(dq_output, &dq_produces_graph_output));
      if (dq_produces_graph_output) {
        std::ostringstream oss;
        oss << "QDQ node group cannot have DQ node that produces a graph output. DQ node: " << dq_node_name
            << ", target node: " << target_node_name;
        return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
      }
    }

    EdgeSet dq_output_edges;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(dq_outputs, dq_output_edges));
    bool dq_has_single_output_edge_to_target = dq_output_edges.size() == 1 &&
                                               &dq_output_edges.begin()->GetNode() == &target_node;
    if (!dq_has_single_output_edge_to_target) {
      std::ostringstream oss;
      oss << "QDQ node group cannot have a DQ that doesn't have a single output edge to the target node. "
          << "DQ node: " << dq_node_name << ", target node: " << target_node_name;
      return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
    }
  }

  // TODO
  (void)graph;
  (void)redundant_clip_node;
  (void)q_nodes;

  return Ort::Status{nullptr};
}

}  // namespace QDQ
}  // namespace OrtEpUtils

#endif  // ORT_EP_UTILS_QDQ_UTILS_IMPL
