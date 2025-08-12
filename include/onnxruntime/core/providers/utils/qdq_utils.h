// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// DO NOT include ORT header files as this is meant to be a header-only utility that can be copied
// to other projects.

#ifndef INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_QDQ_UTILS_H_
#define INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_QDQ_UTILS_H_

#include <gsl/gsl>
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
                                        gsl::span<const OrtNode* const> dq_nodes,
                                        gsl::span<const OrtNode* const> q_nodes);
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

#include <cassert>
#include <sstream>

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

// Get the number of input edges that come from another node upstream.
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
    auto add_all_io = [](std::vector<NodeUnitIODef>& node_unit_defs, gsl::span<const OrtValueInfo* const> node_defs) {
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

}  // namespace QDQ
}  // namespace OrtEpUtils

#endif  // ORT_EP_UTILS_QDQ_UTILS_IMPL
