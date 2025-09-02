// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// DO NOT include ORT header files as this is meant to be a header-only utility that can be copied
// to other projects.

/*
 SUMMARY:
   Example utilities to group OrtGraph nodes into NodeUnits. Only supports QDQ Conv for now.

   !! Users may copy this file and modify as needed. !!

 USAGE:
   This is a header-only implementation that includes both the function declarations and definitions. Copy this file
   into a project that links with both ONNX Runtime and ONNX.

   Define the ORT_EP_UTILS_QDQ_UTILS_IMPL preprocessor macro before the #include statement in exactly one C++
   file to define the implementation. Example:

     #define ORT_EP_UTILS_QDQ_UTILS_IMPL
     #include "qdq_utils.h"

   Other compilation units that depend on these utilities should include this file without defining the
   preprocessor macro.

   Example program snippets are shown below. Refer to the function declarations for detailed usage information.

 EXAMPLE SNIPPET:

   ```C++
   #define ORT_EP_UTILS_QDQ_UTILS_IMPL
   #include "qdq_utils.h"

   OrtStatus* ORT_API_CALL GetCapabilityImpl(OrtEp* this_ptr, const OrtGraph* ort_graph,
                                             OrtEpGraphSupportInfo* graph_support_info) {

     std::vector<std::unique_ptr<OrtEpUtils::QDQ::NodeUnit>> node_units;
     std::unordered_map<const OrtNode*, const OrtEpUtils::QDQ::NodeUnit*> node_unit_map;

     // Group all nodes into NodeUnit instances.
     Ort::Status status = OrtEpUtils::QDQ::GetAllNodeUnits(graph, logger, node_units, node_unit_map);

     // ...
   }
   ```

 HOW TO ADD SUPPORT FOR NEW OPERATOR:
   This utility only supports QDQ Conv. To add a new operator type:
     1. Create a new class the derives from `NodeGroupSelector`. Ex: struct ConvNodeGroupSelector: NodeGroupSelector.
     2. Implement the virtual NodeGroupSelector::Check() function, which should check for allowed structure
        and data types.
     3. Update SelectorManager::CreateSelectors() to register your new `NodeGroupSelector` derived class.

   The code in this file was ported from the following internal ONNX Runtime files. Please use them as a reference:
     - onnxruntime/core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h (and qdq_selectors.cc)
       - Contains NodeGroupSelector classes and utilities.
     - onnxruntime/core/optimizer/qdq_transformer/selectors_actions/shared/utils.h (and utils.cc)
       - Contains SelectorManger classes and implementation of GetAllNodeUnits().
     - onnxruntime/core/framework/node_unit.h (and node_unit.cc)
       - Contains NodeUnit class and utilities.

*/

#ifndef INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_QDQ_UTILS_H_
#define INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_QDQ_UTILS_H_

#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
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
  static Ort::Status CanCreateNodeGroup(const OrtNode& target_node,
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
  static Ort::Status MakeQDQGroup(const NodeGroup& node_group, /*out*/ std::unique_ptr<NodeUnit>& node_unit);

  Type UnitType() const noexcept { return type_; }

  const std::vector<NodeUnitIODef>& Inputs() const noexcept { return inputs_; }
  const std::vector<NodeUnitIODef>& Outputs() const noexcept { return outputs_; }
  const OrtNode& GetNode() const noexcept { return target_node_; }
  const OrtNode* GetRedundantClipNode() const noexcept { return redundant_clip_node_; }
  const std::vector<const OrtNode*>& GetDQNodes() const noexcept { return dq_nodes_; }
  const std::vector<const OrtNode*>& GetQNodes() const noexcept { return q_nodes_; }
  std::vector<const OrtNode*> GetAllNodesInGroup() const noexcept;

  size_t InputEdgeCount() const noexcept { return input_edge_count_; }
  EdgeConstIterator OutputEdgesBegin() const;
  EdgeConstIterator OutputEdgesEnd() const;

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

Ort::Status GetAllNodeUnits(const OrtGraph& graph, const Ort::Logger& logger,
                            /*out*/ std::vector<std::unique_ptr<NodeUnit>>& node_units,
                            /*out*/ std::unordered_map<const OrtNode*, const NodeUnit*>& node_to_node_unit);

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
#include <cstring>
#include <sstream>
#include <string_view>
#include <unordered_set>
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
constexpr const char* DQ_OP_TYPE = "DequantizeLinear";
constexpr const char* Q_OP_TYPE = "QuantizeLinear";

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

Ort::Status FindParentsByType(const std::vector<const OrtValueInfo*>& inputs, std::string_view parent_type,
                              std::vector<const OrtNode*>& result) {
  const OrtApi& ort_api = Ort::GetApi();

  std::vector<const OrtNode*> parents;
  parents.reserve(inputs.size());

  for (const OrtValueInfo* input : inputs) {
    if (input == nullptr) continue;  // Skip missing optional input

    const OrtNode* producer_node = nullptr;
    size_t producer_output_index = 0;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueProducer(input, &producer_node, &producer_output_index));

    if (producer_node == nullptr) {
      continue;  // No producer node (e.g., a graph input)
    }

    const char* producer_op_type = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(producer_node, &producer_op_type));

    if (parent_type == producer_op_type) {
      parents.push_back(producer_node);
    }
  }

  result = std::move(parents);
  return Ort::Status{nullptr};
}

Ort::Status GetNodeOutputEdgeCount(const std::vector<const OrtValueInfo*>& outputs, size_t& num_output_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  num_output_edges = 0;
  for (const OrtValueInfo* output : outputs) {
    size_t num_consumers = 0;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueNumConsumers(output, &num_consumers));

    std::vector<const OrtNode*> consumer_nodes(num_consumers);
    std::vector<int64_t> consumer_indices(num_consumers);

    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueConsumers(output, consumer_nodes.data(),
                                                                       consumer_indices.data(), num_consumers));

    // Could all edges
    for (int64_t index : consumer_indices) {
      if (index >= 0) {  // Do not count implicit inputs to a consumer node.
        num_output_edges += 1;
      }
    }
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

Ort::Status GetNodeOutputEdges(const OrtNode& node, EdgeSet& output_edges) {
  std::vector<const OrtValueInfo*> outputs;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(node, outputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(outputs, output_edges));
  return Ort::Status{nullptr};
}

Ort::Status GetChildrenByType(const std::vector<const OrtValueInfo*>& outputs, std::string_view child_type,
                              std::vector<const OrtNode*>& result) {
  const OrtApi& ort_api = Ort::GetApi();

  std::vector<const OrtNode*> children;
  children.reserve(outputs.size());  // Can be larger than `outputs.size()` if an output is consumed by multiple nodes

  for (const OrtValueInfo* output : outputs) {
    size_t num_consumers = 0;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueNumConsumers(output, &num_consumers));

    std::vector<const OrtNode*> consumer_nodes(num_consumers);
    std::vector<int64_t> consumer_indices(num_consumers);

    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueConsumers(output, consumer_nodes.data(),
                                                                       consumer_indices.data(), num_consumers));

    // Build up an edge with consumer nodes as the destinations.
    for (size_t c_idx = 0; c_idx < num_consumers; c_idx++) {
      const OrtNode* consumer_node = consumer_nodes[c_idx];
      int64_t consumer_input_index = consumer_indices[c_idx];

      if (consumer_input_index < 0) {
        continue;  // Skip implicit input to consumer node.
      }

      const char* consumer_op_type = nullptr;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(consumer_node, &consumer_op_type));

      if (child_type == consumer_op_type) {
        children.push_back(consumer_node);
      }
    }
  }

  result = std::move(children);
  return Ort::Status{nullptr};
}

Ort::Status CountActualNodeIOValues(const std::vector<const OrtValueInfo*>& io_values, int& num_actual_values) {
  num_actual_values = 0;
  for (const OrtValueInfo* value : io_values) {
    if (value != nullptr) {
      num_actual_values += 1;
    }
  }

  return Ort::Status{nullptr};
}

Ort::Status ProducesGraphOutput(const std::vector<const OrtValueInfo*>& outputs, bool& any_is_graph_output) {
  const OrtApi& ort_api = Ort::GetApi();

  any_is_graph_output = false;
  for (const OrtValueInfo* output : outputs) {
    bool is_graph_output = false;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsGraphOutput(output, &is_graph_output));
    if (is_graph_output) {
      any_is_graph_output = true;
      break;
    }
  }

  return Ort::Status{nullptr};
}

Ort::Status GetTensorElemType(const OrtValueInfo& ort_value_info, /*out*/ ONNXTensorElementDataType& elem_type) {
  const OrtApi& ort_api = Ort::GetApi();

  elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

  const OrtTypeInfo* ort_type_info = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetValueInfoTypeInfo(&ort_value_info, &ort_type_info));

  ONNXType ort_onnx_type = ONNX_TYPE_UNKNOWN;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetOnnxTypeFromTypeInfo(ort_type_info, &ort_onnx_type));
  ORT_EP_UTILS_C_RETURN_IF(ort_onnx_type != ONNX_TYPE_TENSOR, ort_api, "Expected OrtValueInfo to represent a Tensor");

  const OrtTensorTypeAndShapeInfo* ort_type_shape = nullptr;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.CastTypeInfoToTensorInfo(ort_type_info, &ort_type_shape));
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.GetTensorElementType(ort_type_shape, &elem_type));

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

enum class CheckConsumerResult {
  kOK,
  kErrorProducesGraphOutput,
  kErrorWrongConsumer,
};

Ort::Status CheckOnlyOneNodeConsumer(const OrtNode& node, const OrtNode& expected_consumer,
                                     CheckConsumerResult& result) {
  const OrtApi& ort_api = Ort::GetApi();

  std::vector<const OrtValueInfo*> node_outputs;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(node, node_outputs));

  // Node should only have one output.
  if (node_outputs.size() != 1 || node_outputs[0] == nullptr) {
    result = CheckConsumerResult::kErrorWrongConsumer;
    return Ort::Status{nullptr};
  }

  // Check that node does not produce a graph output.
  bool is_graph_output = false;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsGraphOutput(node_outputs[0], &is_graph_output));

  if (is_graph_output) {
    result = CheckConsumerResult::kErrorProducesGraphOutput;
    return Ort::Status{nullptr};
  }

  // Check for one consumer.
  EdgeSet output_edges;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(node_outputs, output_edges));

  if (output_edges.size() != 1 || &output_edges.begin()->GetNode() != &expected_consumer) {
    result = CheckConsumerResult::kErrorWrongConsumer;
    return Ort::Status{nullptr};
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
Ort::Status NodeUnit::MakeQDQGroup(const NodeGroup& node_group, /*out*/ std::unique_ptr<NodeUnit>& result) {
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(NodeGroup::CanCreateNodeGroup(*node_group.target_node,
                                                                 node_group.redundant_clip_node,
                                                                 node_group.dq_nodes, node_group.q_nodes));

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
  node_unit->input_edge_count_ = input_edge_count;
  node_unit->output_edges_ = std::move(node_unit_output_edges);

  result = std::move(node_unit);
  return Ort::Status{nullptr};
}

std::vector<const OrtNode*> NodeUnit::GetAllNodesInGroup() const noexcept {
  std::vector<const OrtNode*> all_nodes = dq_nodes_;
  all_nodes.push_back(&target_node_);

  if (redundant_clip_node_) {
    all_nodes.push_back(redundant_clip_node_);
  }

  all_nodes.reserve(all_nodes.size() + q_nodes_.size());

  for (auto& n : q_nodes_) {
    all_nodes.push_back(n);
  }

  return all_nodes;
}

EdgeConstIterator NodeUnit::OutputEdgesBegin() const {
  return output_edges_.begin();
}

EdgeConstIterator NodeUnit::OutputEdgesEnd() const {
  return output_edges_.end();
}

Ort::Status NodeGroup::CanCreateNodeGroup(const OrtNode& target_node,
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

    CheckConsumerResult check_consumer_result = CheckConsumerResult::kOK;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(CheckOnlyOneNodeConsumer(*dq_node, target_node, check_consumer_result));

    if (check_consumer_result == CheckConsumerResult::kErrorProducesGraphOutput) {
      std::ostringstream oss;
      oss << "QDQ node group cannot have DQ node that produces a graph output. DQ node: " << dq_node_name
          << ", target node: " << target_node_name;
      return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
    }

    if (check_consumer_result == CheckConsumerResult::kErrorWrongConsumer) {
      std::ostringstream oss;
      oss << "QDQ node group cannot have a DQ that doesn't have a single output edge to the target node. "
          << "DQ node: " << dq_node_name << ", target node: " << target_node_name;
      return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
    }
  }

  if (redundant_clip_node != nullptr) {
    const char* clip_node_name = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetName(redundant_clip_node, &clip_node_name));

    CheckConsumerResult target_consumer_check = CheckConsumerResult::kOK;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(CheckOnlyOneNodeConsumer(target_node, *redundant_clip_node,
                                                              target_consumer_check));

    if (target_consumer_check != CheckConsumerResult::kOK) {
      std::ostringstream oss;
      oss << "A redundant Clip/Relu node in a QDQ node group must be the only consumer of the target node. "
          << "target node: " << target_node_name << ", redundant Clip/Relu node: " << clip_node_name;
      return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
    }

    if (q_nodes.size() != 1) {
      return Ort::Status(
          "Currently only support QDQ node groups with a redundant Clip/Relu node if there is only one Q",
          OrtErrorCode::ORT_FAIL);
    }

    CheckConsumerResult clip_consumer_check = CheckConsumerResult::kOK;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(CheckOnlyOneNodeConsumer(*redundant_clip_node, *q_nodes[0],
                                                              clip_consumer_check));
    if (clip_consumer_check != CheckConsumerResult::kOK) {
      std::ostringstream oss;
      oss << "A redundant Clip/Relu node in a QDQ node group must have a single output edge to a Q node. "
          << "target node: " << target_node_name << ", redundant Clip/Relu node: " << clip_node_name;
      return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
    }
  }

  // an output from the target node can have either Q consumers or direct consumers. it cannot have both.
  // this must be checked on a per output basis.
  // e.g. TopK produces values and indices. The indices output won't be quantized, so even if we replace the TopK QDQ
  // node group with a quantized TopK, an int64_t indices value will be produced and can provide a graph output.
  if (!q_nodes.empty()) {
    const OrtNode& node_producing_outputs = redundant_clip_node ? *redundant_clip_node : target_node;
    const char* node_produce_outputs_name = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetName(&node_producing_outputs, &node_produce_outputs_name));

    std::vector<const OrtValueInfo*> outputs;
    EdgeSet output_edges;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(node_producing_outputs, outputs));
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(outputs, output_edges));

    std::vector<const OrtNode*> output_to_first_consumer(outputs.size(), nullptr);

    for (const EdgeEnd& cur_edge : output_edges) {
      int output_idx = cur_edge.GetSrcArgIndex();
      const OrtNode& this_consumer = cur_edge.GetNode();
      const OrtNode* existing_consumer = output_to_first_consumer[output_idx];

      if (existing_consumer != nullptr) {
        // another edge for this output. either both are Q or both are not.
        // note: a single output could lead to multiple Q nodes if the output is variadic.
        const char* this_consumer_op_type = nullptr;
        const char* existing_consumer_op_type = nullptr;
        ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(&this_consumer, &this_consumer_op_type));
        ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(existing_consumer, &existing_consumer_op_type));

        bool valid = true;
        if (std::strcmp(existing_consumer_op_type, Q_OP_TYPE) == 0) {
          valid = std::strcmp(this_consumer_op_type, Q_OP_TYPE) == 0;
        } else {
          valid = std::strcmp(this_consumer_op_type, Q_OP_TYPE) != 0;
        }

        if (!valid) {
          std::ostringstream oss;
          oss << "QDQ node group cannot have an output from the target (or clip) node that is consumed by both "
              << "a Q node and a non-Q node. target (or clip) node: " << node_produce_outputs_name;
          return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
        }
      } else {
        output_to_first_consumer[output_idx] = &this_consumer;
      }
    }

    // Check that any output with a Q cannot be a graph output as it will disappear if the QDQ node unit is converted to
    // a quantized op.
    for (size_t idx = 0, end = output_to_first_consumer.size(); idx < end; ++idx) {
      if (output_to_first_consumer[idx] == nullptr) {
        continue;
      }

      const char* op_type = nullptr;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(output_to_first_consumer[idx], &op_type));

      if (std::strcmp(op_type, Q_OP_TYPE) != 0) {
        continue;
      }

      bool is_graph_output = false;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_IsGraphOutput(outputs[idx], &is_graph_output));

      if (is_graph_output) {
        std::ostringstream oss;
        oss << "QDQ node group cannot have an output from the target (or clip/relu) node that is consumed by a Q "
            << "node and a graph output. target (or clip/relu) node: " << node_produce_outputs_name << ", output idx: "
            << idx;
        return Ort::Status(oss.str().c_str(), OrtErrorCode::ORT_FAIL);
      }
    }
  }

  return Ort::Status{nullptr};
}

//
// QDQ Operator selection logic (ported from core/optimizers/qdq_transformer/selectors_actions/qdq_selectors.cc
//

namespace {
class NodeGroupSelector {
 public:
  Ort::Status GetQDQSelection(const OrtNode& node, std::optional<NodeGroup>& result) const;
  virtual ~NodeGroupSelector() = default;

 protected:
  // base check that we have the expected number of QDQ inputs/outputs, and `node` isn't producing a graph output.
  // num_dq_inputs defaults to the number of inputs `node` has if not explicitly specified
  Ort::Status CheckQDQNodes(/*out*/ bool& is_valid, const OrtNode& node, const OrtNode* redundant_clip_node,
                            const std::vector<const OrtNode*>& dq_nodes,
                            const std::vector<const OrtNode*>& q_nodes,
                            int num_dq_inputs = -1,
                            bool is_empty_q_nodes_allowed = false) const;

 private:
  // derived classes should implement this check
  virtual Ort::Status Check(const OrtNode& node, const OrtNode* redundant_clip_node,
                            const std::vector<const OrtNode*>& dq_nodes,
                            const std::vector<const OrtNode*>& q_nodes,
                            /*out*/ bool& is_valid) const = 0;
};

Ort::Status NodeGroupSelector::CheckQDQNodes(/*out*/ bool& is_valid, const OrtNode& node,
                                             const OrtNode* redundant_clip_node,
                                             const std::vector<const OrtNode*>& dq_nodes,
                                             const std::vector<const OrtNode*>& q_nodes,
                                             int num_dq_inputs, bool is_empty_q_nodes_allowed) const {
  is_valid = false;

  if (num_dq_inputs == -1) {
    std::vector<const OrtValueInfo*> inputs;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(node, inputs));
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(CountActualNodeIOValues(inputs, num_dq_inputs));
  }

  if (num_dq_inputs != static_cast<int>(dq_nodes.size())) {
    return Ort::Status{nullptr};
  }

  if (const auto qdq_validation_status = NodeGroup::CanCreateNodeGroup(node, redundant_clip_node, dq_nodes, q_nodes);
      !qdq_validation_status.IsOK()) {
    return Ort::Status{nullptr};
  }

  if (q_nodes.empty()) {
    is_valid = is_empty_q_nodes_allowed;
    return Ort::Status{nullptr};
  }

  std::vector<const OrtValueInfo*> outputs;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(node, outputs));

  bool produces_graph_output = false;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(ProducesGraphOutput(outputs, produces_graph_output));
  if (produces_graph_output) {
    return Ort::Status{nullptr};
  }

  int num_actual_outputs = 0;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(CountActualNodeIOValues(outputs, num_actual_outputs));
  if (num_actual_outputs != static_cast<int>(q_nodes.size())) {
    return Ort::Status{nullptr};
  }

  size_t num_output_edges = 0;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdgeCount(outputs, num_output_edges));
  if (num_output_edges != q_nodes.size()) {
    return Ort::Status{nullptr};
  }

  is_valid = true;
  return Ort::Status{nullptr};
}

Ort::Status IsClipMadeRedundantByQ(const OrtNode& clip_node, const OrtNode& q_node, bool& is_clip_redundant) {
  (void)clip_node;
  (void)q_node;
  is_clip_redundant = true;  // TODO: implement
  return Ort::Status{nullptr};
}

constexpr bool Is16BitIntType(ONNXTensorElementDataType data_type) {
  return (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16) ||
         (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16);
}

constexpr bool Is4BitIntType(ONNXTensorElementDataType data_type) {
  return (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4) ||
         (data_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4);
}

Ort::Status NodeGroupSelector::GetQDQSelection(const OrtNode& node, std::optional<NodeGroup>& result) const {
  const OrtApi& ort_api = Ort::GetApi();
  result = std::nullopt;

  std::vector<const OrtValueInfo*> node_inputs;
  std::vector<const OrtValueInfo*> node_outputs;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(node, node_inputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(node, node_outputs));

  std::vector<const OrtNode*> dq_nodes;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(FindParentsByType(node_inputs, DQ_OP_TYPE, dq_nodes));

  // For redundant clip node, currently only support node with only one output, which is consumed by Clip/Relu->Q.
  const OrtNode* clip_node = nullptr;
  EdgeSet output_edges;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdges(node_outputs, output_edges));

  if (output_edges.size() == 1) {
    const OrtNode& next_node = output_edges.begin()->GetNode();

    const char* next_op_type = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(&next_node, &next_op_type));

    if (std::strcmp(next_op_type, "Clip") == 0 || std::strcmp(next_op_type, "Relu") == 0) {
      std::vector<const OrtValueInfo*> clip_outputs;
      size_t clip_num_output_edges = 0;
      bool produces_graph_output = false;
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(next_node, clip_outputs));
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputEdgeCount(clip_outputs, clip_num_output_edges));
      ORT_EP_UTILS_CXX_RETURN_IF_ERROR(ProducesGraphOutput(clip_outputs, produces_graph_output));

      if (clip_num_output_edges == 1 && !produces_graph_output) {
        clip_node = &next_node;
      }
    }
  }

  std::vector<const OrtNode*> q_nodes;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetChildrenByType(node_outputs, Q_OP_TYPE, q_nodes));

  if (clip_node != nullptr) {
    bool is_clip_redundant = false;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(IsClipMadeRedundantByQ(*clip_node, *q_nodes[0], is_clip_redundant));

    if (q_nodes.size() != 1 || !is_clip_redundant) {
      return Ort::Status{nullptr};
    }
  }

  bool check_result = false;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(Check(node, clip_node, dq_nodes, q_nodes, check_result));
  if (!check_result) {
    return Ort::Status{nullptr};
  }

  NodeGroup node_group;
  node_group.dq_nodes = dq_nodes;
  node_group.q_nodes = q_nodes;
  node_group.target_node = &node;
  node_group.redundant_clip_node = clip_node;

  result = std::move(node_group);
  return Ort::Status{nullptr};
}

// DQ nodes for X, W and optionally B -> node -> Q
class ConvNodeGroupSelector : public NodeGroupSelector {
 public:
  // default to 'true'
  ConvNodeGroupSelector(bool int8_allowed = true, bool allow_16bit = true, bool allow_4bit_weight = true)
      : int8_allowed_(int8_allowed), allow_16bit_(allow_16bit), allow_4bit_weight_(allow_4bit_weight) {}

 private:
  Ort::Status Check(const OrtNode& node, const OrtNode* redundant_clip_node,
                    const std::vector<const OrtNode*>& dq_nodes,
                    const std::vector<const OrtNode*>& q_nodes,
                    /*out*/ bool& is_valid) const override;

  bool int8_allowed_;
  bool allow_16bit_;
  bool allow_4bit_weight_;
};

Ort::Status ConvNodeGroupSelector::Check(const OrtNode& node, const OrtNode* redundant_clip_node,
                                         const std::vector<const OrtNode*>& dq_nodes,
                                         const std::vector<const OrtNode*>& q_nodes,
                                         /*out*/ bool& result) const {
  result = false;

  bool check_result = false;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(CheckQDQNodes(check_result, node, redundant_clip_node, dq_nodes, q_nodes));
  if (!check_result) {
    return Ort::Status{nullptr};
  }

  std::vector<const OrtValueInfo*> dq_0_inputs;
  std::vector<const OrtValueInfo*> dq_1_inputs;
  std::vector<const OrtValueInfo*> q_0_outputs;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(*dq_nodes[0], dq_0_inputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(*dq_nodes[1], dq_1_inputs));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeOutputs(*q_nodes[0], q_0_outputs));

  // input and output types need to be same
  ONNXTensorElementDataType dt_input = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType dt_weight = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType dt_output = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetTensorElemType(*dq_0_inputs[0], dt_input));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetTensorElemType(*dq_1_inputs[0], dt_weight));
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetTensorElemType(*q_0_outputs[0], dt_output));

  if (dt_input != dt_output) {
    return Ort::Status{nullptr};
  }

  if (!allow_4bit_weight_ && Is4BitIntType(dt_weight)) {
    return Ort::Status{nullptr};
  }

  if (dt_input == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) {
    if (!int8_allowed_ || dt_weight != dt_input) {
      return Ort::Status{nullptr};
    }
  }

  if (dq_nodes.size() == 3) {  // has bias
    std::vector<const OrtValueInfo*> dq_2_inputs;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetNodeInputs(*dq_nodes[2], dq_2_inputs));

    ONNXTensorElementDataType dt_bias = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(GetTensorElemType(*dq_2_inputs[0], dt_bias));

    if (dt_bias != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      return Ort::Status{nullptr};
    }
  }

  // 16-bit int types must be explicitly allowed.
  if (!allow_16bit_ && (Is16BitIntType(dt_input) || Is16BitIntType(dt_weight))) {
    return Ort::Status{nullptr};
  }

  result = true;
  return Ort::Status{nullptr};
}

// struct that provides a join between selector and op versions supported
struct OpVersionsAndSelector {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<int>>;

  OpVersionsAndSelector(const OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<NodeGroupSelector> selector_in)
      : op_versions_map{ops_and_versions_in},
        selector{std::move(selector_in)} {}

  OpVersionsMap op_versions_map;
  std::unique_ptr<NodeGroupSelector> selector;

  // ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpVersionsAndSelector);
};

// class that manages a set of node group selectors
class Selectors {
 public:
  Selectors() = default;

  // register a selector for the specified ops.
  void RegisterSelector(const OpVersionsAndSelector::OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<NodeGroupSelector> selector_in);

  const std::unordered_set<std::unique_ptr<OpVersionsAndSelector>>& SelectorsSet() const {
    return selectors_set_;
  }

  // ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Selectors);

 private:
  std::unordered_set<std::unique_ptr<OpVersionsAndSelector>> selectors_set_;
};

// class that manages qdq node group selections
class SelectorManager {
 public:
  SelectorManager();

  // Methods that finds and returns a vector of QDQ::NodeGroup in a given set of graph nodes.
  // Can be used in QDQ support in different EPs
  Ort::Status GetQDQSelections(const std::vector<const OrtNode*>& nodes, const Ort::Logger& logger,
                               /*out*/ std::vector<NodeGroup>& node_groups) const;

 private:
  Selectors qdq_selectors_;

  std::unordered_map<std::string, const OpVersionsAndSelector*> op_type_to_selectors_map_;

  void InitializeSelectorsMap();

  void CreateSelectors();

  // ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SelectorManager);
};

void Selectors::RegisterSelector(const OpVersionsAndSelector::OpVersionsMap& ops_and_versions_in,
                                 std::unique_ptr<NodeGroupSelector> selector_in) {
  auto entry = std::make_unique<OpVersionsAndSelector>(
      ops_and_versions_in,
      std::move(selector_in));

  (void)selectors_set_.insert(std::move(entry));
}

void RegisterConvSelector(Selectors& qdq_selectors) {
  static OpVersionsAndSelector::OpVersionsMap conv_version_map = {{"Conv", {}}};
  /* register selector for conv op */
  std::unique_ptr<NodeGroupSelector> selector = std::make_unique<ConvNodeGroupSelector>();
  qdq_selectors.RegisterSelector(conv_version_map, std::move(selector));
}

void SelectorManager::CreateSelectors() {
  RegisterConvSelector(qdq_selectors_);
}

void SelectorManager::InitializeSelectorsMap() {
  for (const auto& entry : qdq_selectors_.SelectorsSet()) {
    for (const auto& op_info : entry->op_versions_map) {
      (void)op_type_to_selectors_map_.insert({op_info.first, &*entry}).second;
    }
  }
}

SelectorManager::SelectorManager() {
  CreateSelectors();
  InitializeSelectorsMap();
}

constexpr const char* ONNX_DOMAIN = "";
constexpr const char* MS_INTERNAL_NHWC_DOMAIN = "com.ms.internal.nhwc";
constexpr const char* MS_DOMAIN = "com.microsoft";

Ort::Status SelectorManager::GetQDQSelections(const std::vector<const OrtNode*>& nodes, const Ort::Logger& logger,
                                              /*out*/ std::vector<NodeGroup>& result) const {
  const OrtApi& ort_api = Ort::GetApi();

  std::vector<NodeGroup> qdq_selections;
  for (const OrtNode* node : nodes) {
    const char* domain = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetDomain(node, &domain));

    bool is_onnx_domain = std::strcmp(domain, ONNX_DOMAIN) == 0;
    bool is_nhwc_domain = std::strcmp(domain, MS_INTERNAL_NHWC_DOMAIN) == 0;
    bool is_ms_domain = std::strcmp(domain, MS_DOMAIN) == 0;

    // post layout transformation all the layout sensitive nodes are converted to domain
    // kMSInternalNHWCDomain. Therefore need to allow this domain as well.
    // Allow kMSDomain for contrib op like Gelu
    if (!is_onnx_domain && !is_nhwc_domain && !is_ms_domain) {
      continue;
    }

    const char* op_type_cstr = nullptr;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOperatorType(node, &op_type_cstr));
    std::string op_type = op_type_cstr;

    auto op_rule = op_type_to_selectors_map_.find(op_type);
    if (op_rule == op_type_to_selectors_map_.cend()) {
      continue;
    }

    const auto& op_versions_and_selector = *op_rule->second;

    // check the supported versions if specified
    const std::vector<int>& versions = op_versions_and_selector.op_versions_map.find(op_type)->second;
    if (!versions.empty()) {
      int since_version = 0;
      ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetSinceVersion(node, &since_version));

      if (std::find(versions.cbegin(), versions.cend(), since_version) == versions.cend()) {
        ORT_CXX_LOGF(logger, ORT_LOGGING_LEVEL_VERBOSE, "Op version is not supported for %s", op_type_cstr);
        continue;
      }
    }

    std::optional<NodeGroup> qdq_node_group;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(op_versions_and_selector.selector->GetQDQSelection(*node, qdq_node_group));
    if (qdq_node_group.has_value()) {
      const auto& qdq_group = *qdq_node_group;
      qdq_selections.push_back(qdq_group);
    }
  }

  result = std::move(qdq_selections);
  return Ort::Status{nullptr};
}
}  // namespace

Ort::Status GetAllNodeUnits(const OrtGraph& graph, const Ort::Logger& logger,
                            /*out*/ std::vector<std::unique_ptr<NodeUnit>>& node_units,
                            /*out*/ std::unordered_map<const OrtNode*, const NodeUnit*>& node_to_node_unit) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t num_nodes = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetNumNodes(&graph, &num_nodes));

  std::vector<const OrtNode*> nodes(num_nodes);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Graph_GetNodes(&graph, nodes.data(), nodes.size()));

  std::vector<std::unique_ptr<NodeUnit>> node_unit_holder;
  std::unordered_map<const OrtNode*, const NodeUnit*> node_unit_map;

  const auto add_node_unit_to_map = [&](const std::vector<const OrtNode*>& nodes, const NodeUnit* node_unit) {
    for (const OrtNode* node : nodes) {
      node_unit_map.insert({node, node_unit});
    }
  };

  // Get QDQ NodeUnits first
  QDQ::SelectorManager selector_mgr;
  std::vector<NodeGroup> qdq_selections;
  ORT_EP_UTILS_CXX_RETURN_IF_ERROR(selector_mgr.GetQDQSelections(nodes, logger, qdq_selections));

  for (const NodeGroup& qdq_selection : qdq_selections) {
    std::unique_ptr<NodeUnit> qdq_unit;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(NodeUnit::MakeQDQGroup(qdq_selection, qdq_unit));

    // Fill the node to node_unit map for all nodes in the QDQ Group
    add_node_unit_to_map(qdq_selection.dq_nodes, qdq_unit.get());
    add_node_unit_to_map(qdq_selection.q_nodes, qdq_unit.get());
    add_node_unit_to_map({qdq_selection.target_node}, qdq_unit.get());
    if (qdq_selection.redundant_clip_node != nullptr) {
      node_unit_map.insert({qdq_selection.redundant_clip_node, qdq_unit.get()});
    }

    node_unit_holder.push_back(std::move(qdq_unit));
  }

  // Get the left over SingleNode NodeUnits
  for (const OrtNode* node : nodes) {
    // This is already part of a QDQ NodeUnit
    if (node_unit_map.find(node) != node_unit_map.cend()) {
      continue;
    }

    std::unique_ptr<NodeUnit> node_unit;
    ORT_EP_UTILS_CXX_RETURN_IF_ERROR(NodeUnit::MakeSingleNode(*node, node_unit));
    node_unit_map[node] = node_unit.get();
    node_unit_holder.push_back(std::move(node_unit));
  }

  node_units = std::move(node_unit_holder);
  node_to_node_unit = std::move(node_unit_map);
  return Ort::Status{nullptr};
}

}  // namespace QDQ
}  // namespace OrtEpUtils

#endif  // ORT_EP_UTILS_QDQ_UTILS_IMPL
