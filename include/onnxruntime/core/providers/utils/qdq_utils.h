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

/**
@class EdgeEnd
Class representing the end of an edge. It could be an input or output edge end of a node.
For the node's input edge end, it's the source end, as the destination end is the node itself.
For the node's output edge end, it's the destination end, as the source end is the node itself.
*/
class EdgeEnd {
 public:
  /**
  Construct an EdgeEnd
  @param node The source node if this is an input edge to the current node,
  or the destination node if this is an output edge from the current node.
  @param src_arg_index The node arg index of source node of the edge.
  @param dst_arg_index The node arg index of destination node of the edge.
  */
  EdgeEnd(const OrtNode& node, int src_arg_index, int dst_arg_index) noexcept;

  /** Construct a control edge.
  @param node The node the edge joins to the current node.
  */
  explicit EdgeEnd(const OrtNode& node) noexcept;

  /** Gets the Node that this EdgeEnd refers to. */
  const OrtNode& GetNode() const noexcept { return *node_; }

  /** Gets the source arg index.
  @returns the source arg index of <*this> edge.*/
  int GetSrcArgIndex() const { return src_arg_index_; }

  /** Gets the destination arg index.
  @returns the destination arg index of <*this> edge.*/
  int GetDstArgIndex() const { return dst_arg_index_; }

 private:
  const OrtNode* node_;
  const int src_arg_index_;
  const int dst_arg_index_;
};

/** Struct to provide sorting between EdgeEnd instances based on NodeIndex first, and NodeArg::Name second. */
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
 public:
  // NodeUnit type
  enum class Type : uint8_t {
    SingleNode,  // The NodeUnit contains a single node
    QDQGroup,    // The NodeUnit contain a QDQ group of nodes, such as "DQ->Sigmoid->Q"
  };

 public:
  explicit NodeUnit(const OrtNode& node);
  explicit NodeUnit(const OrtGraph& graph, const NodeGroup& node_group);

 private:
  // Initialization for a NodeUnit that contains a single node
  void InitForSingleNode();

  const std::vector<const OrtNode*> dq_nodes_;  // dq nodes for this NodeUnit, not necessarily all inputs
  const OrtNode& target_node_;
  const OrtNode* redundant_clip_node_;         // Optional redundant clip node for the QDQ group, nullptr if not present.
  const std::vector<const OrtNode*> q_nodes_;  // q-nodes for this NodeUnit. not necessarily all outputs
  const Type type_;

  std::vector<NodeUnitIODef> inputs_;
  std::vector<NodeUnitIODef> outputs_;

  size_t input_edge_count_;  // total number of input edges

  // output edges, hiding any Q nodes involved. src_idx will be value from target node. only used for QDQ node group.
  EdgeSet output_edges_;
};

}  // namespace OrtEpUtils

// End of header
#endif  // INCLUDE_ONNXRUNTIME_CORE_PROVIDERS_UTILS_QDQ_UTILS_H_

//
// IMPLEMENTATION BELOW
//
#ifdef ORT_EP_UTILS_QDQ_UTILS_IMPL

#include <cassert>

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

// Get the number of input edges that come from another node upstream.
static Ort::Status GetNodeInputEdgeCount(const OrtNode* node, size_t& num_input_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t num_inputs = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumInputs(node, &num_inputs));

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetInputs(node, inputs.data(), inputs.size()));

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

static Ort::Status GetNodeOutputEdges(const OrtNode* node, EdgeSet& output_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t num_outputs = 0;
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetNumOutputs(node, &num_outputs));

  std::vector<const OrtValueInfo*> outputs(num_outputs);
  ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()));

  for (size_t src_idx = 0; src_idx < num_outputs; ++src_idx) {
    const OrtValueInfo* value_info = outputs[src_idx];

    size_t num_consumers = 0;
    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueNumConsumers(value_info, &num_consumers));

    std::vector<const OrtNode*> consumer_nodes(num_consumers);
    std::vector<int64_t> consumer_indices(num_consumers);

    ORT_EP_UTILS_C_RETURN_IF_ERROR(ort_api.ValueInfo_GetValueConsumers(value_info, consumer_nodes.data(),
                                                                       consumer_indices.data(), num_consumers));

    // Build up an edge with `node` as the source and consumer nodes as the destinations. Filter out
    // consumer indices of `-1` (implicit).
    for (size_t c_idx = 0; c_idx < num_consumers; c_idx++) {
      const OrtNode* dst_node = consumer_nodes[c_idx];
      int64_t dst_idx = consumer_indices[c_idx];

      if (dst_idx < 0) {
        continue;  // Skip implicit input to consumer node.
      }

      EdgeEnd edge_end(*node, static_cast<int>(src_idx), static_cast<int>(dst_idx));
      output_edges.insert(edge_end);
    }
  }
}

}  // namespace OrtEpUtils

#endif  // ORT_EP_UTILS_QDQ_UTILS_IMPL
