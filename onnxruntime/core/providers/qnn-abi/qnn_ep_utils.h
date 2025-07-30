// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <optional>
#include <tuple>

#include "ort_api.h"
#include "qnn_ep.h"
#include "core/graph/abi_graph_types.h"

// Forward declaration of OrtNode
struct OrtNode;

// Forward declaration
namespace onnxruntime {
class QnnEp;

namespace QDQ {
// Define NodeGroup structure similar to the one in shared/utils.h
struct OrtNodeGroup {
  std::vector<int> dq_nodes;
  std::vector<int> q_nodes;
  int target_node;
  std::optional<int> redundant_clip_node;

  // Validator to check if the set of nodes can form a valid QDQ NodeGroup.
  // Checks target node is only consumer of each DQ, and that the outputs remain valid if the QDQ node group was to
  // be converted into a single node with a quantized operator.
  static Status CanCreateNodeGroup(const OrtGraph& graph,
                                   const OrtNode& target_node,
                                   const OrtNode* redundant_clip_node,
                                   gsl::span<const OrtNode* const> dq_nodes,
                                   gsl::span<const OrtNode* const> q_nodes) {
    graph;
    target_node;
    redundant_clip_node;
    dq_nodes;
    q_nodes;
    // // Within a QDQ node group, a target node input is the only consumer of each DQ.
    // // This should have been ensured by the EnsureUniqueDQForNodeUnit graph transformer, but other graph modifications
    // // may have happened since. Verify that this is still true.
    // for (const auto* dq_node : dq_nodes) {
    //   const bool dq_produces_graph_output = graph_viewer.NodeProducesGraphOutput(*dq_node);
    //   ORT_RETURN_IF(dq_produces_graph_output,
    //                 "QDQ node group cannot have DQ node that produces a graph output. DQ node: ", dq_node->Name(),
    //                 ", target node: ", target_node.Name());

    //   const bool dq_has_single_output_edge_to_target =
    //       dq_node->GetOutputEdgesCount() == 1 &&
    //       dq_node->OutputEdgesBegin()->GetNode().Index() == target_node.Index();
    //   ORT_RETURN_IF_NOT(dq_has_single_output_edge_to_target,
    //                     "QDQ node group cannot have DQ that doesn't have a single output edge to the target node. "
    //                     "DQ node: ",
    //                     dq_node->Name(), ", target node: ", target_node.Name());
    // }

    // // If redundant_clip_node is present, currently we require target node has only one output edge, which is connected to
    // // the redundant_clip_node. The redundant_clip_node's output is consumed by the Q node that can be fused with itself.
    // if (redundant_clip_node) {
    //   ORT_RETURN_IF_NOT(!graph_viewer.NodeProducesGraphOutput(target_node) && target_node.GetOutputEdgesCount() == 1 &&
    //                         target_node.OutputEdgesBegin()->GetNode().Index() == redundant_clip_node->Index(),
    //                     "QDQ node group cannot have target node with more than one output edge if there is redunant clip "
    //                     "node. target node: ",
    //                     target_node.Name());
    //   ORT_RETURN_IF_NOT(
    //       !graph_viewer.NodeProducesGraphOutput(*redundant_clip_node) && q_nodes.size() == 1 &&
    //           redundant_clip_node->GetOutputEdgesCount() == 1 &&
    //           redundant_clip_node->OutputEdgesBegin()->GetNode().Index() == q_nodes[0]->Index(),
    //       "QDQ node group cannot have redudant clip node that doesn't have a single output edge to a Q node. "
    //       "redundant clip node: ",
    //       redundant_clip_node->Name());
    //   return Status::OK();
    // }

    // // an output from the target node can have either Q consumers or direct consumers. it cannot have both.
    // // this must be checked on a per output basis.
    // // e.g. TopK produces values and indices. The indices output won't be quantized, so even if we replace the TopK QDQ
    // // node group with a quantized TopK, an int64_t indices value will be produced and can provide a graph output.
    // if (!q_nodes.empty()) {
    //   auto cur_edge = target_node.OutputEdgesBegin();
    //   auto end_edge = target_node.OutputEdgesEnd();
    //   std::vector<const Node*> output_consumers(target_node.OutputDefs().size(), nullptr);

    //   for (; cur_edge != end_edge; ++cur_edge) {
    //     auto output_idx = cur_edge->GetSrcArgIndex();
    //     const Node& this_consumer = cur_edge->GetNode();
    //     const Node* existing_consumer = output_consumers[output_idx];

    //     if (existing_consumer != nullptr) {
    //       // another edge for this output. either both are Q or both are not.
    //       bool valid = true;
    //       if (existing_consumer->OpType() == "QuantizeLinear") {
    //         valid = this_consumer.OpType() == "QuantizeLinear";
    //       } else {
    //         valid = this_consumer.OpType() != "QuantizeLinear";
    //       }

    //       ORT_RETURN_IF_NOT(valid,
    //                         "QDQ node group cannot have an output from the target node being consumed by a Q node and "
    //                         "a non-Q node. target node: ",
    //                         target_node.Name());
    //     } else {
    //       output_consumers[output_idx] = &this_consumer;
    //     }
    //   }

    //   const auto& graph_outputs = graph_viewer.GetOutputs();
    //   for (size_t idx = 0, end = output_consumers.size(); idx < end; ++idx) {
    //     // any output with a Q cannot be a graph output as it will disappear if the QDQ node unit is converted to
    //     // a quantized op.
    //     if (output_consumers[idx] != nullptr && output_consumers[idx]->OpType() == "QuantizeLinear") {
    //       const auto& output_name = target_node.OutputDefs()[idx]->Name();
    //       bool is_graph_output = std::any_of(graph_outputs.begin(), graph_outputs.end(),
    //                                          [&output_name](const NodeArg* node_arg) {
    //                                            return node_arg->Name() == output_name;
    //                                          });
    //       ORT_RETURN_IF(is_graph_output,
    //                     "QDQ node group cannot have an output from the target node that is consumed by a Q node and "
    //                     "a graph output. target node: ",
    //                     target_node.Name(), " output idx:", idx);
    //     }
    //   }
    // }

    return Status::OK();
  }
};

// Forward declaration
class OrtNodeGroupSelector;

// struct that provides a join between selector and op versions supported
struct OrtOpVersionsAndSelector {
  using OpVersionsMap = std::unordered_map<std::string, std::vector<int>>;

  OrtOpVersionsAndSelector(const OpVersionsMap& ops_and_versions_in,
                           std::unique_ptr<OrtNodeGroupSelector> selector_in)
      : op_versions_map{ops_and_versions_in},
        selector{std::move(selector_in)} {}

  OpVersionsMap op_versions_map;
  std::unique_ptr<OrtNodeGroupSelector> selector;
};

// class that manages a set of node group selectors
class OrtSelectors {
 public:
  OrtSelectors() = default;

  // register a selector for the specified ops.
  void RegisterSelector(const OrtOpVersionsAndSelector::OpVersionsMap& ops_and_versions_in,
                        std::unique_ptr<OrtNodeGroupSelector> selector_in);

  const std::vector<std::unique_ptr<OrtOpVersionsAndSelector>>& SelectorsSet() const {
    return selectors_set_;
  }

 private:
  std::vector<std::unique_ptr<OrtOpVersionsAndSelector>> selectors_set_;
};

// Base class for node group selectors
class OrtNodeGroupSelector {
 public:
  virtual ~OrtNodeGroupSelector() = default;
  // Check if the node group is supported
  bool virtual Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                     const OrtNode* redundant_clip_node,
                     const std::vector<const OrtNode*>& dq_nodes,
                     const std::vector<const OrtNode*>& q_nodes) const = 0;

 protected:
  // Helper function to check if a node has the expected number of DQ inputs and Q outputs
  bool CheckQDQNodes(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
                     const OrtNode* redundant_clip_node,
                     const std::vector<const OrtNode*>& dq_nodes,
                     const std::vector<const OrtNode*>& q_nodes,
                     int num_dq_inputs = -1,
                     bool is_empty_q_nodes_allowed = false) const;
};

// Single DQ -> node that does not change data -> Q.
// Zero point and scale are constant scalars and must match
class OrtDropQDQNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtDropQDQNodeGroupSelector(bool allow_16bit = true, bool allow_4bit = true,
                                       bool allow_nonpositive_scale = true)
      : allow_16bit_(allow_16bit), allow_4bit_(allow_4bit), allow_nonpositive_scale_(allow_nonpositive_scale) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_16bit_;
  bool allow_4bit_;
  bool allow_nonpositive_scale_;
};

// Selector for drop DQ operations
class OrtDropDQNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtDropDQNodeGroupSelector(bool allow_16bit = true, bool allow_4bit = true)
      : allow_16bit_(allow_16bit), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_16bit_;
  bool allow_4bit_;
};

// Selector for unary operations
class OrtUnaryNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtUnaryNodeGroupSelector(bool allow_16bit = true, bool allow_4bit = true)
      : allow_16bit_(allow_16bit), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_16bit_;
  bool allow_4bit_;
};

// Selector for binary operations
class OrtBinaryNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtBinaryNodeGroupSelector(bool allow_16bit = true, bool allow_4bit = true)
      : allow_16bit_(allow_16bit), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_16bit_;
  bool allow_4bit_;
};

// Selector for variadic operations
class OrtVariadicNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtVariadicNodeGroupSelector(bool allow_16bit = true, bool allow_4bit = true)
      : allow_16bit_(allow_16bit), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_16bit_;
  bool allow_4bit_;
};

// DQ node -> Split -> multiple Q nodes with equal quantization types.
// Optionally, the selector can require all input and output quantization parameters to be
// equal and constant.
class OrtSplitNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtSplitNodeGroupSelector(bool req_equal_quant_params = false, bool allow_4bit = true)
      : req_equal_quant_params_(req_equal_quant_params), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool req_equal_quant_params_;  // If true, only selects a node group if the input and output
                                 // quantization parameters are all equal/constant, which enables the
                                 // optimizer to drop the Q/DQ ops if the group is assigned to the CPU EP.
  bool allow_4bit_;
};

// DQ nodes for X, W and optionally B -> node -> Q
class OrtConvNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  // default to 'true'
  OrtConvNodeGroupSelector(bool int8_allowed = true, bool allow_16bit = true, bool allow_4bit_weight = true)
      : int8_allowed_(int8_allowed), allow_16bit_(allow_16bit), allow_4bit_weight_(allow_4bit_weight) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool int8_allowed_;
  bool allow_16bit_;
  bool allow_4bit_weight_;
};

class OrtWhereNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtWhereNodeGroupSelector(bool allow_16bit = true, bool allow_4bit = true)
      : allow_16bit_(allow_16bit), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_16bit_;
  bool allow_4bit_;
};

class OrtPadNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtPadNodeGroupSelector() = default;

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// one ore more DQ nodes for each input -> node -> Q
class OrtEinsumNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtEinsumNodeGroupSelector(bool allow_int8 = true, bool allow_16bit = true, bool allow_4bit = true)
      : allow_int8_(allow_int8), allow_16bit_(allow_16bit), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_int8_;
  bool allow_16bit_;
  bool allow_4bit_;
};

class OrtReciprocalNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtReciprocalNodeGroupSelector(bool allow_int8 = true, bool allow_16bit = true, bool allow_4bit = true)
      : allow_int8_(allow_int8), allow_16bit_(allow_16bit), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_int8_;
  bool allow_16bit_;
  bool allow_4bit_;
};

// 2 DQ nodes for input -> node -> optional Q if QLinearMatMul, MatMulIntegerToFloat if not
// The lack of a trailing Q isn't really a QDQ node group, so we default support for that to off.
class OrtMatMulNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  OrtMatMulNodeGroupSelector(bool int8_allowed = true,
                             bool matmulintegertofloat_allowed = false,
                             bool allow_16bit = true,
                             bool allow_4bit = true)
      : int8_allowed_(int8_allowed),
        matmulintegertofloat_allowed_(matmulintegertofloat_allowed),
        allow_16bit_(allow_16bit),
        allow_4bit_(allow_4bit) {
  }

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool int8_allowed_;
  bool matmulintegertofloat_allowed_;
  bool allow_16bit_;
  bool allow_4bit_;
};

// Convert "1 DQ node for input B -> MatMul" to "MatMulNBits"
class OrtDQMatMulNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// Input: DQ nodes for A, B and optional C
// Output: optional Q node for Y
class OrtGemmNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  explicit OrtGemmNodeGroupSelector(bool allow_16bit = true, bool allow_4bit = true)
      : allow_16bit_(allow_16bit), allow_4bit_(allow_4bit) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool allow_16bit_;
  bool allow_4bit_;
};

// Input: DQ nodes for input, scale, and B
// Output: Q node for output
class OrtInstanceAndLayerNormalizationNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// DQ nodes for X, W and optionally B, not used for mean, var -> node -> Q
class OrtBatchNormalizationNodeGroupSelector : public OrtNodeGroupSelector {
 public:
  // default to 'true'
  OrtBatchNormalizationNodeGroupSelector(bool int8_allowed = true) : int8_allowed_(int8_allowed) {}

  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;

 private:
  bool int8_allowed_;
};

// 2 DQ nodes providing input -> node with bool output tensor.
// Example: Equal, Less, Greater.
class OrtLogicalComparisonNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// TopK has 1 DQ input node and 1 Q output node.
// Zero point and scale are constant scalars and must match
class OrtTopKNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// one DQ node for first input -> node -> Q
class OrtCumSumNodeGroupSelector : public OrtNodeGroupSelector {
  bool Check(const OrtGraph* graph, const OrtApi& ort_api, const OrtNode* node,
             const OrtNode* redundant_clip_node,
             const std::vector<const OrtNode*>& dq_nodes,
             const std::vector<const OrtNode*>& q_nodes) const override;
};

// SelectorManager for OrtGraph
class OrtSelectorManager {
 public:
  OrtSelectorManager();

  // Method that finds and returns a vector of QDQ::NodeGroup in a given OrtGraph
  std::vector<OrtNodeGroup> GetOrtQDQSelections(const OrtGraph* graph, const OrtApi& ort_api, const logging::Logger& logger) const;

 private:
  OrtSelectors ort_selectors_;

  std::unordered_map<std::string, const OrtOpVersionsAndSelector*> op_type_to_selectors_map_;

  void InitializeSelectorsMap();

  void CreateSelectors();
};
}  // namespace QDQ

// Function to get QDQ node units from OrtGraph
std::pair<std::vector<std::unique_ptr<OrtNodeUnit>>, std::unordered_map<const OrtNode*, const OrtNodeUnit*>>
GetAllOrtNodeUnits(OrtApi ort_api, const OrtGraph* graph, const logging::Logger& logger);

}  // namespace onnxruntime
