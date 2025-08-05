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
