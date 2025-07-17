// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

// This compilation unit (ort_api.h/.cc) encapsulates the interface between the EP and ORT in a manner
// that allows QNN EP to built either as a static library or a dynamic shared library.
// The preprocessor macro `BUILD_QNN_EP_STATIC_LIB` is defined and set to 1 if QNN EP
// is built as a static library.

#ifdef _WIN32
#include <Windows.h>
#include <winmeta.h>
#include "core/platform/tracing.h"
#include "core/platform/windows/logging/etw_sink.h"
#endif

#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/capture.h"
#include "core/common/path_string.h"
#include "core/graph/onnx_protobuf.h"
#include "core/platform/env.h"
#include "core/framework/data_types.h"
#include "core/framework/float16.h"
#include "core/framework/run_options.h"
#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/node_unit.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/constants.h"
#include "core/graph/basic_types.h"
#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/common.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/abi_logger.h"
#include "core/graph/abi_graph_types.h"
#include "core/session/abi_session_options_impl.h"

// #define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_c_api.h"
// #endif

#include "core/common/inlined_containers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"


// #include "core/session/onnxruntime_cxx_api.h"

#include <memory>
#include <string>
#include <vector>

namespace onnxruntime {

// inline void InitOrtCppApi() {
//   // Call util function in provider bridge that initializes the global api_ object.
//   InitProviderOrtApi();
// }

// /// <summary>
// /// Creates an onnxruntime or onnx object. Works for both static and shared library builds of QNN EP.
// /// <!-- Example: auto model = Factory<Model>::Create(/* args ... */); -->
// /// Example: auto model = Factory&lt;Model&gt;::Create(/* args ... */);
// /// </summary>
// /// <typeparam name="T">Type of the object to create</typeparam>
// template <typename T>
// struct Factory {
//   template <typename... Params>
//   static inline std::unique_ptr<T> Create(Params&&... params) {
//     return T::Create(std::forward<Params>(params)...);
//   }
// };

// inline const ConfigOptions& RunOptions__GetConfigOptions(const RunOptions& run_options) {
//   return run_options.GetConfigOptions();
// }

// inline std::unique_ptr<IndexedSubGraph>& ComputeCapability__SubGraph(ComputeCapability& compute_cability) {
//   return compute_cability.SubGraph();
// }

// inline std::vector<NodeIndex>& IndexedSubGraph__Nodes(IndexedSubGraph& indexed_sub_graph) {
//   return indexed_sub_graph.Nodes();
// }

// std::vector<const Node*> Graph__Nodes(const Graph& graph);

// inline std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const Node*, const NodeUnit*>>
// GetQDQNodeUnits(const GraphViewer& graph_viewer, const logging::Logger& logger) {
//   return QDQ::GetAllNodeUnits(&graph_viewer, logger);
// }

// Forward declaration for OrtNode which is used in OrtNodeUnit

struct OrtNodeUnitIODef {
  std::string name;
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
};

class OrtNodeUnit {
 public:
  // NodeUnit type
  enum class Type : uint8_t {
    SingleNode,  // The NodeUnit contains a single node
    QDQGroup,    // The NodeUnit contain a QDQ group of nodes, such as "DQ->Sigmoid->Q"
  };

 public:
  explicit OrtNodeUnit(const OrtNode& node, const OrtApi& ort_api);
  // explicit NodeUnit(const GraphViewer& graph_viewer, const QDQ::NodeGroup& node_group);
  // NodeUnit(gsl::span<const Node* const> dq_nodes, const Node& target_node, const Node* redundant_clip_node,
  //          gsl::span<const Node* const> q_nodes, Type unit_type,
  //          gsl::span<const NodeUnitIODef> inputs, gsl::span<const NodeUnitIODef> outputs,
  //          size_t input_edge_count, Node::EdgeSet output_edges);

  Type UnitType() const noexcept { return type_; }

  // const std::vector<NodeUnitIODef>& Inputs() const noexcept { return inputs_; }
  // const std::vector<NodeUnitIODef>& Outputs() const noexcept { return outputs_; }
  const std::vector<OrtNodeUnitIODef>& Inputs() const noexcept { return inputs_; }
  const std::vector<OrtNodeUnitIODef>& Outputs() const noexcept { return outputs_; }

  // const std::string& Domain() const noexcept;
  const std::string& OpType() const noexcept{ return target_node_.GetOpType(); }
  const std::string& Name() const noexcept { return target_node_.GetName(); }
  // int SinceVersion() const noexcept;
  // NodeIndex Index() const noexcept;
  // const std::filesystem::path& ModelPath() const noexcept;
  // ProviderType GetExecutionProviderType() const noexcept;

  const OrtNode& GetNode() const noexcept { return target_node_; }
  // const Node* GetRedundantClipNode() const noexcept { return redundant_clip_node_; }
  // const std::vector<const Node*>& GetDQNodes() const noexcept { return dq_nodes_; }
  // const std::vector<const Node*>& GetQNodes() const noexcept { return q_nodes_; }
  std::vector<const OrtNode*> GetAllNodesInGroup() const noexcept {
    std::vector<const OrtNode*> all_nodes = dq_nodes_;
    all_nodes.push_back(&target_node_);
    if (redundant_clip_node_) {
      all_nodes.push_back(redundant_clip_node_);
    }
    all_nodes.reserve(all_nodes.size() + q_nodes_.size());
    for (auto& n : q_nodes_)
      all_nodes.push_back(n);
    return all_nodes;
  }

  // /// Number of input edges to the logical node. For a QDQ node this is the count of input edges to the DQ nodes
  // /// plus any other edges to the target node for inputs that are not via a DQ node.
  // size_t InputEdgeCount() const { return input_edge_count_; }

  // // output edges. src index is for outputs of the target node. dest index and node is for consumer of node unit
  // // output. any Q nodes are hidden.
  // Node::EdgeConstIterator OutputEdgesBegin() const;
  // Node::EdgeConstIterator OutputEdgesEnd() const;

 private:
  // // Initialization for a NodeUnit that contains a single node
  void InitForSingleNode(const OrtApi& ort_api);

  const std::vector<const OrtNode*> dq_nodes_;  // dq nodes for this NodeUnit, not necessarily all inputs
  const OrtNode& target_node_;
  const OrtNode* redundant_clip_node_;         // Optional redundant clip node for the QDQ group, nullptr if not present.
  const std::vector<const OrtNode*> q_nodes_;  // q-nodes for this NodeUnit. not necessarily all outputs
  const Type type_;

  // std::vector<NodeUnitIODef> inputs_;
  // std::vector<NodeUnitIODef> outputs_;
  std::vector<OrtNodeUnitIODef> inputs_;
  std::vector<OrtNodeUnitIODef> outputs_;

  // size_t input_edge_count_;  // total number of input edges

  // // output edges, hiding any Q nodes involved. src_idx will be value from target node. only used for QDQ node group.
  // Node::EdgeSet output_edges_;
};

/**
 * Wrapping onnxruntime::Node for retrieving attribute values
 */

class OrtNodeAttrHelper {
 public:
  explicit OrtNodeAttrHelper(const OrtApi& ort_api, const OrtNode& node);

  // Get the attributes from the target node of the node_unit
  explicit OrtNodeAttrHelper(const OrtApi& ort_api, const OrtNodeUnit& node_unit);

  /*
   * Get with default
   */
  float Get(const std::string& key, float def_val) const;
//   std::vector<float> Get(const std::string& key, const std::vector<float>& def_val) const;

//   int64_t Get(const std::string& key, int64_t def_val) const;
//   std::vector<int64_t> Get(const std::string& key, const std::vector<int64_t>& def_val) const;

//   const std::string& Get(const std::string& key, const std::string& def_val) const;
//   std::vector<std::string> Get(const std::string& key, const std::vector<std::string>& def_val) const;

//   // Convert the i() or ints() of the attribute from int64_t to int32_t
//   int32_t Get(const std::string& key, int32_t def_val) const;
//   std::vector<int32_t> Get(const std::string& key, const std::vector<int32_t>& def_val) const;

//   // Convert the i() or ints() of the attribute from int64_t to uint32_t
//   uint32_t Get(const std::string& key, uint32_t def_val) const;
//   std::vector<uint32_t> Get(const std::string& key, const std::vector<uint32_t>& def_val) const;

//   /*
//    * Get without default.
//    */
//   std::optional<float> GetFloat(const std::string& key) const;
//   std::optional<std::vector<float>> GetFloats(const std::string& key) const;

//   std::optional<int64_t> GetInt64(const std::string& key) const;
//   std::optional<std::vector<int64_t>> GetInt64s(const std::string& key) const;

//   std::optional<std::string> GetString(const std::string& key) const;

//   bool HasAttr(const std::string& key) const;

 private:
  const OrtNode& node_;
  const OrtApi& ort_api_;
  const OrtArrayOfConstObjects** attributes;
};

}  // namespace onnxruntime
