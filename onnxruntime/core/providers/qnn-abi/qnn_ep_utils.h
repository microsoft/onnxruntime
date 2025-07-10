// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>
#include <memory>
#include <optional>
#include <string>

#include "test/autoep/library/example_plugin_ep_utils.h"

namespace onnxruntime {

// Forward declarations
struct OrtNode;
struct OrtLogger;
struct OrtGraph;
struct OrtApi;

namespace QDQ {
struct NodeGroup {
  std::vector<size_t> dq_nodes;
  std::vector<size_t> q_nodes;
  size_t target_node;
  std::optional<size_t> redundant_clip_node;
};
}  // namespace QDQ

// Simple NodeUnit implementation for ABI layer
class NodeUnit {
 public:
  enum class Type : uint8_t {
    SingleNode,
    QDQGroup,
  };

  explicit NodeUnit(const OrtNode* node)
    : target_node_(node), type_(Type::SingleNode) {}

  NodeUnit(std::vector<const OrtNode*> dq_nodes, const OrtNode* target_node,
           const OrtNode* redundant_clip_node, std::vector<const OrtNode*> q_nodes)
    : dq_nodes_(std::move(dq_nodes)),
      target_node_(target_node),
      redundant_clip_node_(redundant_clip_node),
      q_nodes_(std::move(q_nodes)),
      type_(Type::QDQGroup) {}

  Type UnitType() const { return type_; }
  const OrtNode* GetNode() const { return target_node_; }
  const OrtNode* GetRedundantClipNode() const { return redundant_clip_node_; }
  const std::vector<const OrtNode*>& GetDQNodes() const { return dq_nodes_; }
  const std::vector<const OrtNode*>& GetQNodes() const { return q_nodes_; }

  std::vector<const OrtNode*> GetAllNodesInGroup() const {
    std::vector<const OrtNode*> all_nodes;
    all_nodes.insert(all_nodes.end(), dq_nodes_.begin(), dq_nodes_.end());
    all_nodes.push_back(target_node_);
    if (redundant_clip_node_) {
      all_nodes.push_back(redundant_clip_node_);
    }
    all_nodes.insert(all_nodes.end(), q_nodes_.begin(), q_nodes_.end());
    return all_nodes;
  }

 private:
  std::vector<const OrtNode*> dq_nodes_;
  const OrtNode* target_node_;
  const OrtNode* redundant_clip_node_ = nullptr;
  std::vector<const OrtNode*> q_nodes_;
  Type type_;
};

// Helper function to identify QDQ patterns and create NodeUnit objects
inline std::pair<std::vector<std::unique_ptr<NodeUnit>>, std::unordered_map<const OrtNode*, const NodeUnit*>>
GetQDQNodeUnits(const OrtGraph* graph, const OrtLogger* logger, const ApiPtrs& api_ptrs) {
  const OrtApi& ort_api = api_ptrs.ort_api;
  std::vector<std::unique_ptr<NodeUnit>> node_units;
  std::unordered_map<const OrtNode*, const NodeUnit*> node_map;

  // Get all nodes from the graph
  OrtArrayOfConstObjects* graph_nodes = nullptr;
  if (ort_api.Graph_GetNodes(graph, &graph_nodes) != nullptr) {
    return {std::move(node_units), std::move(node_map)};
  }

  size_t num_nodes = 0;
  if (ort_api.ArrayOfConstObjects_GetSize(graph_nodes, &num_nodes) != nullptr) {
    ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    return {std::move(node_units), std::move(node_map)};
  }

  const void* const* node_data = nullptr;
  if (ort_api.ArrayOfConstObjects_GetData(graph_nodes, &node_data) != nullptr) {
    ort_api.ReleaseArrayOfConstObjects(graph_nodes);
    return {std::move(node_units), std::move(node_map)};
  }

  // Create a map of node index to node pointer for easy lookup
  std::unordered_map<size_t, const OrtNode*> node_index_map;
  // Map to store node type by node pointer
  std::unordered_map<const OrtNode*, std::string> node_type_map;
  // Maps to store node connections
  std::unordered_map<const OrtNode*, std::vector<const OrtNode*>> node_inputs;
  std::unordered_map<const OrtNode*, std::vector<const OrtNode*>> node_outputs;

  // First pass: build node maps and collect node types
  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);
    size_t node_id = 0;
    if (ort_api.Node_GetId(node, &node_id) == nullptr) {
      node_index_map[node_id] = node;

      // Get node type
      const char* op_type = nullptr;
      if (ort_api.Node_GetOperatorType(node, &op_type) == nullptr && op_type != nullptr) {
        node_type_map[node] = std::string(op_type);
      }

      // Get node inputs and outputs to build the graph structure
      // Note: Since Node_GetInputNodes is not available in the ABI, we'll use a simplified approach
      // This is a placeholder for actual implementation that would use available API functions
      OrtArrayOfConstObjects* input_nodes = nullptr;
      if (false) { // Placeholder - would use actual API function when available
        size_t num_inputs = 0;
        // Placeholder for actual implementation
        // In a real implementation, we would use the OrtApi to get input nodes
        // and build the node_inputs and node_outputs maps
      }
    }
  }

    // Second pass: For now, we'll create single node units for all nodes
    // In a real implementation, we would identify QDQ patterns here
    // This is a simplified version that doesn't rely on node connections
    // which would require additional API functions

  // Create single node units for all nodes
  for (size_t i = 0; i < num_nodes; ++i) {
    const OrtNode* node = static_cast<const OrtNode*>(node_data[i]);

    // Create a single node NodeUnit
    auto node_unit = std::make_unique<NodeUnit>(node);
    node_map[node] = node_unit.get();
    node_units.push_back(std::move(node_unit));
  }

  ort_api.ReleaseArrayOfConstObjects(graph_nodes);

  if (logger != nullptr) {
    std::string log_message = "Created " + std::to_string(node_units.size()) + " NodeUnits total";
    ort_api.Logger_LogMessage(logger, ORT_LOGGING_LEVEL_VERBOSE, log_message.c_str(),
                            ORT_FILE, __LINE__, __FUNCTION__);
  }

  return {std::move(node_units), std::move(node_map)};
}

}  // namespace onnxruntime
