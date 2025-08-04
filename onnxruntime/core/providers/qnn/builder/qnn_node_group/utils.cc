#include "core/providers/qnn/builder/qnn_node_group/utils.h"

#include <gsl/gsl>
#include <string_view>
#include <unordered_map>

#include "core/providers/qnn/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

const NodeUnit* GetOnlyChildOfType(const GraphViewer& graph_viewer,
                                   const NodeUnit& parent_node_unit,
                                   gsl::span<const std::string_view> child_op_types,
                                   const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                   const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Node& parent_node = parent_node_unit.GetNode();

  // Parent must have a single child (1 output edge) and must not produce a graph output.
  if (parent_node.GetOutputEdgesCount() != 1 || graph_viewer.NodeProducesGraphOutput(parent_node)) {
    return nullptr;
  }

  // Child must be of a valid type.
  const Node& child_node = parent_node.OutputEdgesBegin()->GetNode();
  if (graph_viewer.GetNode(child_node.Index()) == nullptr) {
    return nullptr;  // Node is not in this GraphViewer
  }
  const std::string& child_type = child_node.OpType();
  bool is_valid_child_type = false;

  for (const auto& valid_op_type : child_op_types) {
    if (valid_op_type == child_type) {
      is_valid_child_type = true;
      break;
    }
  }

  if (!is_valid_child_type) {
    return nullptr;
  }

  const auto child_node_unit_it = node_unit_map.find(&child_node);
  if (child_node_unit_it == node_unit_map.end()) {
    return nullptr;
  }
  const NodeUnit* child_node_unit = child_node_unit_it->second;

  // Check if child node has already been handled. Should not be the case if the calling
  // fusion function has been called in topological order, but check to be safe.
  if (qnn_node_group_map.count(child_node_unit) != 0) {
    return nullptr;
  }

  // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (child_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
    return nullptr;
  }

  return child_node_unit;
}

const NodeUnit* GetChildOfType(const GraphViewer& graph_viewer,
                               const NodeUnit& parent_node_unit,
                               gsl::span<const std::string_view> child_op_types,
                               const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                               const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Node& parent_node = parent_node_unit.GetNode();

  // Parent must not produce a graph output.
  // This check ensures we don't try to fuse with nodes that produce graph outputs
  if (graph_viewer.NodeProducesGraphOutput(parent_node)) {
    return nullptr;
  }

  // Child must be of a valid type.
  for (auto edge = parent_node.OutputEdgesBegin(); edge != parent_node.OutputEdgesEnd(); ++edge) {
    const Node& child_node = edge->GetNode();
    if (graph_viewer.GetNode(child_node.Index()) == nullptr) {
      return nullptr;  // Node is not in this GraphViewer
    }
    const std::string& child_type = child_node.OpType();
    bool is_valid_child_type = false;

    for (const auto& valid_op_type : child_op_types) {
      if (valid_op_type == child_type) {
        is_valid_child_type = true;
        break;
      }
    }

    if (!is_valid_child_type) {
      continue;
    }

    const auto child_node_unit_it = node_unit_map.find(&child_node);
    if (child_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const NodeUnit* child_node_unit = child_node_unit_it->second;

    // Check if child node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(child_node_unit) != 0) {
      return nullptr;
    }

    // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
    if (child_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
      return nullptr;
    }

    return child_node_unit;
  }

  return nullptr;
}

const NodeUnit* GetParentOfType(const GraphViewer& graph_viewer,
                                const NodeUnit& child_node_unit,
                                gsl::span<const std::string_view> parent_op_types,
                                const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Node& child_node = child_node_unit.GetNode();

  for (auto edge = child_node.InputEdgesBegin(); edge != child_node.InputEdgesEnd(); ++edge) {
    const Node& parent_node = edge->GetNode();
    if (graph_viewer.GetNode(parent_node.Index()) == nullptr) {
      // Node is not in this GraphViewer
      return nullptr;
    }

    if (graph_viewer.NodeProducesGraphOutput(parent_node)) {
      // Node is producing a graph output
      return nullptr;
    }

    const std::string& parent_type = parent_node.OpType();
    bool is_valid_parent_type = false;

    for (const auto& valid_op_type : parent_op_types) {
      if (valid_op_type == parent_type) {
        is_valid_parent_type = true;
        break;
      }
    }

    if (!is_valid_parent_type) {
      continue;
    }

    const auto parent_node_unit_it = node_unit_map.find(&parent_node);
    if (parent_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const NodeUnit* p_parent_node_unit = parent_node_unit_it->second;

    // Check if parent node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(p_parent_node_unit) != 0) {
      return nullptr;
    }

    // parent must not already be part of a QDQ NodeUnit (i.e., be standalone).
    if (p_parent_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
      return nullptr;
    }

    return p_parent_node_unit;
  }
  return nullptr;
}

const NodeUnit* GetParentOfInput(const GraphViewer& graph_viewer,
                                 const NodeUnit& node_unit,
                                 const NodeUnitIODef& input,
                                 const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                                 const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Node* p_child_node = nullptr;

  for (auto node : node_unit.GetAllNodesInGroup()) {
    for (auto node_input : node->InputDefs()) {
      if (node_input->Name() == input.node_arg.Name()) {
        p_child_node = node;
        break;
      }

      if (p_child_node != nullptr) {
        break;
      }
    }
  }

  if (p_child_node == nullptr) {
    return nullptr;
  }

  const Node& child_node = *p_child_node;

  for (auto edge = child_node.InputEdgesBegin(); edge != child_node.InputEdgesEnd(); ++edge) {
    const Node& parent_node = edge->GetNode();
    if (parent_node.OutputDefs()[0]->Name() != input.node_arg.Name()) {
      continue;
    }

    if (graph_viewer.GetNode(parent_node.Index()) == nullptr) {
      // Node is not in this GraphViewer
      return nullptr;
    }

    if (graph_viewer.NodeProducesGraphOutput(parent_node)) {
      // Node is producing a graph output
      return nullptr;
    }

    const auto parent_node_unit_it = node_unit_map.find(&parent_node);
    if (parent_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const NodeUnit* p_parent_node_unit = parent_node_unit_it->second;

    // Check if parent node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(p_parent_node_unit) != 0) {
      return nullptr;
    }

    // parent must not already be part of a QDQ NodeUnit (i.e., be standalone).
    if (p_parent_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
      return nullptr;
    }

    return p_parent_node_unit;
  }
  return nullptr;
}

}  // namespace qnn
}  // namespace onnxruntime
