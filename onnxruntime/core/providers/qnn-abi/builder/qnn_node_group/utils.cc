// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"

#include <gsl/gsl>
#include <string_view>
#include <unordered_map>

#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

const OrtNodeUnit* GetOnlyChildOfType(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                      const OrtNodeUnit& parent_node_unit,
                                      gsl::span<const std::string_view> child_op_types,
                                      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Ort::ConstNode parent_node(&parent_node_unit.GetNode());
  std::vector<Ort::ConstValueInfo> outputs = parent_node.GetOutputs();

  // Parent must have a single child and must not produce a graph output.
  if (outputs.size() != 1) {
    return nullptr;
  }
  for (const Ort::ConstValueInfo& output_info : outputs) {
    if (output_info.IsGraphOutput()) {
      return nullptr;
    }
  }

  std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = outputs[0].GetConsumers();
  if (consumers.size() != 1 || consumers[0].node == nullptr) {
    return nullptr;
  }

  const Ort::ConstNode child_node = consumers[0].node;
  const std::string& child_type = child_node.GetOperatorType();
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

  const auto child_node_unit_it = node_unit_map.find(child_node);
  if (child_node_unit_it == node_unit_map.end()) {
    return nullptr;
  }
  const OrtNodeUnit* child_node_unit = child_node_unit_it->second;

  // Check if child node has already been handled. Should not be the case if the calling
  // fusion function has been called in topological order, but check to be safe.
  if (qnn_node_group_map.count(child_node_unit) != 0) {
    return nullptr;
  }

  // child must not already be part of a QDQ NodeUnit (i.e., be standalone).
  if (child_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  return child_node_unit;
}

const OrtNodeUnit* GetParentOfType(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                   const OrtNodeUnit& child_node_unit,
                                   gsl::span<const std::string_view> parent_op_types,
                                   const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                   const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const Ort::ConstNode child_node(&child_node_unit.GetNode());

  for (const Ort::ConstValueInfo& input_info : child_node.GetInputs()) {
    const Ort::ConstNode parent_node = input_info.GetProducerNode().node;
    if (static_cast<const OrtNode*>(parent_node) == nullptr) {
      continue;
    }
    for (const Ort::ConstValueInfo& parent_output_info : parent_node.GetOutputs()) {
      if (parent_output_info.IsGraphOutput()) {
        // Node is producing a graph output
        return nullptr;
      }
    }

    const std::string parent_type = parent_node.GetOperatorType();
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

    const auto parent_node_unit_it = node_unit_map.find(parent_node);
    if (parent_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const OrtNodeUnit* p_parent_node_unit = parent_node_unit_it->second;

    // Check if parent node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(p_parent_node_unit) != 0) {
      return nullptr;
    }

    // parent must not already be part of a QDQ NodeUnit (i.e., be standalone).
    if (p_parent_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode) {
      return nullptr;
    }

    return p_parent_node_unit;
  }
  return nullptr;
}

const OrtNodeUnit* GetParentOfInput(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                    const OrtNodeUnit& node_unit,
                                    const OrtNodeUnitIODef& input,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const OrtNode* p_child_node = nullptr;

  for (const OrtNode* node : node_unit.GetAllNodesInGroup()) {
    for (const Ort::ConstValueInfo& input_info : Ort::ConstNode(node).GetInputs()) {
      if (input_info.GetName() == input.name) {
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

  const Ort::ConstNode child_node(p_child_node);

  for (const Ort::ConstValueInfo& input_info : child_node.GetInputs()) {
    if (input_info.GetName() != input.name) {
      continue;
    }

    const Ort::ConstNode parent_node = input_info.GetProducerNode().node;
    if (static_cast<const OrtNode*>(parent_node) == nullptr) {
      return nullptr;
    }
    for (const Ort::ConstValueInfo& parent_output_info : parent_node.GetOutputs()) {
      if (parent_output_info.IsGraphOutput()) {
        // Node is producing a graph output
        return nullptr;
      }
    }

    const auto parent_node_unit_it = node_unit_map.find(parent_node);
    if (parent_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const OrtNodeUnit* p_parent_node_unit = parent_node_unit_it->second;

    // Check if parent node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(p_parent_node_unit) != 0) {
      return nullptr;
    }

    return p_parent_node_unit;
  }
  return nullptr;
}

const OrtNodeUnit* GetChildOfOutput(const QnnModelWrapper& /*qnn_model_wrapper*/,
                                    const OrtNodeUnit& node_unit,
                                    const OrtNodeUnitIODef& output,
                                    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  const OrtNode* p_parent_node = nullptr;

  for (const OrtNode* node : node_unit.GetAllNodesInGroup()) {
    for (const Ort::ConstValueInfo& output_info : Ort::ConstNode(node).GetOutputs()) {
      if (output_info.GetName() == output.name) {
        p_parent_node = node;
        break;
      }

      if (p_parent_node != nullptr) {
        break;
      }
    }
  }

  if (p_parent_node == nullptr) {
    return nullptr;
  }

  const Ort::ConstNode parent_node(p_parent_node);

  for (const Ort::ConstValueInfo& parent_output_info : parent_node.GetOutputs()) {
    if (parent_output_info.IsGraphOutput()) {
      // Node is producing a graph output
      return nullptr;
    }
  }

  for (const Ort::ConstValueInfo& output_info : parent_node.GetOutputs()) {
    // Check if this is the output we're looking for
    if (output_info.GetName() != output.name) {
      continue;
    }

    std::vector<Ort::ValueInfoConsumerProducerInfo> consumers = output_info.GetConsumers();
    if (consumers.size() != 1 || consumers[0].node == nullptr) {
      return nullptr;
    }

    const Ort::ConstNode child_node = consumers[0].node;
    const auto child_node_unit_it = node_unit_map.find(child_node);
    if (child_node_unit_it == node_unit_map.end()) {
      return nullptr;
    }
    const OrtNodeUnit* p_child_node_unit = child_node_unit_it->second;

    // Check if child node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (qnn_node_group_map.count(p_child_node_unit) != 0) {
      return nullptr;
    }

    return p_child_node_unit;
  }
  return nullptr;
}

}  // namespace qnn
}  // namespace onnxruntime
