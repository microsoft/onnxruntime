#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"

#include <gsl/gsl>
#include <string_view>
#include <unordered_map>

#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/qnn_node_group.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

const OrtNodeUnit* GetOnlyChildOfType(const QnnModelWrapper& qnn_model_wrapper,
                                      const OrtNodeUnit& parent_node_unit,
                                      gsl::span<const std::string_view> child_op_types,
                                      const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                                      const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map) {
  // const OrtGraph& graph = qnn_model_wrapper.GetOrtGraph();
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtNode& parent_node = parent_node_unit.GetNode();

  // Parent must have a single child (1 output edge) and must not produce a graph output.
  size_t num_outputs = 0;
  ort_api.Node_GetNumOutputs(&parent_node, &num_outputs);
  if (num_outputs != 1) {
    return nullptr;
  }
  std::vector<const OrtValueInfo*> outputs(num_outputs);
  ort_api.Node_GetOutputs(&parent_node, outputs.data(), outputs.size());

  // Check if any of the node's outputs are graph outputs
  const OrtValueInfo* output_info = outputs[0];

  bool is_graph_output = false;
  Status status = output_info->IsGraphOutput(is_graph_output);
  if (status.IsOK() && is_graph_output) {
    return nullptr;
  }

  // Get the consumers of this output
  std::vector<OrtValueInfo::ConsumerInfo> consumer_infos;
  status = output_info->GetConsumerInfos(consumer_infos);
  if (!status.IsOK() || consumer_infos.empty()) {
    return nullptr;
  }

  // We should have exactly one consumer
  if (consumer_infos.size() != 1) {
    return nullptr;
  }

  // Get the child node
  const OrtNode* child_node_ptr = consumer_infos[0].node;
  if (child_node_ptr == nullptr) {
    return nullptr;
  }

  // Child must be of a valid type.
  const std::string& child_type = child_node_ptr->GetOpType();
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

  const auto child_node_unit_it = node_unit_map.find(child_node_ptr);
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

}  // namespace qnn
}  // namespace onnxruntime
