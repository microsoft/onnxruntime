/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "hailo_node_capability.h"

namespace onnxruntime {
// HailoDefaultNodeCapability class
//-------------------------------------
HailoDefaultNodeCapability::HailoDefaultNodeCapability(): m_supported_input_types({type_uint8, type_uint16, type_float32}) {}

bool HailoDefaultNodeCapability::Supported(const Node& node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) {
      return false;
  }
  return true;
}

bool HailoDefaultNodeCapability::IsTypeSupported(const Node& node) const {
    auto node_inputs = node.InputDefs();
    if ((node_inputs.size() == 1) && (node_inputs[0]->TypeAsProto() != nullptr)) {
        auto node_datatype = node_inputs[0]->TypeAsProto()->tensor_type().elem_type();
        for (auto inputType : m_supported_input_types) {
            if (inputType == node_datatype) {
                return true;
            }
        }
    }
    return false;
}

// HailoMultiInputNodeCapability
//-------------------------------------
HailoMultiInputNodeCapability::HailoMultiInputNodeCapability() : HailoDefaultNodeCapability() {}

bool HailoMultiInputNodeCapability::IsTypeSupported(const Node& node) const {
    auto node_inputs = node.InputDefs();
    if (node_inputs.empty()) {
        return false;
    }
    for (auto& input : node_inputs) {
        bool input_type_supported = false;
        if (input->TypeAsProto() != nullptr) {
            for (auto& supported_input : m_supported_input_types) {
                if (supported_input == input->TypeAsProto()->tensor_type().elem_type()) {
                    input_type_supported = true;
                    continue;
                }
            }
        }
        if (!input_type_supported) {
            return false;
        }
    }
    return true;
}

}  // namespace onnxruntime
