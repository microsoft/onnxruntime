// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_node_capability.h"

namespace onnxruntime {
// DnnlDefaultNodeCapability class
//-------------------------------------
DnnlDefaultNodeCapability::DnnlDefaultNodeCapability() {
  inputTypes_.push_back("float");
}

DnnlDefaultNodeCapability::DnnlDefaultNodeCapability(std::vector<std::string> inputTypes) {
  for (std::string s : inputTypes)
    inputTypes_.push_back(s);
}

bool DnnlDefaultNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  return true;
}

bool DnnlDefaultNodeCapability::IsTypeSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  if (!node_inputs.empty() && node_inputs[0]->Type() != nullptr) {
    for (auto inputType : inputTypes_) {
      if (node_inputs[0]->Type()->find(inputType) != std::string::npos) {
        return true;
      }
    }
  }
  return false;
}

// DnnlPoolNodeCapability class
//-------------------------------------
bool DnnlPoolNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsAttributeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

bool DnnlPoolNodeCapability::IsAttributeSupported(const Node* node) const {
  if (node->OpType() == "MaxPool") {
    const NodeAttributes& attributes = node->GetAttributes();
    auto attr = attributes.find("dilations");
    if (attr != attributes.end()) {
      for (int i = 0; i < attr->second().ints_size(); ++i) {
        if (attr->second().ints(i) > 1) {
          return false;
        }
      }
    }
  }
  return true;
}

bool DnnlPoolNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
#ifdef ENABLE_TRAINING
  if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() < 3) {
#else
  if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() <= 3) {
#endif  // ENABLE_TRAINING
    return false;
  }

#ifdef ENABLE_TRAINING
  if (node->OutputDefs().size() > 2)
    return false;
#else
  if (node->OutputDefs().size() > 1)
    return false;
#endif  // ENABLE_TRAINING
  return true;
}

// DnnlBatchNormalizationNodeCapability class
//-------------------------------------
bool DnnlBatchNormalizationNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

bool DnnlBatchNormalizationNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() == 3) {
    return false;
  }
  return true;
}

// DnnlReduceMeanNodeCapability class
//-------------------------------------
bool DnnlReduceMeanNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsAttributeSupported(node)) return false;
  return true;
}

bool DnnlReduceMeanNodeCapability::IsAttributeSupported(const Node* node) const {
  const NodeAttributes& attributes = node->GetAttributes();
  auto attr = attributes.find("keepdims");
  if (attr != attributes.end() && attr->second().i() == 0) {
    return false;
  }
  return true;
}

// DnnlMatMulNodeCapability class
bool DnnlMatMulNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

bool DnnlMatMulNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  if ((node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() >= 2) &&
      (node_inputs[1]->Shape() != nullptr && node_inputs[1]->Shape()->dim_size() >= 2) &&
      (node_inputs[0]->Shape()->dim_size() == node_inputs[1]->Shape()->dim_size())) {
    for (const onnx::TensorShapeProto_Dimension& dim : node_inputs[0]->Shape()->dim()) {
      if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
        return false;
      }
    }
    for (const onnx::TensorShapeProto_Dimension& dim : node_inputs[1]->Shape()->dim()) {
      if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
        return false;
      }
    }
  } else {
    return false;
  }
  return true;
}

}  // namespace onnxruntime
