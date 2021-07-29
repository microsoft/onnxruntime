// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_node_capability.h"
#include "dnnl.hpp"

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
  const NodeAttributes& attributes = node->GetAttributes();
  if (node->OpType() == "MaxPool") {
    auto attr = attributes.find("dilations");
    if (attr != attributes.end()) {
      for (int i = 0; i < attr->second().ints_size(); ++i) {
        if (attr->second().ints(i) > 1) {
          return false;
        }
      }
    }
  }
  auto attr = attributes.find("ceil_mode");
  if (attr != attributes.end()) {
    if (attr->second().i() != 0) {
      return false;
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
//-------------------------------------
bool DnnlMatMulNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

bool DnnlMatMulNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();

  // if shape not specified (nullptr), support it (might fail if end up being not valid)
  if (node_inputs[0]->Shape() == nullptr || node_inputs[1]->Shape() == nullptr) {
    return true;
  }
  // if shape not nullptr, need to have at least 2 dims (a matrix)
  if ((node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() >= 2) &&
      (node_inputs[1]->Shape() != nullptr && node_inputs[1]->Shape()->dim_size() >= 2)) {
    for (const auto& dim : node_inputs[0]->Shape()->dim()) {
      if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
        return false;
      }
    }
    for (const auto& dim : node_inputs[1]->Shape()->dim()) {
      if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
        return false;
      }
    }
  } else {
    return false;
  }

  return true;
}

// DnnlMatMulIntegerNodeCapability class
//-------------------------------------
bool DnnlMatMulIntegerNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;

  // if weight is u8, onednn doesn't support
  auto node_inputs = node->InputDefs();
  if (node_inputs[1]->Type()->find("uint8") != std::string::npos) {
    return false;
  }

  // do not support matmulint on gpu
  if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_gpu)) {
    return false;
  }
  return true;
}

bool DnnlMatMulIntegerNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();

  //if shape nullptr, attempt to run it (no gaurantee)
  if (node_inputs[0]->Shape() == nullptr || node_inputs[1]->Shape() == nullptr) {
    return true;
  }

  if ((node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() >= 2) &&
      (node_inputs[1]->Shape() != nullptr && node_inputs[1]->Shape()->dim_size() >= 2)) {
    for (const auto& dim : node_inputs[0]->Shape()->dim()) {
      if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
        return false;
      }
    }
    for (const auto& dim : node_inputs[1]->Shape()->dim()) {
      if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
        return false;
      }
    }
  } else {
    return false;
  }

  //do not support other than single zero point (not per column)
  if (node_inputs.size() > 2) {
    if (node_inputs.size() >= 3 && node_inputs[2] && node_inputs[2]->Exists()) {
      if (node_inputs[2]->Shape() != nullptr && node_inputs[2]->Shape()->dim_size() >= 1) {
        return false;
      }
    }

    if (node_inputs.size() >= 4 && node_inputs[3] && node_inputs[3]->Exists()) {
      if (node_inputs[3]->Shape() != nullptr && node_inputs[3]->Shape()->dim_size() >= 1) {
        return false;
      }
    }
  }

  return true;
}

}  // namespace onnxruntime
