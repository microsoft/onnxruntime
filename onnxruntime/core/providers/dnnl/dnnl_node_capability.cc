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
    constexpr size_t TENSOR_PREFIX_LEN = 7; //Precomputed value of std::string("tensor(").length()
    for (auto inputType : inputTypes_) {
      // Check for string "tensor(data_type)"
      // Exact length check done for 2 reasons to avoid the out_of_range exception from the compare call and
      // to check that the string ends in ')'.  This will prevent false matching "float" data type for "float16" data type.
      // The "+ 1" in the formula is to account for the ending ')' in the string length.
      // check that the node_inputs[0]->Type() string after the prefix "tensor(" matches the inputType.
      if ((TENSOR_PREFIX_LEN + inputType.length() + 1 ) == node_inputs[0]->Type()->length() &&
          node_inputs[0]->Type()->compare(TENSOR_PREFIX_LEN, inputType.length(), inputType) == 0) {
        return true;
      }
      if (node_inputs[0]->Type()->compare(inputType) == 0) {
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
  if (!IsMaxPoolIndicesSupported(node)) return false;
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
  if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() < 3) {
    return false;
  }
  return true;
}

// In Onnx MaxPool can have an Indices output. OneDNN does not support outputing the
// indices. If we are training and the indices are only going to the MaxPoolGrad we
// are using the indices output to pass OneDNN's internal representation of the indices
// We can only claim MaxPools with Indices going to the MaxPoolGrad op. If the indices
// go to any other node we can not support the MaxPool node.
bool DnnlPoolNodeCapability::IsMaxPoolIndicesSupported(const Node* node) const {
#ifdef ENABLE_TRAINING
  if (node->OpType() == "MaxPool") {
    if (node->OutputDefs().size() > 2)
      return false;
    if (node->OutputDefs().size() > 1) {
      bool edge_to_maxpoolgrad_found = false;
      for (auto it = node->OutputEdgesBegin(); it != node->OutputEdgesEnd(); ++it) {
        // if the MaxPool indice output `GetSrcArgIndex() == 1` goes to an OpType other than
        // "MaxPoolGrad" the DNNL Execution providers implementaiton of MaxPool should not be used.
        if (it->GetSrcArgIndex() == 1 && it->GetNode().OpType() != "MaxPoolGrad") {
          return false;
        }
        // if the MaxPool indice output is going to the MaxPoolGrad indice output then it is possible we can
        // use our implementation of MaxPool for training.
        if (it->GetSrcArgIndex() == 1 && it->GetDstArgIndex() == 1 && it->GetNode().OpType() == "MaxPoolGrad") {
          edge_to_maxpoolgrad_found = true;
        }
      }
      return edge_to_maxpoolgrad_found;
    }
  }
  // Check that the input from MaxPool is from a supported node
  if (node->OpType() == "MaxPoolGrad") {
    for (auto it = node->InputNodesBegin(); it != node->InputNodesEnd(); ++it) {
      if (it->OpType() == "MaxPool") {
        // the code `&(*it)` will convert the `NodeConstIterator` to a `const Node*`
        return (IsMaxPoolIndicesSupported(&(*it)));
      }
    }
  }
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
  if (node_inputs[1]->Shape() != nullptr && node_inputs[1]->Shape()->dim_size() != 1) {
    return false;
  }
  if (node_inputs[2]->Shape() != nullptr && node_inputs[2]->Shape()->dim_size() != 1) {
    return false;
  }
  if (node_inputs[3]->Shape() != nullptr && node_inputs[3]->Shape()->dim_size() != 1) {
    return false;
  }
  if (node_inputs[4]->Shape() != nullptr && node_inputs[4]->Shape()->dim_size() != 1) {
    return false;
  }
  return true;
}

// DnnlReduceMeanNodeCapability class
//-------------------------------------
bool DnnlReduceMeanNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsAttributeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
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

bool DnnlReduceMeanNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() == 0) {
    return false;
  }
  return true;
}

// DnnlSoftmaxNodeCapability class
bool DnnlSoftmaxNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsAttributeSupported(node)) return false;
  return true;
}

//DNNL Softmax supports opset version of 13 and above, or only axis value of 2 for opset version < 13
bool DnnlSoftmaxNodeCapability::IsAttributeSupported(const Node* node) const {
  const NodeAttributes& attributes = node->GetAttributes();
  auto opset = node->SinceVersion();
  auto attr = attributes.find("axis");
  int64_t axis = 1;
  if (attr != attributes.end() && attr->second().i() == 0) {
    axis = attr->second().i();
  }
  if (opset < 13 && axis != 2)
    return false;
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

// DnnlSumNodeCapability class
//-------------------------------------
bool DnnlSumNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

// OneDNN version of Sum does not support Numpy style broadcasting.
// If the dimentions of all inputs do not match return false
bool DnnlSumNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  // find first non-null shape
  const ONNX_NAMESPACE::TensorShapeProto* data_0_shape = nullptr;
  for (auto input : node_inputs) {
    data_0_shape = input->Shape();
    if (data_0_shape != nullptr) {
      break;
    }
  }
  // if all input shapes are null accept the 'Sum'
  if (data_0_shape == nullptr) {
    return true;
  }
  // assume first shape was not nullptr and start at index 1
  // even if fist shape was nullptr this will just compare a shape with itself
  for (size_t i = 1; i < node_inputs.size(); ++i) {
    if (node_inputs[i]->Shape() == nullptr) {
      continue;
    }
    if (data_0_shape->dim_size() != node_inputs[i]->Shape()->dim_size()) {
      return false;
    }
    for (int j = 0; j < data_0_shape->dim_size(); ++j) {
      if (data_0_shape->dim(j).dim_value() != node_inputs[i]->Shape()->dim(j).dim_value()) {
        return false;
      }
    }
  }
  return true;
}

bool DnnlBinaryNodeCapability::Supported(const Node* node) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  //gpu broadcast for source 0 not supported
  if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_gpu)) {
    return false;
  }
  return true;
}

bool DnnlBinaryNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  if (node_inputs[0]->Shape() == nullptr || node_inputs[1]->Shape() == nullptr) {
    return true;
  }

  // DNNL binary ops Add and Mul do not work if both inputs are scalar values
  bool src0_is_scalar = false;
  bool src1_is_scalar = false;
  if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() == 0) {
    src0_is_scalar = true;
  }
  if (node_inputs[1]->Shape() != nullptr && node_inputs[1]->Shape()->dim_size() == 0) {
    src1_is_scalar = true;
  }
  if (src0_is_scalar && src1_is_scalar) {
    return false;
  }
  return true;
}

}  // namespace onnxruntime
