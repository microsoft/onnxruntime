// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_node_capability.h"
#include "dnnl.hpp"
#include "subgraph/dnnl_util.h"

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

namespace onnxruntime {
// DnnlDefaultNodeCapability class
//-------------------------------------
DnnlDefaultNodeCapability::DnnlDefaultNodeCapability() {
  inputTypes_.push_back(type_float32);
}

DnnlDefaultNodeCapability::DnnlDefaultNodeCapability(std::vector<ORT_DataType> inputTypes) {
  for (ORT_DataType datatype : inputTypes)
    inputTypes_.push_back(datatype);
}

bool DnnlDefaultNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  return true;
}

bool DnnlDefaultNodeCapability::IsTypeSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  if (!node_inputs.empty() && node_inputs[0]->TypeAsProto() != nullptr) {
    auto node_datatype = node_inputs[0]->TypeAsProto()->tensor_type().elem_type();
    for (auto inputType : inputTypes_) {
      if (inputType == node_datatype) {
        if (inputType == type_bfloat16) {
          return ort_dnnl::dnnl_util::IsBF16Supported();
        }
        return true;
      }
    }
  }
  return false;
}

// DnnlDefaultMultiInputNodeCapability
//-------------------------------------
DnnlDefaultMultiInputNodeCapability::DnnlDefaultMultiInputNodeCapability(std::vector<std::unordered_set<ORT_DataType>> inputTypes) {
  for (auto outer : inputTypes) {
    std::unordered_set<ORT_DataType> per_tensor_types;
    for (auto inputType : outer) {
      per_tensor_types.insert(inputType);
    }
    inputTypes_.push_back(per_tensor_types);
  }
}

bool DnnlDefaultMultiInputNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  return true;
}

bool DnnlDefaultMultiInputNodeCapability::IsTypeSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  bool all_inputs_supported = true;
  if (!node_inputs.empty()) {
    std::vector<bool> input_supported(node_inputs.size(), false);
    for (size_t i = 0; i < node_inputs.size(); ++i) {
      if (node_inputs[i]->TypeAsProto() != nullptr) {
        ORT_DataType node_datatype = static_cast<ORT_DataType>(node_inputs[i]->TypeAsProto()->tensor_type().elem_type());
        input_supported[i] = (inputTypes_[i].find(node_datatype) != inputTypes_[i].end());
        if (node_datatype == type_bfloat16 && !ort_dnnl::dnnl_util::IsBF16Supported()) {
          input_supported[i] = false;
        }
      }
    }
    // Walk the input_supported make sure they are all supported
    all_inputs_supported = true;
    for (bool input_support : input_supported) {
      all_inputs_supported = all_inputs_supported && input_support;
      if (!all_inputs_supported) break;
    }
  }
  return all_inputs_supported;
}

// DnnlDefaultOptionalMultiInputNodeCapability
//-------------------------------------
DnnlDefaultOptionalMultiInputNodeCapability::DnnlDefaultOptionalMultiInputNodeCapability(rule_map op_rules) {
  for (const auto& rule : op_rules) {
    // Get the rules into our map and count the number of mandatory inputs
    op_rules_[rule.first] = rule.second;

    // Check if the input is mandatory
    if (rule.second.first) {
      ++num_mandatory;
    }
  }
}

bool DnnlDefaultOptionalMultiInputNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  return true;
}

unsigned int DnnlDefaultOptionalMultiInputNodeCapability::GetNumMandatoryInputs() {
  return num_mandatory;
}

bool DnnlDefaultOptionalMultiInputNodeCapability::IsTypeSupported(const Node* node) const {
  // Get the node list its size
  auto node_inputs = node->InputDefs();
  auto num_nodes = node_inputs.size();

  // We need to make sure that we have at least the mandatory inputs
  if (num_mandatory <= num_nodes) {
    // Iterate over each entry to check if the input is available, optional and supported
    for (const auto& rule : op_rules_) {
      if (rule.first < num_nodes) {
        // Get the target node
        auto node_input = node_inputs[rule.first];

        // If we found the node we want and it is valid
        if (node_input->TypeAsProto() != nullptr) {
          // Get its datatype
          ORT_DataType node_datatype = static_cast<ORT_DataType>(node_input->TypeAsProto()->tensor_type().elem_type());

          // If the datatype is NOT on the supported list, we dont support the op
          if (rule.second.second.find(node_datatype) == rule.second.second.end()) {
            return false;
          }
        }
      }
    }
    // If the node doesn't have the minimum mandatory inputs
  } else {
    return false;
  }

  return true;
}

// DnnlPoolNodeCapability class
//-------------------------------------
bool DnnlPoolNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
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
bool DnnlBatchNormalizationNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
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

// DnnlReduceNodeCapability class
//-------------------------------------
bool DnnlReduceNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  // These reduction operators use elementwise ops so elementwise operators must also be supported.
  if (node->OpType() == "ReduceLogSum" ||
      node->OpType() == "ReduceLogSumExp" ||
      node->OpType() == "ReduceSumSquare") {
    if (!_eltwise.Supported(node, graph_viewer)) return false;
  }
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

bool DnnlReduceNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  if (node_inputs[0]->Shape() != nullptr && node_inputs[0]->Shape()->dim_size() == 0) {
    LOGS_DEFAULT(INFO) << "Reduction op not supported because input data is a scalar\n";
    return false;
  }
  return true;
}

// DnnlSoftmaxNodeCapability class
//-------------------------------------
bool DnnlSoftmaxNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsAttributeSupported(node)) return false;
  return true;
}

// DNNL Softmax supports opset version of 13 and above, or only axis value of 2 for opset version < 13
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
bool DnnlMatMulNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

// DnnlLRNNodeCapability class
//-------------------------------------
bool DnnlLRNNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
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
bool DnnlMatMulIntegerNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node, graph_viewer)) return false;

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

// assume weight zp is s8 since u8 will get rejected and weight zp matches weight data type
bool DnnlMatMulIntegerNodeCapability::IsWeightZeroPointConstantZero(const NodeArg* node_arg, const GraphViewer& graph_viewer) const {
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  // if node_arg is not initializer
  if (!graph_viewer.GetInitializedTensor(node_arg->Name(), tensor_proto)) {
    return false;
  }
  // if node_arg is not initializer
  if (tensor_proto == nullptr) {
    return false;
  }

  const auto& dims = tensor_proto->dims();
  auto dim_size = tensor_proto->dims_size();
  int num_elements = 1;
  for (int i = 0; i < dim_size; i++) {
    num_elements *= int(dims[i]);
  }

  // check if weight zp is all zeros
  bool all_zero = true;
  std::vector<int8_t> unpacked_tensor;
  // delibrately make a vector of 1s instead of 0s
  unpacked_tensor.resize(num_elements, 1);
  ORT_THROW_IF_ERROR(onnxruntime::utils::UnpackTensor(*tensor_proto, tensor_proto->has_raw_data() ? tensor_proto->raw_data().data() : nullptr, tensor_proto->has_raw_data() ? tensor_proto->raw_data().size() : 0, reinterpret_cast<int8_t*>(unpacked_tensor.data()), num_elements));

  // check if the initializer zero point contains all zeros
  for (const auto& val : unpacked_tensor) {
    if (val != 0) {
      all_zero = false;
      break;
    }
  }

  // if initializer zero point is not all zeros, reject it
  // TODO: if initializer zero point is a vector of a unique value, we can still compute it
  if (!all_zero) {
    return false;
  }
  return true;
}

bool DnnlMatMulIntegerNodeCapability::IsDimensionSupported(const Node* node, const GraphViewer& graph_viewer) const {
  auto node_inputs = node->InputDefs();

  // do not support other than single zero point (not per column)
  if (node_inputs.size() > 2) {
    if (node_inputs.size() >= 3 && node_inputs[2] && node_inputs[2]->Exists()) {
      if (node_inputs[2]->Shape() != nullptr && node_inputs[2]->Shape()->dim_size() >= 1) {
        return false;
      }
    }

    if (node_inputs.size() >= 4 && node_inputs[3] && node_inputs[3]->Exists()) {
      if (node_inputs[3]->Shape() != nullptr) {
        auto dim_size = node_inputs[3]->Shape()->dim_size();
        if (dim_size >= 1) {
          // non scalar zero point, but if the weight zp is constant of 0s, still accept it and let the fusion rule remove the zero point
          if (!IsWeightZeroPointConstantZero(node_inputs[3], graph_viewer)) {
            return false;
          }
        }
      }
    }
  }

  // if shape nullptr, not enough information to reject it. attempt to run it (no gaurantee)
  if (node_inputs[0]->Shape() == nullptr || node_inputs[1]->Shape() == nullptr) {
    return true;
  }
  // if matmul src and weight have shape but have dim value of 0, reject it
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

// DnnlSumNodeCapability class
//-------------------------------------
bool DnnlSumNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
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

// DnnlBinaryNodeCapability class
//-------------------------------------
bool DnnlBinaryNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsBF16Supported(node)) return false;
  return true;
}

bool DnnlBinaryNodeCapability::IsBF16Supported(const Node* node) const {
  // This is special caseing the Binary operators Less, LessOrEqual, Greator, GreatorOrEqual, and Equal
  // The listed operators are known to not work on GPU and we should indicate lack of GPU support.
  auto node_inputs = node->InputDefs();
  if (!node_inputs.empty() && node_inputs[0]->TypeAsProto() != nullptr) {
    auto node_datatype = node_inputs[0]->TypeAsProto()->tensor_type().elem_type();
    if (type_bfloat16 == node_datatype) {
      if (node->OpType() == "Less" ||
          node->OpType() == "Greator" ||
          std::string::npos != node->OpType().find("Equal")) {
        if (ort_dnnl::dnnl_util::IsGPURuntimeAvalible()) {
          return false;
        }
      }
      return true;
    }
  }
  return true;
}

// DnnlElementwiseNodeCapability class
//-------------------------------------
bool DnnlElementwiseCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

bool DnnlElementwiseCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  if (node_inputs[0]->Shape() == nullptr) {
    return true;
  }

  // reject gpu elmentwise op with 5 dims or more
  if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_gpu)) {
    if (node_inputs[0]->Shape()->dim_size() > 5) {
      return false;
    }
  }

  return true;
}

bool IsScalar(const NodeArg* node_arg) {
  if (node_arg->Shape()->dim_size() == 0) {
    return true;
  }
  for (int j = 0; j < node_arg->Shape()->dim_size(); ++j) {
    if (node_arg->Shape()->dim(j).dim_value() != 1) {
      return false;
    }
  }
  return true;
}

// DnnlPowNodeCapability class
//-------------------------------------
bool DnnlPowNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node, graph_viewer)) return false;
  return true;
}

bool DnnlPowNodeCapability::IsDimensionSupported(const Node* node, const GraphViewer& graph_viewer) const {
  auto node_inputs = node->InputDefs();
  if (node_inputs[0]->Shape() == nullptr) {
    return true;
  }

  // We are limited to only one dimension tensors if the shape of the Pow op is unknown don't claim support.
  if (node_inputs[0]->Shape() == nullptr) {
    return false;
  }

  if (node_inputs[1]->Shape() != nullptr) {
    // At this time OneDNN can only set the exponent at primitive creation time. This means the exponent must be a
    // ConstantInitializer or the Pow function will fail when running the graph.
    if (!graph_viewer.IsConstantInitializer(node_inputs[1]->Name(), true)) {
      return false;
    }
    if (!IsScalar(node_inputs[1])) {
      return false;
    }
  }

  return true;
}

// DnnlGemmNodeCapability class
//-------------------------------------
bool DnnlGemmNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  if (!_matmul.Supported(node, graph_viewer)) return false;
  if (!_binary.Supported(node, graph_viewer)) return false;
  return true;
}

// DnnlReshapeNodeCapability class
//-------------------------------------
bool DnnlReshapeNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}
bool DnnlReshapeNodeCapability::IsDimensionSupported(const Node* node) const {
  auto node_inputs = node->InputDefs();
  // We can not reshape a one dimentional tensor to a scalar output
  if (node_inputs[1]->Shape() != nullptr &&
      node_inputs[1]->Shape()->dim_size() == 1 &&
      node_inputs[1]->Shape()->dim(0).dim_value() == 0) {
    return false;
  }

  return true;
}

// DnnlDynamicQuantizeLinearNodeCapability class
// reserve for future capability change
//-------------------------------------
bool DnnlDynamicQuantizeLinearNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  return true;
}

// DnnlSqueezeCapability class
//-------------------------------------
bool DnnlSqueezeNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node, graph_viewer)) return false;
  return true;
}
bool DnnlSqueezeNodeCapability::IsDimensionSupported(const Node* node, const GraphViewer& graph_viewer) const {
  // we don't support scalar output
  auto node_out = node->OutputDefs()[0];
  if (node_out->Exists() &&
      node_out->Shape() != nullptr &&
      node_out->Shape()->dim_size() == 0) {
    return false;
  }

  // Before opset version 13 the axis comes from an attribute. After opset version
  // 13 we must check that the optional axis (input[1]) is a ConstantInitializer because
  // we only handle the axis at compile time. If it changes at runtime we can not support
  // the operator
  auto opset = node->SinceVersion();
  auto node_inputs = node->InputDefs();
  if (opset >= 13 && node_inputs.size() > 1 && node_inputs[1]->Shape() != nullptr) {
    if (!graph_viewer.IsConstantInitializer(node_inputs[1]->Name(), true)) {
      return false;
    }
  }
  return true;
}

bool DnnlErfNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsErfPartOfGelu(node, graph_viewer)) return false;
  return true;
}

bool DnnlErfNodeCapability::IsInitilizedWithExpectedValue(const GraphViewer& graph_viewer, const NodeArg* node_arg, float expected_value) const {
  // TypeAsProto()->tensor_type().elem_type()
  if ((ORT_DataType)node_arg->TypeAsProto()->tensor_type().elem_type() == type_float32) {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    graph_viewer.GetInitializedTensor(node_arg->Name(), tensor_proto);
    const float* val = reinterpret_cast<const float*>(tensor_proto->raw_data().data());

    // Check for NaN and Inf
    if (std::isnan(val[0]) || std::isinf(val[0])) {
      if (std::isinf(val[0]) && std::isinf(expected_value) && (std::signbit(val[0]) == std::signbit(expected_value))) {
        return true;
      }
      return false;
    }

    const float atol = 1e-8f;
    const float rtol = 1e-5f;
    float diff = std::abs(val[0] - expected_value);
    if (diff > (atol + rtol * std::abs(expected_value))) {
      return false;
    }
    return true;
  }
  return false;
}

const Node* DnnlErfNodeCapability::FirstParentByType(const Node& node, const std::string& parent_type) const {
  for (auto it = node.InputNodesBegin(); it != node.InputNodesEnd(); ++it) {
    if ((*it).OpType().compare(parent_type) == 0) {
      return &(*it);
    }
  }
  return nullptr;
}

bool DnnlErfNodeCapability::IsNodeFusable(const Node* node, const GraphViewer& graph_viewer) const {
  if (nullptr == node) {
    return false;
  }
  if (1 != node->GetOutputEdgesCount() &&
      1 != node->OutputDefs().size()) {
    return false;
  }

  // Check if the NodeArg outputs to the graph
  if (std::find(graph_viewer.GetOutputs().begin(), graph_viewer.GetOutputs().end(), node->OutputDefs()[0]) != graph_viewer.GetOutputs().end()) {
    return false;
  }
  return true;
}

/*
OneDNN only suports Erf if it is part of Gelu.  Gelu is only possible thanks to fusion.

This code only runs when and Erf node is detected. So we check to see if the Erf is is part
a Gelu starting from the Erf.  This means checking the inputs befor and after Erf.  The
following pattern and some code were directly referenced from code\optimizer\gelu_fusion.cc

 Subgraphs like the following fuse into Gelu.
 Subgraph pattern 1:
               +-------Mul(0.5)---------------------+
               |                                    |
               |                                    v
            [root] --> Div -----> Erf  --> Add --> Mul ==>
                      (B=1.4142...)        (1)

  Subgraph pattern 2:
               +------------------------------------+
               |                                    |
               |                                    v
            [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
                      (B=1.4142...)        (1)            (0.5)

   After Fusion:
            [root]--> Gelu ==>
*/
bool DnnlErfNodeCapability::IsErfPartOfGelu(const Node* node, const GraphViewer& graph_viewer) const {
  // if DNNL fusion is not enabled then Erf is not supported.
  const std::string fusion_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_ENABLE_FUSION");
  if (!fusion_env.empty() && std::stoi(fusion_env) == 0) {
    return false;
  }
  if (node->InputDefs().size() != 1) {
    return false;
  }
  if (!IsNodeFusable(node, graph_viewer)) {
    return false;
  }

  const Node* div_node = FirstParentByType(*node, "Div");

  if (!IsNodeFusable(div_node, graph_viewer)) {
    return false;
  }
  if (!_binary.Supported(div_node, graph_viewer)) {
    return false;
  }

  auto divisor = div_node->InputDefs()[1];
  if (divisor->Shape() != nullptr) {
    if (!graph_viewer.IsConstantInitializer(divisor->Name(), true)) {
      return false;
    }
    if (!IsScalar(divisor)) {
      return false;
    }
    // Some Bert models uses this approximation of SQRT2 in the Gelu function
    float approximated_sqrt_two = 1.4142099618911743f;
    if (!IsInitilizedWithExpectedValue(graph_viewer, divisor, approximated_sqrt_two) &&
        !IsInitilizedWithExpectedValue(graph_viewer, divisor, static_cast<float>(M_SQRT2))) {
      return false;
    }
  }

  std::vector<const Node*> add_nodes;
  for (auto i = node->OutputNodesBegin(); i != node->OutputNodesEnd(); ++i) {
    add_nodes.push_back(&(*i));
  }
  if (add_nodes.size() != 1 && add_nodes[0]->OpType() != "Add") {
    return false;
  }
  if (!_binary.Supported(add_nodes[0], graph_viewer)) {
    return false;
  }
  if (!IsNodeFusable(add_nodes[0], graph_viewer)) {
    return false;
  }
  // check that the Add Op node_arg is 1.0f
  bool is_add_input0 = node->OutputDefs()[0]->Name() == add_nodes[0]->InputDefs()[0]->Name();
  auto add_val = add_nodes[0]->InputDefs()[is_add_input0 ? 1 : 0];
  if (add_val->Shape() != nullptr) {
    if (!graph_viewer.IsConstantInitializer(add_val->Name(), true)) {
      return false;
    }
    if (!IsScalar(add_val)) {
      return false;
    }
    if (!IsInitilizedWithExpectedValue(graph_viewer, add_val, 1.0f)) {
      return false;
    }
  }

  std::vector<const Node*> mul1_nodes;
  for (auto i = add_nodes[0]->OutputNodesBegin(); i != add_nodes[0]->OutputNodesEnd(); ++i) {
    mul1_nodes.push_back(&(*i));
  }
  if (mul1_nodes.size() != 1 && mul1_nodes[0]->OpType() != "Mul") {
    return false;
  }
  if (!_binary.Supported(mul1_nodes[0], graph_viewer)) {
    return false;
  }

  // Subgraph pattern 1
  // Don't return if subgraph pattern 1 is not found instead we dropdown and look for pattern 2
  {
    auto mul2_node = FirstParentByType(*mul1_nodes[0], "Mul");
    if (mul2_node != nullptr) {
      bool is_mul2_input0 = div_node->InputDefs()[0]->Name() == mul2_node->InputDefs()[0]->Name();
      bool is_mul2_input1 = div_node->InputDefs()[0]->Name() == mul2_node->InputDefs()[1]->Name();
      if (is_mul2_input0 ^ is_mul2_input1) {
        auto mul2_val = mul2_node->InputDefs()[is_mul2_input0 ? 1 : 0];
        if (mul2_val->Shape() != nullptr) {
          if (IsScalar(mul2_val) &&
              IsInitilizedWithExpectedValue(graph_viewer, mul2_val, 0.5f) &&
              IsNodeFusable(mul2_node, graph_viewer)) {
            return true;
          }
        }
      }
    }
  }
  // Subgraph pattern 2
  {
    if (!IsNodeFusable(mul1_nodes[0], graph_viewer)) {
      return false;
    }
    std::vector<const Node*> mul2_nodes;
    for (auto i = mul1_nodes[0]->OutputNodesBegin(); i != mul1_nodes[0]->OutputNodesEnd(); ++i) {
      mul2_nodes.push_back(&(*i));
    }

    if (mul2_nodes.size() != 1 && mul2_nodes[0]->OpType() != "Mul") {
      return false;
    }

    // Check that Mul Op node_arg is 0.5f
    bool is_mul2_input0 = mul1_nodes[0]->OutputDefs()[0]->Name() == mul2_nodes[0]->InputDefs()[0]->Name();
    auto mul2_val = mul2_nodes[0]->InputDefs()[is_mul2_input0 ? 1 : 0];
    if (mul2_val->Shape() != nullptr) {
      if (!graph_viewer.IsConstantInitializer(mul2_val->Name(), true)) {
        return false;
      }
      if (!IsScalar(mul2_val)) {
        return false;
      }
      if (!IsInitilizedWithExpectedValue(graph_viewer, mul2_val, 0.5f)) {
        return false;
      }
    }
  }
  return true;
}

// DnnlQAttentionNodeCapability class
//-------------------------------------
bool DnnlQAttentionNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  auto node_inputs = node->InputDefs();
  auto node_outputs = node->OutputDefs();
  // if have 9th input (past state) and 9th input Exists
  if (node_inputs.size() == 9 && node_inputs[8]->Exists()) {
    return false;
  }
  // if have 2nd output (present state) and 2nd output Exists, return false
  if (node_outputs.size() == 2 && node_outputs[1]->Exists()) {
    return false;
  }
  // qattention doesn't support unidriectional
  const NodeAttributes& attributes = node->GetAttributes();
  auto attr = attributes.find("unidirectional");
  if (attr != attributes.end()) {
    if (attr->second().i() == 1) {
      return false;
    }
  }
  // only support scalar input scale and weight scale
  if (!IsScalar(node_inputs[3]) || !IsScalar(node_inputs[4])) {
    return false;
  }
  // only support 2D raw mask
  if (node_inputs.size() >= 6 && node_inputs[5]->Exists() && node_inputs[5]->Shape() != nullptr && node_inputs[5]->Shape()->dim_size() != 2) {
    return false;
  }
  // only support scalar input zero point
  if (node_inputs.size() >= 7 && node_inputs[6]->Exists() && !IsScalar(node_inputs[6])) {
    return false;
  }
  // only support scalar weight zero point
  if (node_inputs.size() >= 8 && node_inputs[7]->Exists() && !IsScalar(node_inputs[7])) {
    return false;
  }

  return true;
}

bool DnnlQAttentionNodeCapability::IsDimensionSupported(const Node* node) const {
  ORT_UNUSED_PARAMETER(node);
  return true;
}

// DnnlCastNodeCapability class
//-------------------------------------
DnnlCastNodeCapability::DnnlCastNodeCapability(std::vector<ORT_DataType> validTypes)
    : DnnlDefaultNodeCapability(validTypes) {
  for (ORT_DataType datatype : validTypes)
    validTypes_.push_back(datatype);
}

bool DnnlCastNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node) || !IsCastSupported(node)) return false;
  return true;
}

bool DnnlCastNodeCapability::IsCastSupported(const Node* node) const {
  // Get input and attributes
  const NodeAttributes& node_attr = node->GetAttributes();
  auto node_input = node->InputDefs();
  auto attr_to = node_attr.find("to");

  // If we have valid results
  if (!node_input.empty() &&
      node_input[0]->TypeAsProto() != nullptr &&
      attr_to != node_attr.end()) {
    // Get the input and cast target type
    auto input_type = node_input[0]->TypeAsProto()->tensor_type().elem_type();
    auto cast_type = attr_to->second().i();

    // Some FP16 operations are not supported yet on CPU
#if defined(DNNL_CPU_RUNTIME)
    // From uint8 and int8 => To Float16
    if ((input_type == type_uint8 ||
         input_type == type_int8) &&
        cast_type == type_float16) {
      return false;
    }
    // From uint32 => To Float16 and BFloat16
    if (input_type == type_int32 &&
        (cast_type == type_float16 ||
         cast_type == type_bfloat16)) {
      return false;
    }
    // From Float16 => To int(uint8, int8 and int32) and BFloat16
    if (input_type == type_float16 &&
        (cast_type == type_uint8 ||
         cast_type == type_int8 ||
         cast_type == type_int32 ||
         cast_type == type_bfloat16)) {
      return false;
    }
    // From BFloat16 => To int32 and BFloat16
    if (input_type == type_bfloat16 &&
        (cast_type == type_int32 ||
         cast_type == type_float16)) {
      return false;
    }
#endif  // defined(DNNL_CPU_RUNTIME)

    // Check if the cast type is supported
    for (auto validType : validTypes_) {
      if (validType == cast_type) {
        return true;
      }
    }
  }

  return false;
}

bool DnnlDequantizeLinearNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  return true;
}

bool DnnlLayerNormalizationNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) return false;
  if (!IsTrainingSupported(node)) return false;
  if (!IsDimensionSupported(node)) return false;
  return true;
}

bool DnnlLayerNormalizationNodeCapability::IsAxisSupported(const Node* node) const {
  // At the moment of implementation OneDNN does not support broadcasting
  // on LayerNorm so we can only accept the op when the normalization
  // is done on the last dim
  const NodeAttributes& attr = node->GetAttributes();
  auto axis = attr.find("axis");

  if (axis != attr.end()) {
    // Get the norm dim
    auto norm_dim = axis->second().i();
    // Get the input dims
    auto input_dims = node->InputDefs()[0]->Shape()->dim_size();

    // If the axis is not the last dim, it's not supported
    if ((norm_dim != -1) && (norm_dim != (input_dims - 1))) {
      return false;
    }
  }
  // If no axis is provided or the conditions above were not met,
  // we normalize on the last dim
  return true;
}

bool DnnlLayerNormalizationNodeCapability::IsTrainingSupported(const Node* node) const {
  // Get the output defs
  auto num_outputs = node->OutputDefs().size();
  // Training support is in progress, for the moment we don't support multiple outputs
  if (num_outputs > 1) {
    return false;
  } else {
    return true;
  }
}
bool DnnlLayerNormalizationNodeCapability::IsDimensionSupported(const Node* node) const {
  // Get the input shape
  auto input_shape = node->InputDefs()[0]->Shape();

  if (input_shape != nullptr) {
    auto input_dims = input_shape->dim_size();
    // OneDNN is supports from 2D to 5D tensors
    if ((input_dims <= 5) && (input_dims >= 2)) {
      return true;

      // If the tensor is 1D or >6D then we dont support it
    } else {
      return false;
    }
  }
  // If we dont have shape info, accept the data and catch the error on the implementation
  return true;
}

// DnnlConcatNodeCapability class
//-------------------------------------

bool DnnlConcatNodeCapability::Supported(const Node* node, const GraphViewer& graph_viewer) const {
  ORT_UNUSED_PARAMETER(graph_viewer);
  if (!IsTypeSupported(node)) {
    return false;
  }
  if (!AreAllInputsOfSameType(node)) {
    return false;
  }
  if (!AreAxisAndDimensionsSupported(node)) {
    return false;
  }
  return true;
}

bool DnnlConcatNodeCapability::AreAllInputsOfSameType(const Node* node) const {
  // All inputs must be of the same type
  const auto node_inputs = node->InputDefs();
  if (!node_inputs.empty() && node_inputs[0]->TypeAsProto() != nullptr) {
    const auto ref_datatype = node_inputs[0]->TypeAsProto()->tensor_type().elem_type();
    // Ensure that other inputs have the same datatype as the first
    for (size_t i = 1; i < node_inputs.size(); ++i) {
      if (node_inputs[i]->TypeAsProto() != nullptr &&
          node_inputs[i]->TypeAsProto()->tensor_type().elem_type() != ref_datatype) {
        return false;
      }
    }
  }
  return true;
}

bool DnnlConcatNodeCapability::AreAxisAndDimensionsSupported(const Node* node) const {
  auto& attributes = node->GetAttributes();
  auto axis_attr = attributes.find("axis");
  if (axis_attr != attributes.end()) {
    if (axis_attr->second().type() != ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
      // Axis must be an integer
      return false;
    }
  } else {
    // Axis is required
    return false;
  }

  int64_t signed_axis = axis_attr->second().i();

  // Veriy input dimensions
  const auto& node_inputs = node->InputDefs();
  auto ref_input_it = std::find_if(node_inputs.begin(), node_inputs.end(),
                                   [](const auto input) {
                                     return input->Shape() != nullptr;
                                   });
  if (ref_input_it != node_inputs.end()) {
    const auto ref_input = *ref_input_it;
    const auto ref_input_shape = ref_input->Shape();
    const auto ref_input_rank = ref_input_shape->dim_size();

    // DNNL only supports tensors of 6 or fewer dimensions on GPU
    if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_gpu)) {
      if (ref_input_rank > 6) {
        return false;
      }
    }

    // Verify that input tensors match shapes except on the axis dimension.
    const auto adjusted_axis = (signed_axis >= 0) ? signed_axis : signed_axis + ref_input_rank;
    for (auto input : node_inputs) {
      if (input == ref_input) {
        continue;
      }

      const auto input_shape = input->Shape();
      if (input_shape == nullptr) {
        continue;
      }
      if (input_shape->dim_size() != ref_input_rank) {
        // Rank doesn't match reference input.
        return false;
      }
      for (int d = 0; d < ref_input_rank; ++d) {
        if (d == adjusted_axis) {
          continue;
        }
        const auto& ref_input_dim = ref_input_shape->dim(d);
        const auto& input_dim = input_shape->dim(d);
        if (ref_input_dim.has_dim_value() && input_dim.has_dim_value()) {
          if (ref_input_dim.dim_value() != input_dim.dim_value()) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

}  // namespace onnxruntime
