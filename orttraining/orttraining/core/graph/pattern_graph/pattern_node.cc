// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pattern_graph/pattern_node.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

static std::vector<ONNX_NAMESPACE::TensorProto_DataType>
    all_interger_tensor_types = {ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
                                 ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
                                 ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16,
                                 ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8,
                                 ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64,
                                 ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32,
                                 ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16,
                                 ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8};

static std::vector<ONNX_NAMESPACE::TensorProto_DataType>
    all_float_tensor_types = {ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
                              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE,
                              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16,
                              ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16};

PGraphInputTypes::PGraphInputTypes(TypesCategory category) {
  switch (category) {
    case TypesCategory::AllIntegerTensorTypes: {
      Init(all_interger_tensor_types);
    } break;
    case TypesCategory::AllFloatTensorTypes: {
      Init(all_float_tensor_types);
    } break;
    case TypesCategory::AllIntegerAndFloatTensorTypes: {
      std::vector<ONNX_NAMESPACE::TensorProto_DataType>
          all_integer_and_float_tensor_types(all_interger_tensor_types);
      all_integer_and_float_tensor_types.insert(all_integer_and_float_tensor_types.end(), all_float_tensor_types.begin(), all_float_tensor_types.end());
      Init(all_integer_and_float_tensor_types);
    } break;
    default:
      break;
  }
}

bool PGraphInput::MatchesDataType(const Graph&, const NodeArg& input_arg) const {
  const TypeProto* type_proto = input_arg.TypeAsProto();
  return std::find(allowed_data_types_.begin(), allowed_data_types_.end(),
                   type_proto->tensor_type().elem_type()) != allowed_data_types_.end();
}

bool PGraphInput::MatchesShape(const Graph&, const NodeArg& input_arg) const {
  if (CanBeAnyRank()) {
    return true;
  }
  const TypeProto* type_proto = input_arg.TypeAsProto();
  auto shape = type_proto->tensor_type().shape();
  auto rank = static_cast<int>(shape.dim_size());
  return std::find(allowed_ranks_.begin(), allowed_ranks_.end(), rank) != allowed_ranks_.end();
}

void PGraphNode::CreateNodeName() {
  std::stringstream ss;
  ss << op_type_ << "_";
  for (size_t i = 0; i < input_args_names_.size(); i++) {
    ss << input_args_names_[i] << "_";
  }
  for (size_t i = 0; i < output_args_names_.size(); i++) {
    ss << output_args_names_[i] << "_";
  }
  node_name_ = ss.str();
}

bool PGraphNode::MatchesDomainVersion(const std::string& domain, const int version) const {
  if (domain_version_maps_.empty()) {
    return true;
  }
  auto ret = domain_version_maps_.find(domain);
  if (ret == domain_version_maps_.end()) {
    return false;
  }

  return std::find(ret->second.begin(), ret->second.end(), version) != ret->second.end();
}

}  // namespace training
}  // namespace onnxruntime
