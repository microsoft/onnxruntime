// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pattern_graph/pattern_node.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace training {

PGraphInputTypes::PGraphInputTypes(TypesCategory category) {
  switch (category) {
    case TypesCategory::AllIntegerTensorTypes: {
      Init({ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8});
    } break;
    case TypesCategory::AllFloatTensorTypes: {
      Init({ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_DOUBLE,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16,
            ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BFLOAT16});
    } break;
    default:
      break;
  }
}

bool PGraphInput::MatchesDataType(const Graph& graph, const NodeArg& input_arg) const {
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (is_constant_) {
    tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  } else if (!graph.GetInitializedTensor(input_arg.Name(), tensor_proto)) {
    return false;
  }
  const auto data_type = tensor_proto->data_type();
  return std::find(allowed_types_.begin(), allowed_types_.end(), data_type) != allowed_types_.end();
}

bool PGraphInput::MatchesShape(const Graph& graph, const NodeArg& input_arg) const {
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (is_constant_) {
    tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
  } else {
    ORT_ENFORCE(graph.GetInitializedTensor(input_arg.Name(), tensor_proto));
  }

  if (t_proto_.dims_size() != tensor_proto->dims_size()) {
    return false;
  }
  auto pdims = t_proto_.dims();
  auto tdims = tensor_proto->dims();
  for (int i = 0; i < t_proto_.dims_size(); i++) {
    if (pdims[i] != tdims[i]) {
      return false;
    }
  }
  return true;
}

void PGraphInput::SetTensorProto(ONNX_NAMESPACE::TensorProto_DataType type) {
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32:
      t_proto_ = ONNX_NAMESPACE::ToTensor<int32_t>(0);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64:
      t_proto_ = ONNX_NAMESPACE::ToTensor<int64_t>(0);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16:
      t_proto_ = ONNX_NAMESPACE::ToTensor<int32_t>(0);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8:
      t_proto_ = ONNX_NAMESPACE::ToTensor<int32_t>(0);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL:
      t_proto_ = ONNX_NAMESPACE::ToTensor<int32_t>(0);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT16:
      t_proto_ = ONNX_NAMESPACE::ToTensor<uint64_t>(0);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT32:
      t_proto_ = ONNX_NAMESPACE::ToTensor<uint64_t>(0);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT64:
      t_proto_ = ONNX_NAMESPACE::ToTensor<uint64_t>(0);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8:
      t_proto_ = ONNX_NAMESPACE::ToTensor<uint64_t>(0);
      break;
    default:
      t_proto_ = ONNX_NAMESPACE::ToTensor(.0, type);
      break;
  }
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

NodeDef PGraphNode::GetNodeDef() const {
  auto IA = [](const std::string& argSuffix, const TypeProto* type_proto = nullptr) {
    return ArgDef(argSuffix, type_proto);
  };

  std::string domain = "";
  int version = 9;
  if (!domain_version_maps_.empty()) {
    auto first_pair = domain_version_maps_.begin();
    domain = first_pair->first;
    version = *first_pair->second.begin();
  }

  std::vector<ArgDef> input_args(input_args_names_.size());
  std::vector<ArgDef> output_args(output_args_names_.size());
  for (size_t i = 0; i < input_args_names_.size(); i++) {
    input_args[i] = IA(input_args_names_[i]);
  }
  for (size_t i = 0; i < output_args_names_.size(); i++) {
    output_args[i] = IA(output_args_names_[i]);
  }
  return NodeDef(OpDef(op_type_, domain, version), input_args, output_args, attributes, GetNodeName());
}

}  // namespace training
}  // namespace onnxruntime