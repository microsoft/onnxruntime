// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/node_arg.h"
#include "./vai_assert.h"
#include "core/providers/shared_library/provider_api.h"

#include "./tensor_proto.h"

namespace vaip {
bool node_arg_is_constant(const Graph& graph, const NodeArg& node_arg) {
  assert(node_arg.Exists());
  assert(!node_arg.Name().empty());
  return graph.GetConstantInitializer(node_arg.Name(), true) != nullptr;
}
vaip_core::DllSafe<std::vector<int64_t>> node_arg_get_shape_i64(const NodeArg& node_arg) {
  auto shape = node_arg.Shape();
  if (nullptr == shape) return vaip_core::DllSafe<std::vector<int64_t>>();
  auto shape_vector = std::vector<int64_t>();
  shape_vector.reserve(shape->dim_size());
  for (auto i = 0; i < shape->dim_size(); ++i) {
    auto& dim = shape->dim(i);
    shape_vector.push_back(dim.has_dim_value() ? dim.dim_value() : (int64_t)-1);
  }
  return vaip_core::DllSafe(shape_vector);
}
void node_arg_set_shape_i64(const NodeArg& node_arg, const std::vector<int64_t>& shape) {
  auto shape_proto = const_cast<ONNX_NAMESPACE::TensorShapeProto*>(node_arg.Shape());
  assert(shape_proto != nullptr);
  assert(shape.size() == static_cast<size_t>(shape_proto->dim_size()));
  auto rank = shape_proto->dim_size();
  for (auto i = 0; i < rank; ++i) {
    shape_proto->mutable_dim(i)->set_dim_value(shape[i]);
  }
}
vaip_core::DllSafe<std::vector<std::string>> node_arg_get_denotation(const NodeArg& node_arg) {
  auto shape = node_arg.Shape();
  if (shape == nullptr) {
    return vaip_core::DllSafe<std::vector<std::string>>();
  }
  auto ret = std::vector<std::string>();
  auto rank = shape->dim_size();
  for (auto i = 0; i < rank; ++i) {
    ret.push_back(shape->dim(i).denotation());
  }
  return vaip_core::DllSafe<std::vector<std::string>>(ret);
}
void node_arg_set_denotation(const NodeArg& node_arg, const std::vector<std::string>& denotation) {
  auto shape_proto = const_cast<ONNX_NAMESPACE::TensorShapeProto*>(node_arg.Shape());
  assert(shape_proto != nullptr);
  assert(denotation.size() == static_cast<size_t>(shape_proto->dim_size()));
  auto rank = shape_proto->dim_size();
  for (auto i = 0; i < rank; ++i) {
    shape_proto->mutable_dim(i)->set_denotation(denotation[i]);
  }
}
void node_arg_set_element_type(NodeArg& node_arg, int type) {
  if (type < 0 || type > 16) {
    vai_assert(false, "TensorProto::DataType not supoort");
  }
  auto data_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(type);
  auto type_proto = const_cast<ONNX_NAMESPACE::TypeProto*>(node_arg.TypeAsProto());
  assert(type_proto != nullptr);
  auto current_elem_type = type_proto->mutable_tensor_type()->elem_type();
  auto input_elem_type = data_type;
  type_proto->mutable_tensor_type()->set_elem_type(input_elem_type);
  auto status = node_arg.OverrideTypesHelper(*type_proto, input_elem_type,
                                             current_elem_type, true);
  vai_assert(status.IsOK(), status.ErrorMessage());
}
const ONNX_NAMESPACE::TensorProto& node_arg_get_const_data_as_tensor(
    const Graph& graph, const NodeArg& node_arg) {
  auto tensor_proto = graph.GetConstantInitializer(node_arg.Name(), true);
  assert(tensor_proto != nullptr);
  return *tensor_proto;
}
int node_arg_get_element_type(const NodeArg& node_arg) {
  auto type_proto = node_arg.TypeAsProto();
  assert(type_proto != nullptr);
  if (type_proto->value_case() != ONNX_NAMESPACE::TypeProto::kTensorType) {
    return -1;
  }
  return type_proto->tensor_type().elem_type();
}
NodeArg& node_arg_clone(Graph& graph, const NodeArg& node_arg, const std::string& name) {
  vai_assert(name != node_arg.Name(), "node arg must have a new unique name");
  vai_assert(graph.GetNodeArg(name) == nullptr, std::string("node arg " + name + " already exists. "));
  auto type_proto = node_arg.TypeAsProto();
  assert(type_proto != nullptr);
  auto& ret = graph.GetOrCreateNodeArg(name, type_proto);
  return ret;
}
NodeArg& node_arg_new(Graph& graph, const std::string& name, const std::vector<int64_t>* shape, int element_type) {
  vai_assert(graph.GetNodeArg(name) == nullptr, std::string("node arg " + name + " already exists. "));
  auto type_proto = ONNX_NAMESPACE::TypeProto::Create();
  auto tensor_type = type_proto->mutable_tensor_type();
  tensor_type->set_elem_type(element_type);
  if (shape != nullptr) {
    auto shape_proto = tensor_type->mutable_shape();
    for (auto s : *shape) {
      shape_proto->add_dim()->set_dim_value(s);
    }
  } else {
    assert(tensor_type->has_shape() == false);
  }
  return graph.GetOrCreateNodeArg(name, type_proto.release());
}
}  // namespace vaip
