// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/node_arg.h"
#include "vaip/vai_assert.h"

#include <cstdint>

#include "./tensor_proto.h"
#include "core/graph/node_arg.h"

namespace vaip {

bool node_arg_is_exists(const NodeArg& node_arg) {
  return node_arg.Exists();
}
bool node_arg_is_constant(const Graph& graph, const NodeArg& node_arg) {
  assert(node_arg.Exists());
  assert(!node_arg.Name().empty());
  auto constant_tensor_proto =
      graph.GetConstantInitializer(node_arg.Name(), true);
  return constant_tensor_proto != nullptr;
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

static void LayoutTransformRule_set_shape(onnx::TensorShapeProto& shape_proto,
                                          const std::vector<int64_t>& shape) {
  assert(shape.size() == static_cast<size_t>(shape_proto.dim_size()));
  auto rank = shape_proto.dim_size();
  for (auto i = 0; i < rank; ++i) {
    shape_proto.mutable_dim(i)->set_dim_value(shape[i]);
  }
}

static void LayoutTransformRule_set_shape(onnx::TypeProto& type_proto,
                                          const std::vector<int64_t>& shape) {
  assert(type_proto.value_case() == onnx::TypeProto::kTensorType);
  //<< type_proto.DebugString();
  auto& tensor_type = *type_proto.mutable_tensor_type();
  auto& shape_prot = *tensor_type.mutable_shape();
  return LayoutTransformRule_set_shape(shape_prot, shape);
}

static void LayoutTransformRule_set_shape(NodeArg* node_arg,
                                          const std::vector<int64_t>& shape) {
  assert(node_arg != nullptr);
  auto* type_proto = node_arg->TypeAsProto();
  assert(type_proto != nullptr);
  return LayoutTransformRule_set_shape(
      *const_cast<onnx::TypeProto*>(type_proto), shape);
}

void node_arg_set_shape_i64(const NodeArg& node_arg,
                            const std::vector<int64_t>& shape) {
  LayoutTransformRule_set_shape(const_cast<NodeArg*>(&node_arg), shape);
}

static std::vector<std::string> LayoutTransformRule_get_denotation(
    const onnx::TensorShapeProto& shape) {
  auto ret = std::vector<std::string>();
  auto rank = shape.dim_size();
  ret.reserve(rank);
  for (auto i = 0; i < rank; ++i) {
    auto& d = shape.dim(i).denotation();
    ret.push_back(d);
  }
  return ret;
}

static vaip_core::DllSafe<std::vector<std::string>> LayoutTransformRule_get_denotation(
    const onnx::TypeProto& type_proto) {
  vai_assert(type_proto.value_case() == onnx::TypeProto::kTensorType, type_proto.DebugString());
  auto& tensor_type = type_proto.tensor_type();
  if (!tensor_type.has_shape()) {
    return vaip_core::DllSafe<std::vector<std::string>>();
  }
  auto& shape = tensor_type.shape();
  auto denotation = LayoutTransformRule_get_denotation(shape);
  return vaip_core::DllSafe<std::vector<std::string>>(denotation);
}

static vaip_core::DllSafe<std::vector<std::string>> LayoutTransformRule_get_denotation(
    const NodeArg* node_arg) {
  assert(node_arg != nullptr);
  auto* type_proto = node_arg->TypeAsProto();
  assert(type_proto != nullptr);
  return LayoutTransformRule_get_denotation(*type_proto);
}

vaip_core::DllSafe<std::vector<std::string>> node_arg_get_denotation(const NodeArg& node_arg) {
  return LayoutTransformRule_get_denotation(&node_arg);
}

static onnx::TensorShapeProto* node_arg_get_tensor_mutable_shape(
    NodeArg* node_arg) {
  assert(node_arg != nullptr);
  auto type_proto = const_cast<onnx::TypeProto*>(node_arg->TypeAsProto());
  assert(type_proto != nullptr);
  vai_assert(type_proto->value_case() == onnx::TypeProto::kTensorType,
             type_proto->DebugString());
  return type_proto->mutable_tensor_type()->mutable_shape();
}

static void LayoutTransformRule_set_denotation(
    onnx::TensorShapeProto& shape, const std::vector<std::string>& denotation) {
  assert(denotation.size() == static_cast<size_t>(shape.dim_size()));
  auto rank = shape.dim_size();
  for (auto i = 0; i < rank; ++i) {
    shape.mutable_dim(i)->set_denotation(denotation[i]);
  }
}
void node_arg_set_denotation(const NodeArg& node_arg,
                             const std::vector<std::string>& denotation) {
  auto mutable_shape =
      node_arg_get_tensor_mutable_shape(const_cast<NodeArg*>(&node_arg));

  return LayoutTransformRule_set_denotation(*mutable_shape, denotation);
}

void node_arg_set_element_type(NodeArg& node_arg,
                               onnx::TensorProto::DataType data_type) {
  auto type_proto = const_cast<onnx::TypeProto*>(node_arg.TypeAsProto());
  assert(type_proto != nullptr);
  auto current_elem_type = type_proto->mutable_tensor_type()->elem_type();
  auto input_elem_type = data_type;
  type_proto->mutable_tensor_type()->set_elem_type(input_elem_type);
  auto status = node_arg.OverrideTypesHelper(*type_proto, input_elem_type,
                                             current_elem_type, true);
  vai_assert(status.IsOK(), status.ErrorMessage());
}
void node_arg_set_shape(NodeArg& node_arg, std::vector<int64_t> shape) {
  auto type_proto = const_cast<onnx::TypeProto*>(node_arg.TypeAsProto());
  assert(type_proto != nullptr);
  for (auto i = 0u; i < shape.size(); i++) {
    type_proto->mutable_tensor_type()
        ->mutable_shape()
        ->mutable_dim(i)
        ->set_dim_value(shape[i]);
  }
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

NodeArg& node_arg_clone(Graph& graph, const NodeArg& node_arg,
                        const std::string& name) {
  vai_assert(name != node_arg.Name(), "node arg must have a new unique name");
  vai_assert(graph.GetNodeArg(name) == nullptr, std::string("node arg " + name + " already exists. "));
  auto type_proto = node_arg.TypeAsProto();
  assert(type_proto != nullptr);
  auto& ret = graph.GetOrCreateNodeArg(name, type_proto);
  return ret;
}

NodeArg& node_arg_new(Graph& graph,
                      const std::string& name, const std::vector<int64_t>* shape, int element_type) {
  vai_assert(graph.GetNodeArg(name) == nullptr, std::string("node arg " + name + " already exists. "));
  auto type_proto = onnx::TypeProto();
  auto tensor_type = type_proto.mutable_tensor_type();
  tensor_type->set_elem_type(element_type);
  if (shape != nullptr) {
    auto shape_proto = tensor_type->mutable_shape();
    for (auto s : *shape) {
      shape_proto->add_dim()->set_dim_value(s);
    }
  } else {
    assert(tensor_type->has_shape() == false);
  }
  auto& ret = graph.GetOrCreateNodeArg(name, &type_proto);
  return ret;
}

}  // namespace vaip
