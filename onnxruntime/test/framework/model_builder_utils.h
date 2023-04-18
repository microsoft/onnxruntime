// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"

namespace onnxruntime {
namespace test {

namespace modelbuilder {

// Shape: a wrapper to build a TensorShapeProto
struct Shape {
  ONNX_NAMESPACE::TensorShapeProto value;

  // construct a shape with given constant dimensions
  Shape(std::initializer_list<int> dims) {
    for (auto d : dims) {
      auto dim = value.add_dim();
      dim->set_dim_value(d);
    }
  }

  // construct a shape with given symbolic dimensions
  Shape(std::initializer_list<std::string> dims) {
    for (auto d : dims) {
      auto dim = value.add_dim();
      dim->set_dim_param(d);
    }
  }
};

// Type: a wrapper to build a TypeProto
struct Type {
  ONNX_NAMESPACE::TypeProto value;

  // construct a float-tensor-type
  Type() {
    value.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  }

  // construct a float-tensor-type with given constant dimensions
  Type(std::initializer_list<int> dims) {
    value.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto p_shape = value.mutable_tensor_type()->mutable_shape();
    for (auto d : dims) {
      auto dim = p_shape->add_dim();
      dim->set_dim_value(d);
    }
  }

  // construct a float-tensor-type with given symbolic dimensions
  Type(std::initializer_list<std::string> symbolic_dims) {
    value.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto p_shape = value.mutable_tensor_type()->mutable_shape();
    for (auto d : symbolic_dims) {
      auto dim = p_shape->add_dim();
      dim->set_dim_param(d);
    }
  }

  static Type MakeSequence(const ONNX_NAMESPACE::TypeProto& element_proto) {
    ONNX_NAMESPACE::TypeProto proto;
    proto.mutable_sequence_type()->mutable_elem_type()->CopyFrom(element_proto);
    return Type(std::move(proto));
  }

  static Type MakeMap(ONNX_NAMESPACE::TensorProto_DataType dtype, const ONNX_NAMESPACE::TypeProto& value_proto) {
    ONNX_NAMESPACE::TypeProto proto;
    auto& mut_map = *proto.mutable_map_type();
    mut_map.set_key_type(static_cast<int32_t>(dtype));
    mut_map.mutable_value_type()->CopyFrom(value_proto);
    return Type(std::move(proto));
  }

  static Type MakeOptional(const ONNX_NAMESPACE::TypeProto& contained_proto) {
    ONNX_NAMESPACE::TypeProto proto;
    proto.mutable_optional_type()->mutable_elem_type()->CopyFrom(contained_proto);
    return Type(std::move(proto));
  }

 private:
  explicit Type(ONNX_NAMESPACE::TypeProto type_proto) : value(std::move(type_proto)) {}
};

}  // namespace modelbuilder
}  // namespace test
}  // namespace onnxruntime
