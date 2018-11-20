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
};

}  // namespace modelbuilder
}  // namespace test
}  // namespace onnxruntime
