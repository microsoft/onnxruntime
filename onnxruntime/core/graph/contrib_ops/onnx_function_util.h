#pragma once

// Utility functions for building the body of a context-dependent function.
// Temporary placeholder for utilities to be moved into ONNX repo. TODO.

#include <string>
#include <vector>

#include "onnx/onnx-operators_pb.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"

namespace ONNX_NAMESPACE {

// For floating-value constants of different precision:
TensorProto ToTensor(double value, TensorProto_DataType elem_type);

// Utility function to construct a constant of given type/precision.
inline static FunctionBodyHelper::NodeDef Const(const std::string& name, double value, TensorProto_DataType elem_type) {
  return FunctionBodyHelper::NodeDef{
      {name}, "Constant", {}, {{"value", ToTensor(value, elem_type)}}};
}

class FunctionBuilder {
 public:
  FunctionBuilder(FunctionProto& funProto_) : funProto(funProto_) {}

  FunctionBuilder& Add(const char* nodes_txt) {
    OnnxParser parser(nodes_txt);
    auto& nodes = *funProto.mutable_node();

    while (!parser.EndOfInput()) {
      auto status = parser.Parse(*nodes.Add());
      if (!status.IsOK())
        ONNX_THROW_EX(std::logic_error("Error parsing node:" + status.ErrorMessage()));
    }

    return *this;
  }

  FunctionBuilder& Add(const char* node_txt, const AttributeProto& attr) {
    OnnxParser parser(node_txt);
    auto& node = *funProto.add_node();
    auto status = parser.Parse(node);
    if (!status.IsOK()) {
      ONNX_THROW_EX(std::logic_error("Error parsing node:" + status.ErrorMessage()));
    }

    if (!parser.EndOfInput()) {
      ONNX_THROW_EX(std::logic_error("Error unexpected extra input in node:" + status.ErrorMessage()));
    }

    *node.add_attribute() = attr;

    return *this;
  }

  template <typename T>
  FunctionBuilder& Add(const char* node_txt, const std::string& attr_name, T attr_value) {
    return Add(node_txt, MakeAttribute(attr_name, attr_value));
  }

  FunctionBuilder& Const(const std::string& name, double value, int64_t elem_type) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    return Add(constant_op.c_str(), MakeAttribute("value", ToTensor(value, (TensorProto_DataType) elem_type)));
  }

  template <typename T>
  FunctionBuilder& Const(const std::string& name, T const_value) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    return Add (constant_op.c_str(), MakeAttribute("value", ToTensor(const_value)));
  }

  template <typename T>
  FunctionBuilder& Const(const std::string& name, const std::vector<T>& values) {
    std::string constant_op(name);
    constant_op += " = Constant()";
    auto tensor = ToTensor(values);
    tensor.add_dims(values.size());  // Treat as 1D tensor.

    return Add (constant_op.c_str(), MakeAttribute("value", tensor));
  }

  FunctionBuilder& AddOpset(const char* domain, int version) {
    auto* opset = funProto.add_opset_import();
    opset->set_domain(domain);
    opset->set_version(version);
    return *this;
  }

 private:
  FunctionProto& funProto;
};

}  // namespace ONNX_NAMESPACE