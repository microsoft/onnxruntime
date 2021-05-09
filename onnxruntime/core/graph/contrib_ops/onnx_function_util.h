#pragma once

// Utility functions for building the body of a context-dependent function.
// Temporary placeholder for utilities to be moved into ONNX repo. TODO.

#include <string>
#include <vector>

#include "onnx/onnx-operators_pb.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/function.h"

namespace ONNX_NAMESPACE {

// For floating-value constants of different precision:
TensorProto ToTensor(double value, TensorProto_DataType elem_type);

// Utility function to construct a constant of given type/precision.
inline static FunctionBodyHelper::NodeDef Const(const std::string& name, double value, TensorProto_DataType elem_type) {
  return FunctionBodyHelper::NodeDef{
      {name}, "Constant", {}, {{"value", ToTensor(value, elem_type)}}};
}

// Utility function to construct a FunctionProto from an opschema (for the signature information),
// a sequence of NodeDefs (for the function body), and the relied opsets.
bool BuildFunctionProto(FunctionProto& functionProto,
                        const OpSchema& schema,
                        const std::vector<FunctionBodyHelper::NodeDef>& node_defs,
                        const std::vector<OperatorSetIdProto>& relied_opsets = {});

struct FunctionBuilder {
  const FunctionBodyBuildContext& ctx;
  FunctionBuilder(const FunctionBodyBuildContext& ctx_) : ctx(ctx_) {}

  inline int64_t GetAttrOrDefault(const std::string& attributeName, int64_t defaultValue) const {
    auto attr_proto = ctx.getAttribute(attributeName);
    if ((nullptr != attr_proto) && attr_proto->has_i())
      return attr_proto->i();
    return defaultValue;
  }

  inline float GetAttrOrDefault(const std::string& attributeName, float defaultValue) const {
    auto attr_proto = ctx.getAttribute(attributeName);
    if ((nullptr != attr_proto) && attr_proto->has_f())
      return attr_proto->f();
    return defaultValue;
  }

  inline bool GetElementType(int i, TensorProto_DataType& elem_type) {
    auto* tp = ctx.getInputType(i);
    if ((tp == nullptr) || (!tp->has_tensor_type()))
      return false;
    elem_type = (ONNX_NAMESPACE::TensorProto_DataType)tp->tensor_type().elem_type();
    return true;
  }
};

}  // namespace ONNX_NAMESPACE