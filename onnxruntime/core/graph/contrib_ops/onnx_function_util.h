#pragma once

// Utility functions for building the body of a context-dependent function.
// Temporary placeholder for utilities to be moved into ONNX repo. TODO.

#include <string>
#include <vector>

#include "core/graph/onnx_protobuf.h"
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
}  // namespace ONNX_NAMESPACE
