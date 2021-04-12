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

}  // namespace ONNX_NAMESPACE