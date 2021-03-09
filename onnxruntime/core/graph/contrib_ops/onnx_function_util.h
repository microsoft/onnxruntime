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

bool BuildFunctionProto(FunctionProto& functionProto, const OpSchema& schema, const std::vector<FunctionBodyHelper::NodeDef>& node_defs);

}  // namespace ONNX_NAMESPACE