// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/framework/data_types.h"
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

class CodeGenContext;

// Helper function that converts a onnxruntime MLDataType to TVM DLDataType
DLDataType ToTvmDLDataType(MLDataType ml_type);

tvm::DataType ToTvmType(ONNX_NAMESPACE::TensorProto_DataType proto_type);

tvm::Array<tvm::PrimExpr> ShapeToTvmArray(const NodeArg* def, CodeGenContext& ctx);

 tvm::PrimExpr ShapeDimToTvmDim(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim, CodeGenContext& ctx);

#ifdef CODEGEN_ENABLE_PROFILER
// Helper functions to inspect into lowered function
tvm::te::Tensor ProfileBegin(tvm::te::Tensor X, const std::string& event_name);

tvm::te::Tensor ProfileEnd(tvm::te::Tensor X, const std::string& event_name);
#endif

}  // namespace tvm_codegen
}  // namespace onnxruntime
