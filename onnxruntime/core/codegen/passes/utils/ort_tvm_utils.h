// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/framework/data_types.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

class CodeGenContext;

// Helper function that converts a onnxruntime MLDataType to TVM DLDataType
DLDataType ToTvmDLDataType(MLDataType ml_type);

tvm::Type ToTvmType(ONNX_NAMESPACE::TensorProto_DataType proto_type);

tvm::Array<tvm::Expr> ShapeToTvmArray(const NodeArg* def, CodeGenContext& ctx);

tvm::Expr ShapeDimToTvmDim(const ONNX_NAMESPACE::TensorShapeProto_Dimension& dim, CodeGenContext& ctx);

#ifdef CODEGEN_ENABLE_PROFILER
// Helper functions to inspect into lowered function
tvm::Tensor ProfileBegin(tvm::Tensor X, const std::string& event_name);

tvm::Tensor ProfileEnd(tvm::Tensor X, const std::string& event_name);
#endif

}  // namespace tvm_codegen
}  // namespace onnxruntime
