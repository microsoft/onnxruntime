// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <tvm/te/operation.h>
#include <tvm/relay/expr.h>
#include "core/codegen/mti/common.h"

namespace onnxruntime {
namespace tvm_codegen {

tvm::Array<tvm::PrimExpr> ToTvmArray(const std::vector<int64_t>& shape);

tvm::Array<tvm::Integer> ToTvmArrayInt(const std::vector<int64_t>& shape);

// Helper function to compute sub shape size to axis (not included)
 tvm::PrimExpr SizeToDimension(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis);

// Helper function to compute sub shape size from axis (included)
 tvm::PrimExpr SizeFromDimension(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis);

// Helper function to align
 tvm::PrimExpr RoundUp( tvm::PrimExpr value,  tvm::PrimExpr alignment);

tvm::Array<tvm::PrimExpr> ConcatShapes(
    const tvm::Array<tvm::PrimExpr>& shape1,
    const tvm::Array<tvm::PrimExpr>& shape2);

// Helper function to rename tvm::te::Tensor
tvm::te::Tensor Rename(tvm::te::Tensor X, const std::string& name);

// Helper function to slice TVM shape
tvm::Array<tvm::PrimExpr> SliceShape(const tvm::Array<tvm::PrimExpr>& shape, const std::vector<int64_t>& axes);

// Helper function to slice TVM shape from axis (inclusive).
// Basically, this function returns the shape of [axis, shape.size()-1]
tvm::Array<tvm::PrimExpr> SliceShapeFromDimension(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis);

// this function returns the shape of [0, axis-1]
tvm::Array<tvm::PrimExpr> SliceShapeToDimension(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis);

// check if dimension is 1
bool IsOne(const tvm::Array<tvm::PrimExpr>& shape, int64_t axis);

// Helper function to convert  tvm::PrimExpr to tvm::te::Tensor
tvm::te::Tensor Promote(const  tvm::PrimExpr& expr,
                    const tvm::Array<tvm::PrimExpr>& shape,
                    const std::string& name = "PromoteExpr");

tvm::te::Tensor MakeZeroTensor(const tvm::Array<tvm::PrimExpr>& shape, tvm::runtime::DataType type, const std::string& name);

void DumpTVMModuleToFile(const std::string& filename, tvm::runtime::Module& module);

bool BroadcastDim(const tvm::Array<tvm::PrimExpr>& shape, size_t i, size_t output_rank,  tvm::PrimExpr& dim);

inline int64_t HandleNegativeAxis(int64_t axis, int64_t rank) {
  MTI_ASSERT(axis >= -rank && axis <= rank - 1);
  return axis = axis < 0 ? (axis + rank) : axis;
}

// Make sure idx is clamped in the range of [-bound, bound - 1]
 tvm::PrimExpr ClampIndex(const  tvm::PrimExpr& idx, const  tvm::PrimExpr& bound);

// Helper function to workaround tvm ExternOp issue when input has symbolic dimensions
tvm::Array<tvm::te::Tensor> MakeInputsForExtern(const tvm::Array<tvm::te::Tensor>& inputs, const std::string& name = "make_inputs_for_extern");

}  //  namespace tvm_codegen
}  //  namespace onnxruntime
