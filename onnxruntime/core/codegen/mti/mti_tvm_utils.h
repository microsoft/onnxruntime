// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <gsl/gsl>
#include <tvm/tvm.h>
#include "core/codegen/mti/common.h"

namespace onnxruntime {
namespace tvm_codegen {

tvm::Array<tvm::Expr> ToTvmArray(gsl::span<const int64_t> shape);

tvm::Array<tvm::Integer> ToTvmArrayInt(gsl::span<const int64_t> shape);

// Helper function to compute sub shape size to axis (not included)
tvm::Expr SizeToDimension(const tvm::Array<tvm::Expr>& shape, int64_t axis);

// Helper function to compute sub shape size from axis (included)
tvm::Expr SizeFromDimension(const tvm::Array<tvm::Expr>& shape, int64_t axis);

// Helper function to align
tvm::Expr RoundUp(tvm::Expr value, tvm::Expr alignment);

tvm::Array<tvm::Expr> ConcatShapes(
    const tvm::Array<tvm::Expr>& shape1,
    const tvm::Array<tvm::Expr>& shape2);

// Helper function to rename tvm::Tensor
tvm::Tensor Rename(tvm::Tensor X, const std::string& name);

// Helper function to slice TVM shape
tvm::Array<tvm::Expr> SliceShape(const tvm::Array<tvm::Expr>& shape, const std::vector<int64_t>& axes);

// Helper function to slice TVM shape from axis (inclusive).
// Basically, this function returns the shape of [axis, shape.size()-1]
tvm::Array<tvm::Expr> SliceShapeFromDimension(const tvm::Array<tvm::Expr>& shape, int64_t axis);

// this function returns the shape of [0, axis-1]
tvm::Array<tvm::Expr> SliceShapeToDimension(const tvm::Array<tvm::Expr>& shape, int64_t axis);

// check if dimension is 1
bool IsOne(const tvm::Array<tvm::Expr>& shape, int64_t axis);

// Helper function to convert tvm::Expr to tvm::Tensor
tvm::Tensor Promote(const tvm::Expr& expr,
                    const tvm::Array<tvm::Expr>& shape,
                    const std::string& name = "PromoteExpr");

tvm::Tensor MakeZeroTensor(const tvm::Array<tvm::Expr>& shape, HalideIR::Type type, const std::string& name);

void DumpTVMModuleToFile(const std::string& filename, tvm::runtime::Module& module);

bool BroadcastDim(const tvm::Array<tvm::Expr>& shape, size_t i, size_t output_rank, tvm::Expr& dim);

inline int64_t HandleNegativeAxis(int64_t axis, int64_t rank) {
  MTI_ASSERT(axis >= -rank && axis <= rank - 1);
  return axis = axis < 0 ? (axis + rank) : axis;
}

// Make sure idx is clamped in the range of [-bound, bound - 1]
tvm::Expr ClampIndex(const tvm::Expr& idx, const tvm::Expr& bound);

// Helper function to workaround tvm ExternOp issue when input has symbolic dimensions
tvm::Array<tvm::Tensor> MakeInputsForExtern(const tvm::Array<tvm::Tensor>& inputs, const std::string& name = "make_inputs_for_extern");

}  //  namespace tvm_codegen
}  //  namespace onnxruntime
