// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Array<tvm::PrimExpr>
ComputeMatMulShape(
    const tvm::Array<tvm::PrimExpr>& A_shape,
    const tvm::Array<tvm::PrimExpr>& B_shape,
    bool trans_a = false,
    bool trans_b = false);

tvm::te::Tensor MatMul2D(const tvm::te::Tensor& A, const tvm::te::Tensor& B, bool trans_a = false, bool trans_b = false, const std::string& name = "matmul2d");

tvm::te::Tensor MatMul(const tvm::te::Tensor& A, const tvm::te::Tensor& B, const std::string& name = "matmul");

}  // namespace tvm_codegen
}  // namespace onnxruntime
