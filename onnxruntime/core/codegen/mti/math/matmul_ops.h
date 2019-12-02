// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Array<tvm::Expr>
ComputeMatMulShape(
    const tvm::Array<tvm::Expr>& A_shape,
    const tvm::Array<tvm::Expr>& B_shape,
    bool trans_a = false,
    bool trans_b = false);

tvm::Tensor MatMul2D(const tvm::Tensor& A, const tvm::Tensor& B, bool trans_a = false, bool trans_b = false, const std::string& name = "matmul2d");

tvm::Tensor MatMul(const tvm::Tensor& A, const tvm::Tensor& B, const std::string& name = "matmul");

}  // namespace tvm_codegen
}  // namespace onnxruntime
